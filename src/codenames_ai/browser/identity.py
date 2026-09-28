import asyncio
import logging

from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from codenames_ai.domain.models import PlayerAssignment

from . import selectors as s
from .errors import PlayerAssignmentError

logger = logging.getLogger(__name__)

# The site sometimes submits the welcome form before it has registered the typed
# nickname; a short pause between typing and "Enter Game" lets it catch up.
NICKNAME_CONFIRM_DELAY = 1.0


async def open_player_settings(page: Page) -> None:
    if await page.locator(s.NICKNAME_INPUT).is_visible():
        return
    settings = page.locator(s.SETTINGS_BUTTON)
    if await settings.get_attribute("aria-expanded") != "true":
        await settings.click(timeout=5000)
    # Hosts initially see the Admin tab; other users initially see Player.
    tab = (
        page.get_by_role("dialog")
        .get_by_role("button", name="Player", exact=True)
        .filter(visible=True)
    )
    if await tab.count() == 1:
        await tab.click(timeout=5000)
    await page.locator(s.NICKNAME_INPUT).wait_for(timeout=5000)


async def dismiss_guide(page: Page) -> None:
    """Dismiss only the verified instructional overlay, never a game dialog."""
    # Observed after a headed browser sits idle while a local model is thinking.
    idle = page.get_by_role("dialog").filter(has_text="Are you still there?")
    if await idle.count() == 1 and await idle.is_visible():
        button = idle.get_by_role("button", name="I'm here!", exact=True)
        if await button.count() == 1:
            await button.click(timeout=3000)
    switch = page.get_by_role("switch", name="Auto-show", exact=True)
    if await switch.count() == 1 and await switch.is_visible():
        await switch.uncheck(timeout=3000)
    close = page.get_by_role("button", name="Close tip", exact=True)
    if await close.count() == 1 and await close.is_visible():
        await close.click(timeout=3000)


async def detect_assignment(
    page: Page, expected: PlayerAssignment | None = None
) -> PlayerAssignment:
    """Match the local Settings nickname to exactly one role panel.

    Admins can open other players' avatars: avatar buttons alone are NOT proof
    of local identity. Top-level Settings is the local-user source.
    """
    local_name = await get_local_nickname(page)
    matches = []
    for (team, role), selector in s.ROLE_PANELS.items():
        names = (
            await page.locator(selector)
            .locator('button[aria-haspopup="dialog"]')
            .all_text_contents()
        )
        count = sum(name.strip() == local_name for name in names)
        if count == 1:
            matches.append(PlayerAssignment(team=team, role=role, nickname=local_name))
        elif count > 1:
            raise PlayerAssignmentError("Duplicate nickname in role panel")
    if len(matches) != 1:
        raise PlayerAssignmentError("Local nickname is absent or ambiguous across role panels")
    assignment = matches[0]
    if expected is not None and (
        assignment.team != expected.team or assignment.role != expected.role
    ):
        raise PlayerAssignmentError(
            f"Expected {expected.team}/{expected.role} ({expected.nickname}), "
            f"found {assignment.team}/{assignment.role} ({assignment.nickname})"
        )
    logger.info("[%s] verified %s_%s", local_name, assignment.team, assignment.role)
    return assignment


async def get_local_nickname(page: Page) -> str:
    await dismiss_guide(page)
    try:
        await open_player_settings(page)
        nickname = page.locator(s.NICKNAME_INPUT)
        await nickname.wait_for(timeout=5000)
        local_name = (await nickname.input_value()).strip()
    except PlaywrightTimeoutError as exc:
        raise PlayerAssignmentError("Cannot read local Settings > Nickname; run inspect") from exc
    finally:
        if await page.locator(s.NICKNAME_INPUT).is_visible():
            await page.keyboard.press("Escape")
    if not local_name:
        raise PlayerAssignmentError("Local Settings nickname is empty")
    return local_name


class NicknameEntryError(PlayerAssignmentError):
    """The welcome form never held the requested nickname long enough to submit it."""


async def enter_game(page: Page, nickname: str, *, timeout_ms: float = 10_000) -> None:
    """Type the nickname into the welcome form (home page, room modal, or welcome dialog),
    pause, check it is still there, then press that form's Enter Game.

    The live site renders the form on the server and hydrates it later: text typed
    before hydration can be wiped when the client re-renders the field. The pause
    doubles as that race's window, so a reset is detected and entry retried once.
    """
    # Hydration scripts have run by the load event; the field then stops being replaced.
    await page.wait_for_load_state("load", timeout=timeout_ms)
    entry = page.locator(s.ENTRY_NICKNAME)
    value = ""
    for attempt in (1, 2):
        try:
            await entry.wait_for(state="visible", timeout=timeout_ms)
            await entry.click(timeout=timeout_ms)  # focus; waits until visible and enabled
            await entry.fill("", timeout=timeout_ms)  # clear; waits until editable
            await entry.press_sequentially(nickname, delay=40, timeout=timeout_ms)
        except PlaywrightTimeoutError as exc:
            raise NicknameEntryError(f"Nickname field was not ready: {exc}") from exc
        await asyncio.sleep(NICKNAME_CONFIRM_DELAY)
        value = await entry.input_value(timeout=timeout_ms)
        if value == nickname:
            break
        logger.warning(
            "Nickname field holds %r instead of %r (attempt %d); retyping",
            value,
            nickname,
            attempt,
        )
    else:
        raise NicknameEntryError(
            f"Nickname field was reset before submit (holds {value!r}, wanted {nickname!r})"
        )
    # The Enter Game of the form or dialog that owns this field, not any other one.
    owner = entry.locator("xpath=ancestor::*[self::form or @role='dialog'][1]")
    scope = owner if await owner.count() == 1 else page
    await scope.get_by_role("button", name="Enter Game", exact=True).click(timeout=timeout_ms)


async def join_room(page: Page, player: PlayerAssignment) -> None:
    """Enter the room and join only the assigned role panel."""
    entry = page.locator(s.ENTRY_NICKNAME)
    role_selector = s.ROLE_PANELS[(player.team, player.role)]
    panel = page.locator(role_selector)
    try:
        await entry.wait_for(state="visible", timeout=10_000)
    except PlaywrightTimeoutError:
        # An already entered context has no welcome modal.
        if not await panel.is_visible():
            raise PlayerAssignmentError(
                f"[{player.nickname}] nickname entry and role lobby did not appear"
            ) from None
    else:
        requested_name = {
            ("blue", "spymaster"): "BSpy",
            ("blue", "operative"): "BOp",
            ("red", "spymaster"): "RSpy",
            ("red", "operative"): "ROp",
        }[(player.team.value, player.role.value)]
        await enter_game(page, requested_name)
        try:
            await entry.wait_for(state="hidden", timeout=15_000)
        except PlaywrightTimeoutError as exc:
            raise PlayerAssignmentError(f"[{player.nickname}] could not enter the room") from exc
    logger.info("[%s] entered room", player.nickname)

    try:
        await panel.wait_for(state="visible", timeout=10_000)
        members = panel.locator('button[aria-haspopup="dialog"]')
        before = await members.count()
        join = panel.locator(s.JOIN_TEAM)
        await join.wait_for(state="visible", timeout=10_000)
        await join.click(timeout=10_000)
        await page.wait_for_function(
            """([selector, before]) => {
                const panel = document.querySelector(selector);
                return panel && panel.querySelectorAll('button[aria-haspopup="dialog"]').length > before;
            }""",
            arg=[role_selector, before],
            timeout=15_000,
        )
    except (PlaywrightTimeoutError, PlayerAssignmentError) as exc:
        count = await panel.locator(s.JOIN_TEAM).count()
        try:
            panel_text = (await panel.inner_text(timeout=1_000)).strip()[:200]
        except PlaywrightTimeoutError:
            panel_text = "<panel unavailable>"
        raise PlayerAssignmentError(
            f"[{player.nickname}] could not join {player.team.value} {player.role.value} panel: "
            f"{exc}; JOIN TEAM button count={count}; panel text={panel_text!r}"
        ) from exc
    logger.info(
        "[%s] joined %s %s ✓",
        player.nickname,
        player.team.value,
        player.role.value,
    )
