import logging

from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from codenames_ai.domain.models import PlayerAssignment

from . import selectors as s
from .errors import PlayerAssignmentError

logger = logging.getLogger(__name__)


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
        await entry.click()
        await entry.fill("")
        await entry.type(requested_name, delay=20)
        await page.get_by_role("button", name="Enter Game", exact=True).click()
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
