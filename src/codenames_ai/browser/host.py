"""The arena's own host: a spectator/admin context that creates and starts the room.

It never joins a team, so it is never one of the four AI players.
"""

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

from playwright.async_api import BrowserContext, Page
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from . import selectors as s
from .client import BrowserClient
from .errors import BrowserIntegrationError
from .identity import dismiss_guide, enter_game

logger = logging.getLogger(__name__)

HOME_URL = "https://codenames.game/"
HOST_NICKNAME = "ArenaHost"
START_GAME = re.compile(r"^\s*start game\s*$", re.IGNORECASE)
MEMBERS = 'button[aria-haspopup="dialog"]'


# Observed live: the home page redirects a new host to https://codenames.game/r/<code>.
ROOM_PATH = re.compile(r"^/r/[^/]+/?$")


def is_room_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname == "codenames.game" and bool(ROOM_PATH.match(parsed.path))


class RoomCreationError(BrowserIntegrationError):
    """Room creation stopped before the lobby; carries what the page looked like."""

    def __init__(
        self, message: str, *, stage: str, details: dict[str, Any], screenshot: bytes | None
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.details = details
        self.screenshot = screenshot


class Host:
    def __init__(self, browser: BrowserClient, nickname: str = HOST_NICKNAME) -> None:
        self.browser = browser
        self.nickname = nickname
        self.context: BrowserContext | None = None
        self.room_url: str | None = None

    @property
    def page(self) -> Page:
        if self.context is None:
            raise RuntimeError("Host has no open room")
        return self.context.pages[0]

    async def create_room(self, *, timeout: float = 30) -> str:
        """Enter the home page with the host nickname and return the new lobby's URL.

        Each step is checked on the page itself: the nickname is in the field before
        submitting, the address becomes /r/<code>, and the lobby's role panels render
        with no nickname prompt left open.
        """
        if self.context is not None:
            raise RuntimeError("Host already created a room")
        self.context = await self.browser.open_room(HOME_URL)
        page = self.page
        entry = page.locator(s.ENTRY_NICKNAME)
        ms = timeout * 1000
        stage = "nickname_field"
        try:
            await entry.wait_for(state="visible", timeout=ms)
            stage = "nickname_entry"
            await enter_game(page, self.nickname, timeout_ms=ms)
            stage = "navigate_to_room"
            await page.wait_for_url(is_room_url, timeout=ms)
            stage = "lobby_render"
            # The role panels mark a rendered lobby, not just a changed address.
            await page.locator(s.BLUE_SPYMASTER).wait_for(state="visible", timeout=ms)
            if await entry.is_visible():
                # Some paths ask again in a "Welcome to Codenames" dialog in the room.
                stage = "room_welcome"
                await enter_game(page, self.nickname, timeout_ms=ms)
                await entry.wait_for(state="hidden", timeout=ms)
        except (PlaywrightTimeoutError, BrowserIntegrationError) as exc:
            details = await self.diagnose()
            details["error"] = str(exc).splitlines()[0][:300]
            raise RoomCreationError(
                f"[{self.nickname}] could not create a room (stage {stage}, now at "
                f"{details['url']}): {details['error']}",
                stage=stage,
                details=details,
                screenshot=await self._screenshot(),
            ) from exc
        self.room_url = page.url
        logger.info("[%s] created room %s", self.nickname, self.room_url)
        return self.room_url

    async def diagnose(self) -> dict[str, Any]:
        """Page facts for a startup failure log; every probe is best effort."""
        page = self.page
        details: dict[str, Any] = {"url": page.url}
        entry = page.locator(s.ENTRY_NICKNAME)
        probes: list[tuple[str, Callable[[], Awaitable[Any]]]] = [
            ("title", page.title),
            ("nickname_fields", entry.count),
            ("nickname_visible", lambda: entry.first.is_visible()),
            ("nickname_value", lambda: entry.first.input_value(timeout=1000)),
            (
                "enter_game_buttons",
                page.get_by_role("button", name="Enter Game", exact=True).count,
            ),
            ("role_panels", page.locator(", ".join(s.ROLE_PANELS.values())).count),
            (
                "dialogs",
                lambda: page.get_by_role("dialog").filter(visible=True).all_inner_texts(),
            ),
        ]
        for name, probe in probes:
            try:
                value = await probe()
            except PlaywrightError as exc:
                value = f"unavailable ({type(exc).__name__})"
            if name == "dialogs" and isinstance(value, list):
                value = [" ".join(text.split())[:200] for text in value]
            details[name] = value
        return details

    async def _screenshot(self) -> bytes | None:
        try:
            return await self.page.screenshot(timeout=5000)
        except PlaywrightError:
            return None

    async def wait_for_players(self, *, timeout: float = 15) -> None:
        """All four role panels show a member in the host's own view of the lobby."""
        try:
            await self.page.wait_for_function(
                """([selectors, members]) => selectors.every(selector => {
                    const panel = document.querySelector(selector);
                    return panel && panel.querySelectorAll(members).length > 0;
                })""",
                arg=[list(s.ROLE_PANELS.values()), MEMBERS],
                timeout=timeout * 1000,
            )
        except PlaywrightTimeoutError as exc:
            raise BrowserIntegrationError(
                "The host does not see all four players in the lobby"
            ) from exc
        for selector in s.ROLE_PANELS.values():
            names = await self.page.locator(selector).locator(MEMBERS).all_text_contents()
            if self.nickname in (name.strip() for name in names):
                raise BrowserIntegrationError(f"Host {self.nickname} appears in a role panel")

    async def start_game(self, *, timeout: float = 15) -> None:
        """Click Start game once; the caller waits for the board to go live."""
        await dismiss_guide(self.page)
        button = self.page.get_by_role("button", name=START_GAME)
        try:
            await button.wait_for(state="visible", timeout=timeout * 1000)
            await button.click(timeout=timeout * 1000)
        except PlaywrightTimeoutError as exc:
            raise BrowserIntegrationError(
                f"[{self.nickname}] could not click Start game; is this context the room admin?"
            ) from exc
        logger.info("[%s] clicked Start game", self.nickname)

    async def close(self) -> None:
        if self.context is not None:
            try:
                await self.context.close()
            except PlaywrightError as exc:  # already disconnected
                logger.warning("Host context cleanup: %s", exc)
            self.context = None
