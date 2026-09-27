from types import TracebackType
from typing import Self
from urllib.parse import urlparse

from playwright.async_api import Browser, BrowserContext, Playwright, async_playwright


class BrowserClient:
    """One browser; a fresh isolated context for every open_room call."""

    def __init__(self, *, headless: bool = False) -> None:
        self.headless = headless
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    async def __aenter__(self) -> Self:
        self._playwright = await async_playwright().start()
        try:
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
        except BaseException:
            await self._playwright.stop()
            raise
        return self

    async def open_room(self, room_url: str) -> BrowserContext:
        parsed = urlparse(room_url)
        if parsed.scheme != "https" or parsed.hostname != "codenames.game":
            raise ValueError("Room URL must use https://codenames.game/")
        if self._browser is None:
            raise RuntimeError("Use BrowserClient as an async context manager")
        context = await self._browser.new_context()
        try:
            page = await context.new_page()
            await page.goto(room_url, wait_until="domcontentloaded")
        except BaseException:
            await context.close()
            raise
        return context

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if self._browser:
                await self._browser.close()
        finally:
            if self._playwright:
                await self._playwright.stop()
