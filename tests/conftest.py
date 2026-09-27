from collections.abc import AsyncIterator

import pytest
from playwright.async_api import Page, async_playwright


@pytest.fixture
async def live_page() -> AsyncIterator[Page]:
    """Offline Chromium page for testing captured live DOM snippets."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context()
        await context.route("**/*", lambda route: route.abort())
        yield await context.new_page()
        await browser.close()
