from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from playwright.async_api import Browser, BrowserContext, async_playwright

from codenames_ai.browser.reader import DOMParseError, GameReader
from codenames_ai.domain.enums import CardColor, Role, Team
from codenames_ai.main import format_reading


@pytest.fixture
async def browser() -> AsyncIterator[Browser]:
    async with async_playwright() as playwright:
        instance = await playwright.chromium.launch(headless=True)
        yield instance
        await instance.close()


async def test_real_dom_reader_and_information_boundary(
    browser: Browser,
    caplog: pytest.LogCaptureFixture,
) -> None:
    context = await browser.new_context()
    page = await context.new_page()
    await page.set_content(Path("tests/fixtures/board.html").read_text(encoding="utf-8"))
    reader = GameReader(page)
    await reader.wait_for_board(timeout_ms=1000)
    operative = await reader.read_operative_state(Team.BLUE)
    assert len(operative.cards) == 3
    assert operative.cards[0].word == "MOON"
    assert all(card.color is None for card in operative.cards)
    assert operative.clue.word == "SPACE"
    assert operative.clue.number == 2
    assert "--CardColor" not in caplog.text
    reading = await reader.read(Team.BLUE, Role.OPERATIVE)
    assert reading.turn.team == Team.RED
    assert reading.state.team == Team.BLUE
    assert "UNKNOWN" not in format_reading(reading)
    assert "  RED" not in format_reading(reading)
    spy = await reader.read_spymaster_state(Team.BLUE)
    assert [c.color for c in spy.cards] == [CardColor.BLUE, CardColor.RED, CardColor.UNKNOWN]
    assert "Unable to determine card color" in caplog.text
    await page.set_content("<div>No game</div>")
    with pytest.raises(DOMParseError):
        await reader.read_operative_state(Team.BLUE)
    await context.close()


async def test_browser_client_context_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    from codenames_ai.browser.client import BrowserClient

    async with BrowserClient(headless=True) as client:
        # Route every request locally; this test never contacts the real website.
        assert client._browser is not None
        original_new_context = client._browser.new_context

        async def routed_context() -> BrowserContext:
            context = await original_new_context()
            await context.route(
                "**/*",
                lambda route: route.fulfill(
                    body="<html><body>Offline room</body></html>", content_type="text/html"
                ),
            )
            return context

        monkeypatch.setattr(client._browser, "new_context", routed_context)
        first = await client.open_room("https://codenames.game/room/offline")
        second = await client.open_room("https://codenames.game/room/offline")
        await first.add_cookies(
            [{"name": "identity", "value": "blue", "url": "https://codenames.game"}]
        )
        await first.pages[0].evaluate("sessionStorage.setItem('identity', 'blue')")
        assert await second.cookies() == []
        assert await second.pages[0].evaluate("sessionStorage.getItem('identity')") is None
