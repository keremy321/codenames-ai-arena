import pytest
from playwright.async_api import BrowserContext

import codenames_ai.browser.arena as arena_module
from codenames_ai.browser.arena import Arena
from codenames_ai.browser.client import BrowserClient


async def test_four_isolated_contexts_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    async def joined(page: object, assignment: object) -> None:
        return None

    monkeypatch.setattr(arena_module, "join_room", joined)
    async with BrowserClient(headless=True) as client:
        assert client._browser is not None
        original = client._browser.new_context

        async def offline() -> BrowserContext:
            context = await original()
            await context.route(
                "**/*",
                lambda route: route.fulfill(
                    body="<html>Offline room</html>", content_type="text/html"
                ),
            )
            return context

        monkeypatch.setattr(client._browser, "new_context", offline)
        arena = Arena(client)
        await arena.open("https://codenames.game/room/offline")
        assert len(arena.players) == 4
        assert len(client._browser.contexts) == 4
        for i, player in enumerate(arena.players.values()):
            page = player.context.pages[0]
            assert await page.evaluate("sessionStorage.getItem('identity')") is None
            assert await page.evaluate("localStorage.getItem('identity')") is None
            assert await player.context.cookies() == []
            await page.evaluate(
                "i=>{sessionStorage.setItem('identity',i);localStorage.setItem('identity',i)}",
                str(i),
            )
            await player.context.add_cookies(
                [{"name": "identity", "value": str(i), "url": "https://codenames.game"}]
            )
        contexts = [p.context for p in arena.players.values()]
        await arena.close()
        assert not client._browser.contexts
        assert all(not c.pages for c in contexts)


async def test_partial_open_failure_closes_existing_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def joined(page: object, assignment: object) -> None:
        return None

    monkeypatch.setattr(arena_module, "join_room", joined)

    class FakeContext:
        def __init__(self) -> None:
            self.pages = [object()]
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    context = FakeContext()

    class FailingClient:
        calls = 0

        async def open_room(self, room: str) -> FakeContext:
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("connection failed")
            return context

    arena = Arena(FailingClient())
    with pytest.raises(RuntimeError, match="connection failed"):
        await arena.open("https://codenames.game/room/offline")
    assert context.closed and not arena.players
