import argparse
import builtins
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import pytest
from boards import spymaster_state
from playwright.async_api import BrowserContext, Page, Route

import codenames_ai.browser.arena as arena_module
import codenames_ai.main as main_module
from codenames_ai.browser.arena import Arena
from codenames_ai.browser.client import BrowserClient
from codenames_ai.browser.errors import BrowserIntegrationError
from codenames_ai.browser.host import Host, RoomCreationError, is_room_url
from codenames_ai.browser.identity import NICKNAME_CONFIRM_DELAY
from codenames_ai.browser.selectors import ROLE_PANELS
from codenames_ai.config import Settings
from codenames_ai.domain.enums import GamePhase, Role, Team
from codenames_ai.domain.models import PlayerAssignment, SpymasterGameState
from codenames_ai.llm.base import LLMConfig
from codenames_ai.llm.preflight import WarmUp
from codenames_ai.recording import RecordedOperative, RecordedSpymaster

ROOM = "https://codenames.game/r/radun-kapuf"
SLOTS = [selector.split('"')[1] for selector in ROLE_PANELS.values()]
# What the fake site does; reset for every test by the offline_client fixture.
SITE: dict[str, Any] = {}

# A home page whose nickname field can be reset the way late hydration resets it:
#   replace - the field is swapped for a fresh empty one while typing (keys are lost)
#   clear   - the field's value is emptied during the pause before submitting
#   always  - cleared on every attempt
#   stuck   - submitting never navigates
HOME = """<form id="welcome"><label for="nickname">Enter your nickname</label>
<input id="nickname" name="nickname" type="text"><button type="submit">Enter Game</button></form>
<script>
  const mode = MODE;
  let resets = 0;
  function wire(entry) {
    entry.addEventListener('keyup', () => window.typedAt = Date.now());
    entry.addEventListener('input', () => {
      if (window.pending || mode === 'none' || mode === 'stuck') return;
      if (resets >= 1 && mode !== 'always') return;
      window.pending = true;
      const done = () => { resets++; window.resets = resets; window.pending = false; };
      if (mode === 'replace') setTimeout(() => {
        const fresh = entry.cloneNode(false);
        fresh.value = '';
        entry.replaceWith(fresh);
        wire(fresh);
        done();
      }, 100);
      else setTimeout(() => { entry.value = ''; done(); }, 600);
    });
  }
  wire(document.querySelector('#nickname'));
  document.querySelector('#welcome').onsubmit = event => {
    event.preventDefault();
    const entry = document.querySelector('#nickname');
    sessionStorage.setItem('entry', JSON.stringify(
      {name: entry.value, typedAt: window.typedAt, submittedAt: Date.now(), resets}));
    if (mode !== 'stuck') location.href = '/r/radun-kapuf';
  };
</script>"""

ROOM_WELCOME = """<div role="dialog" id="room-welcome"><h2>Welcome to Codenames</h2>
<form><label for="nickname">Enter your nickname</label><input id="nickname" type="text">
<button type="submit">Enter Game</button></form></div>
<script>
  document.querySelector('#room-welcome form').onsubmit = event => {
    event.preventDefault();
    window.roomWelcomeName = document.querySelector('#nickname').value;
    document.querySelector('#room-welcome').remove();
  };
</script>"""

LOBBY = (
    "".join(
        f'<section data-match-slot="{slot}"><span>ROLE</span>'
        '<button class="joinTeam">JOIN TEAM</button></section>'
        for slot in SLOTS
    )
    + """<button id="start">Start game</button>
<script>
  document.querySelector('#start').onclick = () => {
    window.startedWith = [...document.querySelectorAll('[data-match-slot]')]
      .map(panel => panel.querySelectorAll('button[aria-haspopup="dialog"]').length);
  };
</script>"""
)


async def site(route: Route) -> None:
    path = urlparse(route.request.url).path
    if path == "/":
        body = HOME.replace("MODE", json.dumps(SITE["home"]))
    else:
        body = (ROOM_WELCOME if SITE["room_welcome"] else "") + LOBBY
    await route.fulfill(body=f"<html><body>{body}</body></html>", content_type="text/html")


REDUCED_MOTION = "matchMedia('(prefers-reduced-motion: reduce)').matches"


@pytest.fixture
async def offline_client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[BrowserClient]:
    SITE.clear()
    SITE.update(home="none", room_welcome=False)
    async with BrowserClient(headless=True) as client:
        assert client._browser is not None
        original = client._browser.new_context

        async def offline(**options: Any) -> BrowserContext:
            context = await original(**options)
            await context.route("**/*", site)
            return context

        monkeypatch.setattr(client._browser, "new_context", offline)
        yield client


async def seat_players(page: Page) -> None:
    await page.evaluate(
        """() => setTimeout(() => document.querySelectorAll('[data-match-slot]').forEach(
            (panel, i) => {
              const member = document.createElement('button');
              member.setAttribute('aria-haspopup', 'dialog');
              member.textContent = 'Player' + (i + 1);
              panel.append(member);
            }), 300)"""
    )


async def test_host_creates_room_seats_four_players_then_starts(
    offline_client: BrowserClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    joined: list[tuple[str, PlayerAssignment]] = []

    async def join(page: Page, assignment: PlayerAssignment) -> None:
        joined.append((page.url, assignment))

    monkeypatch.setattr(arena_module, "join_room", join)
    host = Host(offline_client, "ArenaHost")
    room = await host.create_room(timeout=10)
    assert room == ROOM  # the generated URL, not a hard-coded one
    entry = json.loads(await host.page.evaluate("sessionStorage.getItem('entry')"))
    assert entry["name"] == "ArenaHost"
    assert entry["submittedAt"] - entry["typedAt"] >= NICKNAME_CONFIRM_DELAY * 1000 - 50

    arena = Arena(offline_client)
    await arena.open(room)
    assert [url for url, _ in joined] == [ROOM] * 4
    assert {(a.team, a.role) for _, a in joined} == {(t, r) for t in Team for r in Role}
    player_contexts = {id(p.context) for p in arena.players.values()}
    assert id(host.context) not in player_contexts  # the host is never a player
    assert len(offline_client._browser.contexts) == 5  # type: ignore[union-attr]
    # Host and every player shorten the site's animations; the state waits stay.
    for page in [host.page, *(p.context.pages[0] for p in arena.players.values())]:
        assert await page.evaluate(REDUCED_MOTION)

    await seat_players(host.page)
    await host.wait_for_players(timeout=5)
    await host.start_game(timeout=5)
    assert await host.page.evaluate("window.startedWith") == [1, 1, 1, 1]
    await arena.close()
    await host.close()


async def test_host_in_a_role_panel_is_rejected(offline_client: BrowserClient) -> None:
    host = Host(offline_client, "ArenaHost")
    await host.create_room(timeout=10)
    await seat_players(host.page)
    await host.page.locator(ROLE_PANELS[Team.RED, Role.OPERATIVE]).evaluate(
        """panel => { const b = document.createElement('button');
            b.setAttribute('aria-haspopup', 'dialog'); b.textContent = 'ArenaHost';
            panel.append(b); }"""
    )
    with pytest.raises(BrowserIntegrationError, match="appears in a role panel"):
        await host.wait_for_players(timeout=5)
    await host.close()


async def test_missing_start_button_fails_clearly(offline_client: BrowserClient) -> None:
    host = Host(offline_client)
    await host.create_room(timeout=10)
    await host.page.locator("#start").evaluate("button => button.remove()")
    with pytest.raises(BrowserIntegrationError, match="could not click Start game"):
        await host.start_game(timeout=0.5)
    await host.close()


@pytest.mark.parametrize("mode", ["replace", "clear"])
async def test_nickname_reset_before_submit_is_retyped_once(
    offline_client: BrowserClient, mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    SITE["home"] = mode
    host = Host(offline_client, "ArenaHost")
    assert await host.create_room(timeout=10) == ROOM
    entry = json.loads(await host.page.evaluate("sessionStorage.getItem('entry')"))
    assert (entry["name"], entry["resets"]) == ("ArenaHost", 1)  # never an empty submit
    assert entry["submittedAt"] - entry["typedAt"] >= NICKNAME_CONFIRM_DELAY * 1000 - 50
    assert "retyping" in caplog.text
    await host.close()


async def test_nickname_that_keeps_resetting_fails_with_diagnostics(
    offline_client: BrowserClient,
) -> None:
    SITE["home"] = "always"
    host = Host(offline_client, "ArenaHost")
    with pytest.raises(RoomCreationError, match="stage nickname_entry") as info:
        await host.create_room(timeout=5)
    assert await host.page.evaluate("sessionStorage.getItem('entry')") is None  # no submit
    error = info.value
    assert error.stage == "nickname_entry"
    assert error.details["url"] == "https://codenames.game/"
    assert error.details["nickname_fields"] == 1 and error.details["nickname_value"] == ""
    assert error.details["enter_game_buttons"] == 1 and error.details["role_panels"] == 0
    assert "reset before submit" in error.details["error"]
    assert error.screenshot and error.screenshot.startswith(b"\x89PNG")
    await host.close()


async def test_submit_without_navigation_fails_at_that_stage(
    offline_client: BrowserClient,
) -> None:
    SITE["home"] = "stuck"
    host = Host(offline_client, "ArenaHost")
    with pytest.raises(RoomCreationError) as info:
        await host.create_room(timeout=2)
    assert info.value.stage == "navigate_to_room"
    assert info.value.details["nickname_value"] == "ArenaHost"  # typed, but never entered
    await host.close()


async def test_room_welcome_dialog_is_answered_too(offline_client: BrowserClient) -> None:
    SITE["room_welcome"] = True
    host = Host(offline_client, "ArenaHost")
    assert await host.create_room(timeout=10) == ROOM
    assert await host.page.evaluate("window.roomWelcomeName") == "ArenaHost"
    assert not await host.page.get_by_role("dialog").count()
    await host.close()


def test_only_room_paths_count_as_entered() -> None:
    assert is_room_url("https://codenames.game/r/jutih-hikaj")
    assert not is_room_url("https://codenames.game/")
    assert not is_room_url("https://codenames.game/about")
    assert not is_room_url("https://example.com/r/jutih-hikaj")


class Flow:
    """Fakes for everything play_match drives, recording the order of the steps."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.steps: list[str] = []
        self.prompts: list[str] = []
        self.controller: Any = None
        flow = self

        class FakeHost:
            def __init__(self, browser: object, nickname: str) -> None:
                flow.steps.append(f"host {nickname}")

            async def create_room(self) -> str:
                flow.steps.append("create_room")
                if flow.create_error is not None:
                    raise flow.create_error
                return ROOM

            async def wait_for_players(self) -> None:
                flow.steps.append("host sees players")

            async def start_game(self) -> None:
                flow.steps.append("start_game")

            async def close(self) -> None:
                flow.steps.append("host close")

        class FakeReader:
            async def read_spymaster_state(self, team: Team) -> SpymasterGameState:
                flow.steps.append("board read")
                return spymaster_state(team)

        class FakeArena:
            def __init__(self, browser: object) -> None:
                self.players: dict = {
                    GamePhase.BLUE_SPYMASTER: SimpleNamespace(reader=FakeReader())
                }

            async def open(self, room: str) -> None:
                flow.steps.append(f"open {room}")

            async def verify(self) -> None:
                flow.steps.append("verify")

            async def wait_for_start(self, *, timeout: float) -> None:
                flow.steps.append("game live")

            async def close(self) -> None:
                flow.steps.append("arena close")

        class FakeController:
            def __init__(self, players: dict, spies: dict, ops: dict, **kwargs: object) -> None:
                flow.controller = self
                self.spies, self.ops, self.kwargs = spies, ops, kwargs
                self.winner = Team.BLUE

            async def run(self) -> None:
                flow.steps.append("controller")
                if flow.crash:
                    raise BrowserIntegrationError("lobby changed")

        def prompt(text: str = "") -> str:
            self.prompts.append(text)
            return ""

        self.crash = False
        self.create_error: RoomCreationError | None = None
        monkeypatch.setattr(main_module, "Host", FakeHost)
        monkeypatch.setattr(main_module, "Arena", FakeArena)
        monkeypatch.setattr(main_module, "GameController", FakeController)
        monkeypatch.setattr(main_module, "LOGS_DIR", tmp_path / "logs")
        monkeypatch.setattr(builtins, "input", prompt)

    async def play(
        self,
        room: str | None = None,
        settings: Settings | None = None,
        warmups: tuple[WarmUp, ...] = (),
    ) -> None:
        args = argparse.Namespace(room=room, host_nickname="ArenaHost", wait_seconds=5)
        config = LLMConfig(provider="ollama", model="qwen3:14b")
        roles = [(t, r) for t in Team for r in Role]
        await main_module.play_match(
            args,
            settings or Settings(),
            object(),  # type: ignore[arg-type]
            dict.fromkeys(roles, config),
            dict.fromkeys(roles, object()),  # type: ignore[arg-type]
            warmups=warmups,
        )


async def test_automatic_flow_needs_no_terminal_enter_to_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    flow = Flow(monkeypatch, tmp_path)
    await flow.play()
    assert flow.steps == [
        "host ArenaHost",
        "create_room",
        f"open {ROOM}",
        "verify",
        "host sees players",
        "start_game",  # only after all four joined and verified
        "game live",
        "board read",  # the key is captured once, before any clue
        "controller",
        "arena close",
        "host close",
    ]
    assert flow.prompts == [""]  # only the final "close the browsers" pause
    assert flow.controller is not None
    assert all(isinstance(a, RecordedSpymaster) for a in flow.controller.spies.values())
    assert all(isinstance(a, RecordedOperative) for a in flow.controller.ops.values())
    (match,) = (tmp_path / "logs").iterdir()
    meta = json.loads((match / "match.json").read_text("utf-8"))
    assert (meta["room_code"], meta["termination_reason"], meta["winner"]) == (
        "radun-kapuf",
        "game_over",
        "blue",
    )


async def test_existing_room_is_joined_without_creating_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    flow = Flow(monkeypatch, tmp_path)
    await flow.play(room="https://codenames.game/r/existing")
    assert flow.steps == [
        "open https://codenames.game/r/existing",
        "verify",
        "game live",
        "board read",  # the key is captured once, before any clue
        "controller",
        "arena close",
    ]
    assert flow.prompts[0] == "Press ENTER when the game has started: "


async def test_crash_finalizes_match_with_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    flow = Flow(monkeypatch, tmp_path)
    flow.crash = True
    with pytest.raises(BrowserIntegrationError):
        await flow.play(settings=Settings(anthropic_api_key="sk-ant-SECRET-0123456789"))
    assert flow.steps[-2:] == ["arena close", "host close"]
    (match,) = (tmp_path / "logs").iterdir()
    meta = json.loads((match / "match.json").read_text("utf-8"))
    assert meta["termination_reason"] == "error"
    assert meta["error"] == "BrowserIntegrationError: lobby changed"
    types = [json.loads(line)["type"] for line in (match / "events.jsonl").open()]
    assert types == [
        "room_ready",
        "players_ready",
        "match_started",
        "board_snapshot",
        "match_ended",
    ]
    for path in match.iterdir():
        assert "sk-ant-SECRET" not in path.read_text("utf-8")


async def test_warmups_are_logged_before_the_room_and_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    flow = Flow(monkeypatch, tmp_path)
    await flow.play(warmups=(WarmUp("qwen3:14b", 8.4321, False),))
    (match,) = (tmp_path / "logs").iterdir()
    lines = [json.loads(line) for line in (match / "events.jsonl").open()]
    assert [e["type"] for e in lines][:2] == ["llm_warmup", "room_ready"]
    assert lines[0]["models"] == [
        {"provider": "ollama", "model": "qwen3:14b", "seconds": 8.43, "already_loaded": False}
    ]


def test_cli_arena_creates_a_room_when_none_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    seen: list[argparse.Namespace] = []

    async def fake_run(args: argparse.Namespace, settings: Settings) -> None:
        seen.append(args)

    monkeypatch.setattr(main_module.Settings, "from_env", classmethod(lambda cls: Settings()))
    monkeypatch.setattr(main_module, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["m", "arena", "--host-nickname", "Referee"])
    assert main_module.main() == 0
    assert seen[0].room is None and seen[0].host_nickname == "Referee"
    monkeypatch.setattr(sys, "argv", ["m", "inspect"])  # other commands still need a room
    with pytest.raises(SystemExit):
        main_module.main()


async def test_room_creation_failure_leaves_startup_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    flow = Flow(monkeypatch, tmp_path)
    flow.create_error = RoomCreationError(
        "could not create a room",
        stage="navigate_to_room",
        details={"url": "https://codenames.game/", "nickname_value": "ArenaHost"},
        screenshot=b"\x89PNG fake",
    )
    with pytest.raises(RoomCreationError):
        await flow.play(settings=Settings(openai_api_key="sk-openai-SECRET-0123456789"))
    assert flow.steps == ["host ArenaHost", "create_room", "arena close", "host close"]
    (match,) = (tmp_path / "logs").iterdir()
    assert (match / "startup_failure.png").read_bytes() == b"\x89PNG fake"
    lines = [json.loads(line) for line in (match / "events.jsonl").open()]
    assert [e["type"] for e in lines] == ["startup_failure", "match_ended"]
    failure = lines[0]
    assert (failure["stage"], failure["url"], failure["screenshot"]) == (
        "navigate_to_room",
        "https://codenames.game/",
        "startup_failure.png",
    )
    meta = json.loads((match / "match.json").read_text("utf-8"))
    assert meta["termination_reason"] == "error" and meta["started_at"] is None
    for path in match.glob("*.json*"):
        assert "sk-openai-SECRET" not in path.read_text("utf-8")
