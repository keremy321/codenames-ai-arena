import argparse
import asyncio
import logging
from pathlib import Path

from playwright.async_api import Error as PlaywrightError
from pydantic import ValidationError

from codenames_ai.agents.operative import OperativeAgent
from codenames_ai.agents.spymaster import SpymasterAgent
from codenames_ai.browser.arena import Arena
from codenames_ai.browser.client import BrowserClient
from codenames_ai.browser.errors import BrowserIntegrationError
from codenames_ai.browser.host import HOST_NICKNAME, Host, RoomCreationError
from codenames_ai.browser.identity import detect_assignment
from codenames_ai.browser.probe import inspect_loop
from codenames_ai.browser.reader import BoardReading, DOMParseError, GameReader
from codenames_ai.config import Settings, role_key
from codenames_ai.controller import GameController
from codenames_ai.domain.enums import GamePhase, Role, Team
from codenames_ai.llm.base import PROVIDERS, LLMClient, LLMConfig, LLMError
from codenames_ai.llm.factory import open_llm_clients
from codenames_ai.llm.ollama import OllamaClient, OllamaError
from codenames_ai.llm.preflight import llm_preflight
from codenames_ai.recording import LOGS_DIR, MatchRecorder, RecordedOperative, RecordedSpymaster

ARENA_ROLES = [
    (Team.BLUE, Role.SPYMASTER),
    (Team.BLUE, Role.OPERATIVE),
    (Team.RED, Role.SPYMASTER),
    (Team.RED, Role.OPERATIVE),
]


def format_reading(reading: BoardReading) -> str:
    state = reading.state
    lines = [
        "Connected to Codenames",
        "",
        "Instruction:",
        reading.instruction or "Unknown",
        "",
        f"Selected reader identity: {state.team.value.upper()} {state.role.value.upper()}",
    ]
    turn = reading.turn
    lines.append(f"Detected active turn: {turn.team or 'unknown'} {turn.role or 'unknown'}")
    lines.extend(
        [
            "",
            "Current clue:",
            f"{state.clue.word} {state.clue.number}" if state.clue else "None / unreadable",
            "",
            "Cards:",
        ]
    )
    for card in state.cards:
        # Explicit guard also protects printing of accidentally unsafe constructed states.
        color = (
            f"  {card.color.value.upper()}"
            if card.color is not None and (state.role == Role.SPYMASTER or card.revealed)
            else ""
        )
        lines.append(f"{card.index:02d} {card.word}{color}")
    return "\n".join(lines)


def format_llms(llms: dict[tuple[Team, Role], LLMConfig]) -> str:
    """Provider and model per role; names only, never credentials."""
    return "\n".join(
        f"{f'{team.value} {role.value}:'.upper():16} {llms[team, role]}"
        for team, role in ARENA_ROLES
    )


async def run_arena(args: argparse.Namespace, settings: Settings) -> None:
    """Every model is checked (Ollama started if needed) before any browser opens."""
    llms = {(team, role): settings.llm_for(team, role) for team, role in ARENA_ROLES}
    print("Checking LLM providers...", flush=True)
    async with (
        llm_preflight(llms.values(), settings),
        open_llm_clients(llms, settings) as clients,
    ):
        print(f"\n{format_llms(llms)}\n", flush=True)
        async with BrowserClient(headless=settings.headless) as browser:
            await play_match(args, settings, browser, llms, clients)


async def play_match(
    args: argparse.Namespace,
    settings: Settings,
    browser: BrowserClient,
    llms: dict[tuple[Team, Role], LLMConfig],
    clients: dict[tuple[Team, Role], LLMClient],
) -> None:
    """Create (or join) the room, seat four players, start, play, and record the match."""
    arena = Arena(browser)
    host: Host | None = None
    recorder: MatchRecorder | None = None
    try:
        players = {role_key(*key): config for key, config in llms.items()}
        room_url = args.room or settings.room_url
        if not room_url:
            print("Creating Codenames room...", flush=True)
            host = Host(browser, args.host_nickname)
            try:
                room_url = await host.create_room()
            except RoomCreationError as exc:
                recorder = MatchRecorder(
                    LOGS_DIR, room_url="", players=players, secret_values=secret_values(settings)
                )
                screenshot = "startup_failure.png"
                if exc.screenshot is None or not recorder.attach(screenshot, exc.screenshot):
                    screenshot = None
                recorder.event(
                    "startup_failure", stage=exc.stage, screenshot=screenshot, **exc.details
                )
                print(f"Startup diagnostics: {recorder.directory.as_posix()}/", flush=True)
                raise
            print(f"Room created: {room_url}\n", flush=True)
        recorder = MatchRecorder(
            LOGS_DIR, room_url=room_url, players=players, secret_values=secret_values(settings)
        )
        recorder.event("room_ready", room_url=room_url, created_by_arena=host is not None)
        await arena.open(room_url)
        await arena.verify()
        recorder.event("players_ready")
        print("\nAll four AI players are ready.", flush=True)
        if host is not None:
            await host.wait_for_players()
            print("Starting game...", flush=True)
            await host.start_game()
        else:
            print("Start the game from the host browser.", flush=True)
            await asyncio.to_thread(input, "Press ENTER when the game has started: ")
        await arena.wait_for_start(timeout=args.wait_seconds)
        print("Game started.\n", flush=True)
        recorder.start()
        print(f"Recording: {recorder.directory.as_posix()}/\n", flush=True)

        def agent(team: Team, role: Role) -> RecordedSpymaster | RecordedOperative:
            assert recorder is not None
            llm, config = clients[team, role], llms[team, role]
            if role == Role.SPYMASTER:
                return RecordedSpymaster(SpymasterAgent(team, llm), recorder, config)
            return RecordedOperative(OperativeAgent(team, llm), recorder, config)

        controller = GameController(
            arena.players,
            {GamePhase(role_key(t, r)): agent(t, r) for t, r in ARENA_ROLES if r == Role.SPYMASTER},
            {GamePhase(role_key(t, r)): agent(t, r) for t, r in ARENA_ROLES if r == Role.OPERATIVE},
            timeout=args.wait_seconds,
            recorder=recorder,
        )
        await controller.run()
        recorder.finish(reason="game_over", winner=controller.winner)
        winner = f": {controller.winner.upper()} wins" if controller.winner else ""
        print(f"Game over{winner}. Press Enter to close the browser sessions.")
        await asyncio.to_thread(input)
    except BaseException as exc:
        if recorder is not None:
            interrupted = isinstance(exc, KeyboardInterrupt | asyncio.CancelledError)
            recorder.finish(
                reason="interrupted" if interrupted else "error",
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
        raise
    finally:
        await arena.close()
        if host is not None:
            await host.close()


def secret_values(settings: Settings) -> list[str]:
    keys = (settings.openai_api_key, settings.anthropic_api_key)
    return [key.get_secret_value() for key in keys if key is not None]


async def run(args: argparse.Namespace, settings: Settings) -> None:
    if args.check_ollama:
        async with OllamaClient(
            settings.ollama_base_url, settings.ollama_model, timeout=settings.ollama_timeout
        ) as llm:
            result = await llm.chat_json(
                system="You are a test agent. Return JSON only.", user='Return {"status": "ok"}'
            )
            if result != {"status": "ok"}:
                raise OllamaError("Model returned JSON but not the requested status=ok")
            print(f"Ollama connected ({settings.ollama_model}): {result}")
        return
    if args.command == "arena":
        await run_arena(args, settings)
        return
    async with BrowserClient(headless=settings.headless) as browser:
        context = await browser.open_room(args.room or settings.room_url)
        page = context.pages[0]
        if args.command == "inspect":
            await inspect_loop(page, Path(args.snapshot_dir))
            return
        print("Join the room and select your role in the browser, then start the game.", flush=True)
        reader = GameReader(page)
        await reader.wait_for_board(timeout_ms=args.wait_seconds * 1000)
        if not settings.headless:
            await asyncio.to_thread(input, "When your board and role are ready, press Enter here: ")
        identity = await detect_assignment(page)
        team, role = identity.team, identity.role
        if (args.team and Team(args.team) != team) or (args.role and Role(args.role) != role):
            raise BrowserIntegrationError(
                "CLI team/role does not match the local player's DOM assignment"
            )
        print(format_reading(await reader.read(team, role)))


def apply_llm_args(args: argparse.Namespace, settings: Settings) -> None:
    """CLI flags override the environment: the default first, then per-role values."""
    settings.llm_provider = args.provider or settings.llm_provider
    settings.llm_model = args.model or settings.llm_model
    for team, role in ARENA_ROLES:
        key = role_key(team, role)
        if provider := getattr(args, f"{key}_provider"):
            settings.role_providers[key] = provider
        if model := getattr(args, f"{key}_model"):
            settings.role_models[key] = model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and control Codenames with isolated player sessions"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["inspect", "arena"],
        help="Omit for the original board smoke test",
    )
    parser.add_argument(
        "--room",
        help="Codenames room URL (or CODENAMES_ROOM_URL); arena creates a room when omitted",
    )
    parser.add_argument(
        "--host-nickname",
        default=HOST_NICKNAME,
        help=f"Arena host (spectator/admin) nickname (default: {HOST_NICKNAME})",
    )
    parser.add_argument("--team", choices=list(Team))
    parser.add_argument("--role", choices=list(Role))
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wait-seconds", type=float, default=300)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--check-ollama", action="store_true")
    parser.add_argument(
        "--provider", choices=PROVIDERS, help="Default LLM provider for every role (ollama)"
    )
    parser.add_argument("--model", help="Default model for every role (OLLAMA_MODEL on ollama)")
    for team, role in ARENA_ROLES:
        flag = f"--{team.value}-{role.value}"
        parser.add_argument(
            f"{flag}-provider", choices=PROVIDERS, help=f"{team.value} {role.value} provider"
        )
        parser.add_argument(f"{flag}-model", help=f"{team.value} {role.value} model")
    parser.add_argument(
        "--snapshot-dir", default="debug", help="Inspector snippet output (default: ignored debug/)"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = Settings.from_env()
        if args.headless is not None:
            settings.headless = args.headless
        apply_llm_args(args, settings)
        if args.command == "inspect":
            settings.headless = False
        if not args.check_ollama:
            if args.command != "arena" and not (args.room or settings.room_url):
                parser.error("Supply --room or set CODENAMES_ROOM_URL")
            if args.wait_seconds <= 0:
                parser.error("--wait-seconds must be positive")
            if settings.headless and args.command is None and not (args.team and args.role):
                parser.error("Headless mode requires --team and --role")
            if settings.headless and args.command == "arena" and (args.room or settings.room_url):
                parser.error("Joining an existing room requires HEADLESS=false to start its game")
        asyncio.run(run(args, settings))
    except KeyboardInterrupt:
        return 130
    except (
        LLMError,
        PlaywrightError,
        DOMParseError,
        ValidationError,
        ValueError,
        EOFError,
        BrowserIntegrationError,
    ) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
