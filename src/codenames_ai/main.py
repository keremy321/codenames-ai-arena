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
from codenames_ai.browser.identity import detect_assignment
from codenames_ai.browser.probe import inspect_loop
from codenames_ai.browser.reader import BoardReading, DOMParseError, GameReader
from codenames_ai.config import Settings
from codenames_ai.controller import GameController
from codenames_ai.domain.enums import GamePhase, Role, Team
from codenames_ai.llm.ollama import OllamaClient, OllamaError


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
    async with BrowserClient(headless=settings.headless) as browser:
        if args.command == "arena":
            arena = Arena(browser)
            try:
                await arena.open(args.room or settings.room_url)
                await arena.verify()
                print("All four AI players are ready.", flush=True)
                print("Start the game from the host browser.", flush=True)
                await asyncio.to_thread(input, "Press ENTER when the game has started: ")
                await arena.wait_for_start(timeout=args.wait_seconds)
                async with OllamaClient(
                    settings.ollama_base_url, settings.ollama_model, timeout=settings.ollama_timeout
                ) as llm:
                    controller = GameController(
                        arena.players,
                        {
                            GamePhase.BLUE_SPYMASTER: SpymasterAgent(Team.BLUE, llm),
                            GamePhase.RED_SPYMASTER: SpymasterAgent(Team.RED, llm),
                        },
                        {
                            GamePhase.BLUE_OPERATIVE: OperativeAgent(Team.BLUE, llm),
                            GamePhase.RED_OPERATIVE: OperativeAgent(Team.RED, llm),
                        },
                        timeout=args.wait_seconds,
                    )
                    await controller.run()
                print("Game over. Press Enter to close the browser sessions.")
                await asyncio.to_thread(input)
            finally:
                await arena.close()
            return
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
    parser.add_argument("--room", help="Codenames room URL (or CODENAMES_ROOM_URL)")
    parser.add_argument("--team", choices=list(Team))
    parser.add_argument("--role", choices=list(Role))
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wait-seconds", type=float, default=300)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--check-ollama", action="store_true")
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
        if args.command == "inspect":
            settings.headless = False
        if not args.check_ollama:
            if not (args.room or settings.room_url):
                parser.error("Supply --room or set CODENAMES_ROOM_URL")
            if args.wait_seconds <= 0:
                parser.error("--wait-seconds must be positive")
            if settings.headless and args.command is None and not (args.team and args.role):
                parser.error("Headless mode requires --team and --role")
            if settings.headless and args.command == "arena":
                parser.error("Arena requires HEADLESS=false to watch and start the host game")
        asyncio.run(run(args, settings))
    except KeyboardInterrupt:
        return 130
    except (
        OllamaError,
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
