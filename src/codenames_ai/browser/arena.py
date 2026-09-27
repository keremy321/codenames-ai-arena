import asyncio
import logging
from dataclasses import dataclass

from playwright.async_api import BrowserContext

from codenames_ai.domain.enums import GamePhase, Role, Team
from codenames_ai.domain.models import PlayerAssignment, PublicGameState, SpymasterGameState

from .actions import GameActions, GuessResult
from .client import BrowserClient
from .errors import GameStateTimeoutError
from .identity import detect_assignment, join_room
from .reader import GameReader

logger = logging.getLogger(__name__)

ASSIGNMENTS = tuple(
    PlayerAssignment(
        team=team, role=role, nickname=f"{team.value.capitalize()}{role.value.capitalize()}AI"
    )
    for team in Team
    for role in (Role.SPYMASTER, Role.OPERATIVE)
)


@dataclass
class BrowserPlayer:
    assignment: PlayerAssignment
    context: BrowserContext
    reader: GameReader
    actions: GameActions
    trusted_join: bool = False

    async def verify(self) -> None:
        if not self.trusted_join:
            await detect_assignment(self.context.pages[0], self.assignment)

    async def phase(self) -> GamePhase:
        return await self.reader.phase()

    async def state(self) -> PublicGameState | SpymasterGameState:
        if self.assignment.role == Role.OPERATIVE:
            return await self.reader.read_operative_state(self.assignment.team)
        return await self.reader.read_spymaster_state(self.assignment.team)

    async def submit_clue(self, clue: str, number: int) -> None:
        await self.actions.submit_clue(clue, number)

    async def guess_card(self, word: str) -> GuessResult:
        return await self.actions.guess_card(word)

    async def end_guessing(self) -> None:
        await self.actions.end_guessing()

    async def can_end_guessing(self) -> bool:
        return await self.actions.can_end_guessing()


class Arena:
    """Own four contexts in a caller-owned single Chromium browser."""

    def __init__(self, browser: BrowserClient) -> None:
        self.browser = browser
        self.players: dict[GamePhase, BrowserPlayer] = {}

    async def open(self, room: str) -> None:
        if self.players:
            raise RuntimeError("Arena is already open")
        try:
            for assignment in ASSIGNMENTS:
                context = await self.browser.open_room(room)
                page = context.pages[0]
                player = BrowserPlayer(
                    assignment, context, GameReader(page), GameActions(page, assignment)
                )
                self.players[GamePhase(f"{assignment.team}_{assignment.role}")] = player
                await join_room(page, assignment)
                player.trusted_join = True
                player.actions.trusted_join = True
        except BaseException as exc:
            logger.error("Arena setup failed before browser cleanup: %s", exc)
            await self.close()
            raise

    async def verify(self) -> None:
        for player in self.players.values():
            await player.verify()

    async def wait_for_start(self, *, timeout: float = 30) -> None:
        observer = self.players[GamePhase.BLUE_OPERATIVE]
        try:
            async with asyncio.timeout(timeout):
                while await observer.phase() in (GamePhase.WAITING, GamePhase.UNKNOWN):
                    await asyncio.sleep(0.1)
        except TimeoutError as exc:
            raise GameStateTimeoutError("Game did not start in the host browser") from exc

    async def close(self) -> None:
        # Attempt every close even when one context has already disconnected.
        results = await asyncio.gather(
            *(p.context.close() for p in self.players.values()), return_exceptions=True
        )
        self.players.clear()
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("Context cleanup: %s", result)


async def wait_for_phase(player: BrowserPlayer, phase: GamePhase, timeout: float) -> None:
    """Small bounded polling bridges asynchronous updates in separate contexts."""
    try:
        async with asyncio.timeout(timeout):
            while await player.phase() != phase:
                await asyncio.sleep(0.1)
    except TimeoutError as exc:
        raise GameStateTimeoutError(f"{player.assignment.nickname} did not reach {phase}") from exc
