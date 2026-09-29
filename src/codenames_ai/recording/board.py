"""The secret board key, captured once per match for replay and evaluation only.

The snapshot is read from a spymaster view (the only view that shows every colour),
written to the match directory, and never returned to the controller or any agent:
capture_board reports only whether it succeeded.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from codenames_ai.domain.enums import CardColor, Team
from codenames_ai.domain.models import SpymasterGameState

logger = logging.getLogger(__name__)

BOARD_FILE = "board.json"
BOARD_SIZE = 25
SIDES = (CardColor.BLUE, CardColor.RED, CardColor.NEUTRAL, CardColor.ASSASSIN)
CAPTURE_ATTEMPTS = 15
CAPTURE_DELAY = 0.2  # seconds; the key can render a moment after the board


class BoardSnapshotError(ValueError):
    """The board is not a complete, valid key."""


class BoardAlreadyStartedError(BoardSnapshotError):
    """Cards were revealed before the snapshot: it cannot be the initial board."""


@dataclass(frozen=True)
class BoardCard:
    index: int
    word: str
    side: CardColor


@dataclass(frozen=True)
class BoardSnapshot:
    cards: tuple[BoardCard, ...]  # index order
    source: str  # the view it was read from, e.g. blue_spymaster

    def __post_init__(self) -> None:
        validate_cards(self.cards)

    @property
    def starting_team(self) -> Team | None:
        """The team with one card more moves first (9 against 8)."""
        blue, red = (sum(c.side is CardColor(t.value) for c in self.cards) for t in Team)
        if blue == red + 1:
            return Team.BLUE
        if red == blue + 1:
            return Team.RED
        return None

    def to_json(self, schema_version: int) -> dict[str, Any]:
        return {
            "schema_version": schema_version,
            "source": self.source,
            "starting_team": self.starting_team,
            "cards": [{"index": c.index, "word": c.word, "side": c.side} for c in self.cards],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "BoardSnapshot":
        raw = data.get("cards")
        if not isinstance(raw, list):
            raise BoardSnapshotError("board has no card list")
        cards = []
        for position, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise BoardSnapshotError(f"card {position} is not an object")
            index, word, side = item.get("index"), item.get("word"), item.get("side")
            if not isinstance(index, int) or isinstance(index, bool):
                raise BoardSnapshotError(f"card {position} has no integer index")
            if not isinstance(word, str) or not word.strip():
                raise BoardSnapshotError(f"card {index} has no word")
            if side not in {s.value for s in SIDES}:
                raise BoardSnapshotError(f"card {index} ({word}) has invalid side {side!r}")
            cards.append(BoardCard(index, word, CardColor(side)))
        snapshot = cls(tuple(sorted(cards, key=lambda c: c.index)), str(data.get("source", "")))
        recorded = data.get("starting_team")
        if recorded is not None and recorded != snapshot.starting_team:
            raise BoardSnapshotError(
                f"starting_team {recorded!r} does not match the key ({snapshot.starting_team})"
            )
        return snapshot

    @classmethod
    def from_state(cls, state: SpymasterGameState, source: str) -> "BoardSnapshot":
        """The untouched board as a spymaster sees it; incomplete reads are refused."""
        if any(card.revealed for card in state.cards):
            raise BoardAlreadyStartedError("board already has revealed cards")
        cards = []
        for card in sorted(state.cards, key=lambda c: c.index):
            if card.color not in SIDES:
                raise BoardSnapshotError(f"card {card.index} ({card.word}) has no readable side")
            cards.append(BoardCard(card.index, card.word, card.color))  # type: ignore[arg-type]
        return cls(tuple(cards), source)


def validate_cards(cards: tuple[BoardCard, ...]) -> None:
    if len(cards) != BOARD_SIZE:
        raise BoardSnapshotError(f"board has {len(cards)} cards, expected {BOARD_SIZE}")
    if sorted(c.index for c in cards) != list(range(BOARD_SIZE)):
        raise BoardSnapshotError(f"card indexes must be 0-{BOARD_SIZE - 1}, each exactly once")
    words = [c.word.casefold() for c in cards]
    if len(set(words)) != len(words):
        duplicates = sorted({w for w in words if words.count(w) > 1})
        raise BoardSnapshotError(f"duplicate board words: {', '.join(duplicates)}")
    assassins = sum(c.side is CardColor.ASSASSIN for c in cards)
    if assassins != 1:
        raise BoardSnapshotError(f"board has {assassins} assassins, expected exactly 1")


class BoardRecorder(Protocol):
    def record_board(self, snapshot: BoardSnapshot) -> bool: ...

    def event(self, type: str, **fields: Any) -> None: ...


async def capture_board(
    read_state: Callable[[], Awaitable[SpymasterGameState]],
    recorder: BoardRecorder,
    *,
    source: str,
    attempts: int = CAPTURE_ATTEMPTS,
    delay: float = CAPTURE_DELAY,
) -> bool:
    """Read the key from a spymaster view and record it once; never raises.

    Only success is reported back, so the key cannot flow into the caller's state.
    """
    problem = "no read attempted"
    for attempt in range(attempts):
        try:
            snapshot = BoardSnapshot.from_state(await read_state(), source)
        except BoardAlreadyStartedError as exc:
            problem = str(exc)
            break  # waiting cannot turn a started board back into an initial one
        except BoardSnapshotError as exc:
            problem = str(exc)
        except Exception as exc:  # noqa: BLE001 (a failed capture must never stop the match)
            problem = f"{type(exc).__name__}: {exc}"
        else:
            return recorder.record_board(snapshot)
        if attempt + 1 < attempts:
            await asyncio.sleep(delay)
    logger.error("Board snapshot not recorded (%s); this match will not be replayable", problem)
    recorder.event("board_snapshot_failed", source=source, reason=problem[:300])
    return False
