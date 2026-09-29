"""Replayed match state: what happened, rebuilt from the recording alone."""

from dataclasses import dataclass, field
from typing import Any

from codenames_ai.domain.enums import CardColor, GamePhase, Team
from codenames_ai.recording.board import BoardCard, BoardSnapshot


class ReplayError(ValueError):
    """The recording is inconsistent or cannot be read."""


class ReplayUnavailableError(ReplayError):
    """The match was recorded without what replay needs (e.g. an older format)."""


@dataclass(frozen=True)
class ReplayGuess:
    word: str
    index: int
    side: CardColor  # the card's true side, from the board
    result: str  # relative to the guessing team: friendly, opponent, neutral, assassin
    bonus: bool = False


@dataclass
class ReplayTurn:
    number: int  # 1-based, one per clue
    team: Team
    clue: str
    clue_number: int
    guesses: list[ReplayGuess] = field(default_factory=list)


@dataclass
class ReplayState:
    board: BoardSnapshot
    phase: GamePhase
    turns: list[ReplayTurn] = field(default_factory=list)
    revealed: dict[int, Team] = field(default_factory=dict)  # card index -> guessing team
    winner: Team | None = None  # reconstructed from the reveals
    decided_by: str | None = None  # "assassin" or "all cards"
    recorded_winner: Team | None = None
    termination_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def card(self, index: int) -> BoardCard:
        return self.board.cards[index]

    def remaining(self, team: Team) -> int:
        side = CardColor(team.value)
        return sum(1 for c in self.board.cards if c.side is side and c.index not in self.revealed)

    @property
    def current_team(self) -> Team | None:
        if self.phase in (GamePhase.GAME_OVER, GamePhase.UNKNOWN):
            return None
        return Team(self.phase.value.split("_")[0])


@dataclass(frozen=True)
class ReplaySummary:
    turns: int
    guesses: int
    blue_guesses: int
    red_guesses: int
    neutral_hits: int
    enemy_hits: int  # guesses that revealed the other team's card
    assassin_hits: int
    winner: Team | None

    @classmethod
    def of(cls, state: ReplayState) -> "ReplaySummary":
        guesses = [(turn.team, g) for turn in state.turns for g in turn.guesses]
        return cls(
            turns=len(state.turns),
            guesses=len(guesses),
            blue_guesses=sum(team is Team.BLUE for team, _ in guesses),
            red_guesses=sum(team is Team.RED for team, _ in guesses),
            neutral_hits=sum(g.side is CardColor.NEUTRAL for _, g in guesses),
            enemy_hits=sum(g.result == "opponent" for _, g in guesses),
            assassin_hits=sum(g.side is CardColor.ASSASSIN for _, g in guesses),
            winner=state.winner,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "guesses": self.guesses,
            "blue_guesses": self.blue_guesses,
            "red_guesses": self.red_guesses,
            "neutral_hits": self.neutral_hits,
            "enemy_hits": self.enemy_hits,
            "assassin_hits": self.assassin_hits,
            "winner": self.winner.value if self.winner else None,
        }


@dataclass(frozen=True)
class Replay:
    directory: str
    metadata: dict[str, Any]
    state: ReplayState

    @property
    def summary(self) -> ReplaySummary:
        return ReplaySummary.of(self.state)
