from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import CardColor, Role, Team


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class Card(FrozenModel):
    index: int = Field(ge=0)
    word: str = Field(min_length=1)
    color: CardColor | None = None
    revealed: bool = False
    selected: bool = False


class Clue(FrozenModel):
    word: str = Field(min_length=1)
    number: int = Field(ge=0, strict=True)


class PublicClueMemory(FrozenModel):
    clue: str
    number: int = Field(ge=0)
    guesses_made: int = Field(ge=0)
    friendly_hits: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    words_guessed: tuple[str, ...] = ()
    # The team's own public reading of the clue (the operative's strong words when it
    # first ranked it). A friendly card among them revealed later, by any team or under
    # any clue, also counts as found for this clue.
    associated: tuple[str, ...] = ()


class PublicGameState(FrozenModel):
    team: Team
    role: Literal[Role.OPERATIVE] = Role.OPERATIVE
    cards: tuple[Card, ...]
    clue: Clue | None = None
    clue_history: tuple[PublicClueMemory, ...] = ()
    bonus_only: bool = False
    # Public: the team that moved first has 9 cards, the other 8. With the revealed
    # cards this gives the score display's remaining counts, nothing more.
    starting_team: Team | None = None

    @model_validator(mode="after")
    def sanitize(self) -> Self:
        # Copy every card so callers cannot retain a mutable shared container.
        object.__setattr__(
            self,
            "cards",
            tuple(
                Card(
                    index=c.index,
                    word=c.word,
                    revealed=c.revealed,
                    selected=c.selected,
                    color=c.color if c.revealed else None,
                )
                for c in self.cards
            ),
        )
        return self


class SpymasterGameState(FrozenModel):
    team: Team
    role: Literal[Role.SPYMASTER] = Role.SPYMASTER
    cards: tuple[Card, ...]
    clue: Clue | None = None


class ClueDecision(Clue):
    pass


class GuessDecision(FrozenModel):
    indices: tuple[int, ...] = ()
    end_turn: bool = False
    source_clue: str | None = None
    clue_words: tuple[str, ...] = ()  # the operative's strong words for the current clue

    @model_validator(mode="after")
    def valid_indices(self) -> Self:
        if any(i < 0 for i in self.indices) or len(set(self.indices)) != len(self.indices):
            raise ValueError("Guess indices must be nonnegative and unique")
        return self


class PlayerAssignment(FrozenModel):
    team: Team
    role: Role
    nickname: str = Field(min_length=1, max_length=20)
