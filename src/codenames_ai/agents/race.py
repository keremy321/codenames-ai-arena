"""Codenames as a race: win probability from both teams' remaining cards.

Both teams alternate turns; on a typical future turn a team clears 0 to 4
cards with the probabilities in ``RaceModel.progress``. Whoever clears its last card
first wins. From that small Markov model we get, for any position, the probability
that we win the match. A clue is then worth the win probability of the position it
leads to, averaged over its outcomes (clue_scoring), so tempo, finishing now, handing
the opponent a card, and the opponent's threat are all priced in one currency.

Only the spymaster uses this; it needs the counts derived from the hidden key.
"""

from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class RaceModel:
    # P(a typical FUTURE turn clears 0, 1, 2, 3, 4 cards), used for both teams' later
    # turns only; the current clue is valued from its own predicted outcomes. Mean
    # 1.47 cards per turn: the harness averages ~1.5 hits for these agents, and a
    # slower prior would undervalue tempo and push the spymaster toward one-card clues.
    progress: tuple[float, ...] = (0.15, 0.42, 0.28, 0.11, 0.04)

    def __post_init__(self) -> None:
        if abs(sum(self.progress) - 1.0) > 1e-9 or self.progress[0] >= 1.0:
            raise ValueError("progress must be a probability distribution with progress[0] < 1")

    def win_probability(self, ours: int, theirs: int, *, our_turn: bool) -> float:
        """P(we win) with ``ours``/``theirs`` cards left and the given side to move."""
        if ours <= 0:
            return 1.0
        if theirs <= 0:
            return 0.0
        we_move, they_move = _race(self.progress, ours, theirs)
        return we_move if our_turn else they_move

    def expected_turns(self, remaining: int) -> float:
        """Expected own turns to clear ``remaining`` cards, ignoring the opponent."""
        return _turns(self.progress, remaining)

    def finish_chance(self, remaining: int) -> float:
        """P(a team clears all ``remaining`` cards on its next typical turn)."""
        if remaining <= 0:
            return 1.0
        return sum(p for k, p in enumerate(self.progress) if k >= remaining)


@cache
def _race(progress: tuple[float, ...], ours: int, theirs: int) -> tuple[float, float]:
    """(P(win) if we move, P(win) if they move); both sides use ``progress``."""

    def we_move_after(k: int) -> float:  # after the opponent clears k cards
        return 0.0 if k >= theirs else _race(progress, ours, theirs - k)[0]

    def they_move_after(k: int) -> float:  # after we clear k cards
        return 1.0 if k >= ours else _race(progress, ours - k, theirs)[1]

    stay = progress[0]
    ours_progress = sum(p * they_move_after(k) for k, p in enumerate(progress) if k)
    theirs_progress = sum(p * we_move_after(k) for k, p in enumerate(progress) if k)
    # W = stay*V + ours_progress and V = stay*W + theirs_progress (a 0-card turn passes).
    we_move = (stay * theirs_progress + ours_progress) / (1 - stay * stay)
    they_move = stay * we_move + theirs_progress
    return we_move, they_move


@cache
def _turns(progress: tuple[float, ...], remaining: int) -> float:
    if remaining <= 0:
        return 0.0
    rest = sum(p * _turns(progress, remaining - k) for k, p in enumerate(progress) if k)
    return (1 + rest) / (1 - progress[0])


DEFAULT_RACE = RaceModel()


@dataclass(frozen=True)
class RaceState:
    """The spymaster's view of the race before giving its clue (our turn)."""

    ours: int
    theirs: int
    win_probability: float  # if we play a typical turn now
    their_finish_chance: float  # P(opponent finishes on its next typical turn)
    our_turns: float
    their_turns: float

    @classmethod
    def of(cls, ours: int, theirs: int, model: RaceModel = DEFAULT_RACE) -> "RaceState":
        return cls(
            ours=ours,
            theirs=theirs,
            win_probability=model.win_probability(ours, theirs, our_turn=True),
            their_finish_chance=model.finish_chance(theirs),
            our_turns=model.expected_turns(ours),
            their_turns=model.expected_turns(theirs),
        )

    @property
    def pressure(self) -> str:
        """Human-readable label; selection itself uses win probabilities, not this."""
        if self.ours <= 1:
            return "last card"
        if self.their_finish_chance >= 0.5:
            return "critical"
        # We move first, so equal turn counts favour us.
        margin = self.their_turns - self.our_turns
        if margin >= 1.0:
            return "ahead"
        if margin <= -1.0:
            return "behind"
        return "even"

    def describe(self) -> str:
        return (
            f"ours={self.ours} theirs={self.theirs} pressure={self.pressure} "
            f"win~{self.win_probability:.2f} (their finish-next chance "
            f"{self.their_finish_chance:.2f}; turns to finish {self.our_turns:.1f} vs "
            f"{self.their_turns:.1f})"
        )


@dataclass(frozen=True)
class RacePosition:
    """Card counts a player can read off the public score display.

    The operative derives these from the revealed cards and the (public) starting team;
    the spymaster's key gives the same numbers.
    """

    ours: int
    theirs: int
    neutral: int
    assassin_hidden: bool

    def after_hits(self, hits: int) -> "RacePosition":
        return RacePosition(self.ours - hits, self.theirs, self.neutral, self.assassin_hidden)


def guess_or_stop(
    share: float, position: RacePosition, model: RaceModel = DEFAULT_RACE
) -> tuple[float, float]:
    """(win probability if we guess the top strong word, if we stop now).

    ``share`` is the chance the word is ours. A miss lands on a random non-friendly
    card: an enemy card (their progress, their win if it was their last), a neutral
    (turn over), or the assassin (we lose), in proportion to the public counts.
    After a hit we value stopping; further guesses are decided again then.
    """
    stop = model.win_probability(position.ours, position.theirs, our_turn=False)
    hit = model.win_probability(position.ours - 1, position.theirs, our_turn=False)
    assassin = 1 if position.assassin_hidden else 0
    others = position.theirs + position.neutral + assassin
    if others <= 0:
        return hit, stop
    enemy = model.win_probability(position.ours, position.theirs - 1, our_turn=False)
    miss = (position.theirs * enemy + position.neutral * stop) / others
    return share * hit + (1 - share) * miss, stop
