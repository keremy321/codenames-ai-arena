"""Deterministic operative policy over a tiered ranking.

The spymaster runs this same policy to simulate its teammate, so a change here
changes clue selection too; that coupling is deliberate.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .association import Fit, RankedWord
from .race import DEFAULT_RACE, RaceModel, RacePosition, guess_or_stop

# Without public race counts: guess only when the top strong word is probably one of
# the words still needed. With k words still needed and s equally strong candidates,
# P(top is ours) ~ k/s. A wrong guess costs about 1.2 hits on average (turn ends;
# enemy card; small assassin chance), so guessing pays only when P > ~0.55.
MIN_TARGET_SHARE = 0.6


@dataclass(frozen=True)
class PolicyDecision:
    word: str | None
    reason: str
    clue: str | None = None

    @property
    def end_turn(self) -> bool:
        return self.word is None


def decide_current(
    ranking: Sequence[RankedWord],
    remaining: int,
    *,
    first_guess_required: bool,
    position: RacePosition | None = None,
    model: RaceModel = DEFAULT_RACE,
) -> PolicyDecision:
    """Pick the next word for the current clue, or stop.

    With the public card counts (``position``) the choice is a race decision: guess when
    that raises our match win probability, e.g. press on when stopping would hand the
    opponent a likely winning turn, hold back when comfortably ahead.
    """
    if not ranking:
        return PolicyDecision(None, "no ranked words remain")
    if first_guess_required:
        top = ranking[0]
        return PolicyDecision(top.word, f"first guess of the turn ({top.fit.value})")
    if remaining <= 0:
        return PolicyDecision(None, "current clue already satisfied")
    strong = [item for item in ranking if item.fit is Fit.STRONG]
    if not strong:
        return PolicyDecision(None, "no strong match left for the clue")
    share = min(1.0, remaining / len(strong))
    odds = f"{len(strong)} strong for {remaining} remaining"
    if position is not None:
        guess, stop = guess_or_stop(share, position, model)
        if guess <= stop:
            return PolicyDecision(
                None, f"stopping keeps a better race ({odds}; win {stop:.2f} vs {guess:.2f})"
            )
        return PolicyDecision(strong[0].word, f"race favours guessing ({odds}; win {guess:.2f})")
    if share < MIN_TARGET_SHARE:
        return PolicyDecision(None, f"ambiguous: {odds} word(s)")
    return PolicyDecision(strong[0].word, f"strong match; {odds}")


def decide_bonus(
    rankings: Mapping[str, Sequence[RankedWord]], unresolved: Mapping[str, int]
) -> PolicyDecision:
    """Use the optional extra guess only for an isolated strong old-clue match.

    Each old clue is judged on its OWN list: a clue qualifies only when it has at least
    one strong word and no more strong words than it has unresolved words, so every
    strong candidate is probably ours. Stricter than decide_current because the guess
    is optional and public unresolved counts can still be imperfect.
    """
    options: list[tuple[int, str, RankedWord]] = []
    crowded: list[str] = []
    for clue, count in unresolved.items():
        strong = [item for item in rankings.get(clue, ()) if item.fit is Fit.STRONG]
        if not strong or count <= 0:
            continue
        if len(strong) > count:
            crowded.append(f"{clue}: {len(strong)} strong for {count} unresolved")
            continue
        options.append((len(strong), clue, strong[0]))
    if options:
        # Fewest competitors first; ties keep the more recent clue (listed later).
        count, clue, item = min(reversed(options), key=lambda option: option[0])
        return PolicyDecision(
            item.word,
            f"isolated strong match for old clue {clue} "
            f"({count} strong for {unresolved[clue]} unresolved)",
            clue,
        )
    if crowded:
        return PolicyDecision(None, "old-clue matches are ambiguous (" + "; ".join(crowded) + ")")
    return PolicyDecision(None, "no strong old-clue match")
