"""Score a clue by simulating our operative, then valuing the outcome in the race.

Tactical layer: for each candidate the spymaster has a colour-blind ranking in the
real operative's format. We enumerate, exactly, what our operative does with it:

* each guess picks among the listed words with probability proportional to its tier
  weight times a decay per list position. A screened (batched) ranking only roughly
  predicts the operative's order; a verified ranking is the operative's own request,
  reproduced byte for byte, so its order is trusted almost fully;
* after the first guess it continues only when guess_policy.decide_current says so
  (the same code the real operative runs);
* the turn ends at the first non-friendly card.

Strategic layer: each outcome (hits, card that ended the turn) leads to a race
position whose win probability race.RaceModel gives. A clue's value is the win
probability averaged over its outcomes. Finishing all our cards is worth 1, revealing
the opponent's last card is worth 0, and one extra card is worth whatever it changes
in the race; there is no fixed bonus per clue number.

The assassin stays a hard veto: any listed assassin rejects the clue.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from codenames_ai.domain.enums import CardColor, Team

from .association import Fit, RankedWord
from .guess_policy import decide_current
from .race import DEFAULT_RACE, RaceModel, RacePosition, RaceState


class Side(StrEnum):
    FRIENDLY = "friendly"
    ENEMY = "enemy"
    NEUTRAL = "neutral"
    ASSASSIN = "assassin"


def side_of(color: CardColor, team: Team) -> Side:
    if color == CardColor(team.value):
        return Side.FRIENDLY
    if color == CardColor.ASSASSIN:
        return Side.ASSASSIN
    if color == CardColor.NEUTRAL:
        return Side.NEUTRAL
    if color in (CardColor.BLUE, CardColor.RED):
        return Side.ENEMY
    raise ValueError(f"Cannot score unknown card colour {color!r}")


@dataclass(frozen=True)
class ClueUtility:
    # 4+ is never chosen: it needs four strong blind matches with no competitor, which
    # almost never happens, and a wrong fourth guess costs more than it gains.
    max_number: int = 3
    # Relative chance the operative picks a word of this tier over another candidate.
    pick_weight: Mapping[Fit, float] = field(
        default_factory=lambda: MappingProxyType(
            {Fit.STRONG: 1.0, Fit.POSSIBLE: 0.1, Fit.WEAK: 0.05}
        )
    )
    # Factor per position further down the list. Batched rankings only roughly match
    # the operative's order; verified rankings reproduced exactly in every probe, and
    # 0.1 keeps a small allowance for a different board read or clue spelling.
    position_decay: float = 0.6
    verified_position_decay: float = 0.1
    # Tactical EV only (logs and tie-breaks): an enemy card is a point for them, a
    # neutral wastes tempo, the assassin loses. Selection uses race win probability.
    miss_cost: Mapping[Side, float] = field(
        default_factory=lambda: MappingProxyType(
            {Side.NEUTRAL: 0.25, Side.ENEMY: 1.0, Side.ASSASSIN: 10.0}
        )
    )
    race: RaceModel = DEFAULT_RACE


DEFAULT_UTILITY = ClueUtility()


@dataclass(frozen=True)
class Outcome:
    probability: float
    hits: int
    ended_by: Side | None  # the wrong card that ended the turn, if any


@dataclass(frozen=True)
class ClueAssessment:
    clue: str
    number: int
    intended: tuple[str, ...]
    ranking: tuple[RankedWord, ...]
    verified: bool
    expected_value: float  # tactical: hits minus miss costs
    expected_hits: float
    p_first_friendly: float
    finish_chance: float  # P(this clue clears all our remaining cards)
    turns_to_finish: float  # this turn plus expected later turns
    win_probability: float
    hits: tuple[str, ...]  # friendly words the operative most likely finds
    acceptable: bool
    vetoed: bool
    reason: str

    @property
    def risk(self) -> float:
        """Expected miss cost of the clue (tactical)."""
        return self.expected_hits - self.expected_value

    def metrics(self) -> str:
        return (
            f"hits~{self.expected_hits:.2f} finish {self.finish_chance:.0%} "
            f"turns~{self.turns_to_finish:.1f} EV {self.expected_value:+.2f} "
            f"win {self.win_probability:.3f}"
        )


def position_of(sides: Mapping[str, Side]) -> RacePosition:
    """Public counts implied by the unrevealed key (same numbers as the score display)."""
    counts = {side: sum(1 for s in sides.values() if s is side) for side in Side}
    return RacePosition(
        counts[Side.FRIENDLY], counts[Side.ENEMY], counts[Side.NEUTRAL], counts[Side.ASSASSIN] > 0
    )


def _pick_weights(pool: Sequence[RankedWord], utility: ClueUtility, decay: float) -> list[float]:
    return [utility.pick_weight[item.fit] * decay**position for position, item in enumerate(pool)]


def outcomes(
    ranking: Sequence[RankedWord],
    sides: Mapping[str, Side],
    number: int,
    utility: ClueUtility = DEFAULT_UTILITY,
    *,
    verified: bool = False,
) -> list[Outcome]:
    """Exact outcome distribution of our operative playing the clue (<= 7 words, 3 deep)."""
    decay = utility.verified_position_decay if verified else utility.position_decay
    start = position_of(sides)

    def walk(remaining: Sequence[RankedWord], hits: int) -> list[tuple[float, int, Side | None]]:
        if hits >= number or not remaining:
            return [(1.0, hits, None)]
        if (
            hits
            and decide_current(
                remaining,
                number - hits,
                first_guess_required=False,
                position=start.after_hits(hits),
                model=utility.race,
            ).end_turn
        ):
            return [(1.0, hits, None)]
        pool = remaining if not hits else [r for r in remaining if r.fit is Fit.STRONG]
        weights = _pick_weights(pool, utility, decay)
        total = sum(weights)
        result: list[tuple[float, int, Side | None]] = []
        for item, weight in zip(pool, weights, strict=True):
            p = weight / total
            side = sides[item.word]
            if side is Side.FRIENDLY:
                rest = [r for r in remaining if r.word != item.word]
                result += [(p * q, h, s) for q, h, s in walk(rest, hits + 1)]
            else:
                result.append((p, hits, side))
        return result

    return [Outcome(p, h, s) for p, h, s in walk(list(ranking), 0)]


def _tactical(distribution: Sequence[Outcome], utility: ClueUtility) -> float:
    return sum(
        o.probability * (o.hits - (utility.miss_cost[o.ended_by] if o.ended_by else 0.0))
        for o in distribution
    )


def expected_value(
    ranking: Sequence[RankedWord],
    sides: Mapping[str, Side],
    number: int,
    utility: ClueUtility = DEFAULT_UTILITY,
    *,
    verified: bool = False,
) -> float:
    """Tactical value: expected friendly cards minus miss costs."""
    return _tactical(outcomes(ranking, sides, number, utility, verified=verified), utility)


def win_probability(
    distribution: Sequence[Outcome], race: RaceState, model: RaceModel = DEFAULT_RACE
) -> float:
    """Match win probability after this clue; the opponent moves next."""
    total = 0.0
    for o in distribution:
        if o.hits >= race.ours:
            value = 1.0  # our last card is revealed before any miss
        elif o.ended_by is Side.ASSASSIN:
            value = 0.0
        elif o.ended_by is Side.ENEMY:
            value = model.win_probability(race.ours - o.hits, race.theirs - 1, our_turn=False)
        else:
            value = model.win_probability(race.ours - o.hits, race.theirs, our_turn=False)
        total += o.probability * value
    return total


def likely_hits(
    ranking: Sequence[RankedWord],
    sides: Mapping[str, Side],
    number: int,
    model: RaceModel = DEFAULT_RACE,
) -> tuple[str, ...]:
    """Friendly words found if the operative follows the listed order exactly."""
    remaining = list(ranking)
    hits: list[str] = []
    start = position_of(sides)
    while len(hits) < number:
        decision = decide_current(
            remaining,
            number - len(hits),
            first_guess_required=not hits,
            position=start.after_hits(len(hits)),
            model=model,
        )
        if decision.word is None or sides[decision.word] is not Side.FRIENDLY:
            break
        hits.append(decision.word)
        remaining = [r for r in remaining if r.word != decision.word]
    return tuple(hits)


def assess_clue(
    clue: str,
    intended: Sequence[str],
    ranking: Sequence[RankedWord],
    sides: Mapping[str, Side],
    race: RaceState,
    utility: ClueUtility = DEFAULT_UTILITY,
    *,
    verified: bool = False,
) -> ClueAssessment:
    ranking = tuple(r for r in ranking if r.fit in utility.pick_weight)
    intended = tuple(intended)

    def result(
        number: int,
        dist: Sequence[Outcome],
        win: float,
        p_first: float,
        ok: bool,
        veto: bool,
        why: str,
    ) -> ClueAssessment:
        return ClueAssessment(
            clue=clue,
            number=number,
            intended=intended,
            ranking=ranking,
            verified=verified,
            expected_value=_tactical(dist, utility),
            expected_hits=sum(o.probability * o.hits for o in dist),
            p_first_friendly=p_first,
            finish_chance=sum(o.probability for o in dist if o.hits >= race.ours),
            turns_to_finish=sum(
                o.probability * (1 + utility.race.expected_turns(race.ours - o.hits)) for o in dist
            ),
            win_probability=win,
            hits=likely_hits(ranking, sides, number, utility.race) if ranking and not veto else (),
            acceptable=ok,
            vetoed=veto,
            reason=why,
        )

    if not ranking:
        return result(1, (), 0.0, 0.0, False, False, "blind operative connects no word")
    assassin = next((r for r in ranking if sides[r.word] is Side.ASSASSIN), None)
    if assassin is not None:
        why = f"assassin {assassin.word} is a {assassin.fit.value} match"
        return result(1, (), 0.0, 0.0, False, True, why)
    decay = utility.verified_position_decay if verified else utility.position_decay
    weights = _pick_weights(ranking, utility, decay)
    p_first = sum(
        w for r, w in zip(ranking, weights, strict=True) if sides[r.word] is Side.FRIENDLY
    ) / sum(weights)
    # The number states the plan: never more than the likely guessing sequence finds,
    # even if unlikely paths (a lucky early pick) would make a bigger number pay.
    plan = likely_hits(ranking, sides, min(utility.max_number, race.ours), utility.race)
    best: tuple[float, int, list[Outcome]] | None = None
    for number in range(1, max(1, len(plan)) + 1):
        dist = outcomes(ranking, sides, number, utility, verified=verified)
        win = win_probability(dist, race, utility.race)
        # Strictly better only: an equal value keeps the smaller, less exposed number.
        if best is None or win > best[0] + 1e-9:
            best = (win, number, dist)
    assert best is not None
    win, number, dist = best
    top = ranking[0]
    if sides[top.word] is not Side.FRIENDLY:
        why = f"operative's top pick {top.word} is {sides[top.word].value}"
        return result(number, dist, win, p_first, False, False, why)
    return result(number, dist, win, p_first, True, False, f"first guess ours {p_first:.0%}")


def select_clue(assessments: Sequence[ClueAssessment]) -> ClueAssessment | None:
    """Highest win probability among acceptable clues; else the least bad non-vetoed one.

    Never a vetoed clue. Near-equal win probabilities fall back to tactical EV, then to
    the earlier candidate.
    """
    acceptable = [a for a in assessments if a.acceptable]
    pool = acceptable or [a for a in assessments if not a.vetoed and a.ranking]
    if not pool:
        return None
    return max(pool, key=lambda a: (round(a.win_probability, 6), a.expected_value))


def explain_choice(
    choice: ClueAssessment, assessments: Sequence[ClueAssessment], race: RaceState
) -> str:
    """Why the choice beat the lowest-risk alternative, in race terms."""
    context = {
        "critical": "opponent can likely finish next turn",
        "behind": "we are behind in the race",
        "ahead": "we are ahead, so risk costs more than tempo",
        "even": "the race is close",
        "last card": "one card left",
    }[race.pressure]
    if choice.finish_chance >= 0.5:
        context += f"; this clue finishes the game with {choice.finish_chance:.0%} chance"
    rivals = [a for a in assessments if a.acceptable and a.clue != choice.clue]
    safest = min(rivals, key=lambda a: (a.risk, -a.win_probability), default=None)
    if safest is not None and safest.risk < choice.risk - 1e-6:
        context += (
            f"; lower-risk {safest.clue} {safest.number} (risk {safest.risk:.2f}) wins only "
            f"{safest.win_probability:.3f} vs {choice.win_probability:.3f}"
        )
    return context
