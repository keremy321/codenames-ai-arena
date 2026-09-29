"""Score a clue ACTION by simulating our operative, then valuing the outcome in the race.

An action is the pair (clue, number): "MOTION 3" and "MOTION 1" are different moves
and are scored separately. Scoring never changes the number it is given; the numbers
worth considering for a clue are listed once (action_numbers) and each is kept as its
own candidate through screening, verification and selection.

Tactical layer: for each clue the spymaster has a colour-blind ranking in the real
operative's format. We enumerate, exactly, what our operative does with the action:

* each guess picks among the listed words with probability proportional to its tier
  weight times a decay per list position. A screened (batched) ranking only roughly
  predicts the operative's order; a verified ranking is the operative's own request,
  reproduced byte for byte, so its order is trusted almost fully;
* after the first guess it continues only when guess_policy.decide_current says so
  (the same code the real operative runs, with the clue's number);
* the turn ends at the first non-friendly card.

Strategic layer: each outcome (hits, card that ended the turn) leads to a race
position whose win probability race.RaceModel gives. An action's value is the win
probability averaged over its outcomes. Failure types differ through the race itself:
a neutral card only ends the turn; an opponent card also advances the opponent (and
loses if it was their last); the assassin loses the game.

Target consistency: the first ``number`` listed words must share enough words with the
generator's intended targets (target_consistency). A clue whose predicted guesses are
friendly but unrelated to what it was generated for is not acceptable: its meaning
drifted, and the lucky prediction is the least trustworthy part of the simulation.

Assassin handling: listed as strong -> hard veto. Listed as possible -> kept, but its
pick weight gets no position discount (tier and order noise matter most for a
catastrophe), so it costs a large, race-dependent share of win probability.
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
    # Largest clue number considered. Four is rare (it needs four listed friendly words
    # the operative will actually reach) but can be the only winning move at the end.
    max_number: int = 4
    # Relative chance the operative picks a word of this tier over another candidate.
    pick_weight: Mapping[Fit, float] = field(
        default_factory=lambda: MappingProxyType(
            {Fit.STRONG: 1.0, Fit.POSSIBLE: 0.1, Fit.WEAK: 0.05}
        )
    )
    # Factor per position further down the list. Batched rankings only roughly match
    # the operative's order; verified rankings reproduced exactly in almost every probe,
    # and 0.1 keeps a small allowance for a different board read or clue spelling.
    position_decay: float = 0.6
    verified_position_decay: float = 0.1
    # Tactical EV only (logs and tie-breaks); selection uses race win probability.
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
    finish_chance: float  # P(this action clears all our remaining cards)
    turns_to_finish: float  # this turn plus expected later turns
    win_probability: float
    miss_chances: Mapping[Side, float]  # P(the turn ends on each wrong card type)
    hits: tuple[str, ...]  # friendly words the operative most likely finds
    acceptable: bool
    vetoed: bool
    reason: str
    target_overlap: int = 0  # intended words among the first ``number`` listed words
    target_consistent: bool = True

    @property
    def action(self) -> tuple[str, int]:
        return (self.clue, self.number)

    @property
    def target_precision(self) -> float:
        """Share of the predicted guesses (the first ``number`` listed words; fewer when
        fewer are listed) that were intended. Logs only; selection uses the overlap."""
        predicted = min(self.number, len(self.ranking))
        return self.target_overlap / predicted if predicted else 0.0

    @property
    def risk(self) -> float:
        """Expected miss cost of the action (tactical)."""
        return self.expected_hits - self.expected_value

    def metrics(self) -> str:
        misses = " ".join(
            f"{side.value[0].upper()}{p:.0%}" for side, p in self.miss_chances.items() if p >= 0.005
        )
        return (
            f"hits~{self.expected_hits:.2f} finish {self.finish_chance:.0%} "
            f"turns~{self.turns_to_finish:.1f} miss[{misses or '-'}] win {self.win_probability:.3f}"
        )


def position_of(sides: Mapping[str, Side]) -> RacePosition:
    """Public counts implied by the unrevealed key (same numbers as the score display)."""
    counts = {side: sum(1 for s in sides.values() if s is side) for side in Side}
    return RacePosition(
        counts[Side.FRIENDLY], counts[Side.ENEMY], counts[Side.NEUTRAL], counts[Side.ASSASSIN] > 0
    )


def _pick_weights(
    pool: Sequence[RankedWord], sides: Mapping[str, Side], utility: ClueUtility, decay: float
) -> list[float]:
    return [
        utility.pick_weight[item.fit]
        * (1.0 if sides[item.word] is Side.ASSASSIN else decay**position)
        for position, item in enumerate(pool)
    ]


def outcomes(
    ranking: Sequence[RankedWord],
    sides: Mapping[str, Side],
    number: int,
    utility: ClueUtility = DEFAULT_UTILITY,
    *,
    verified: bool = False,
) -> list[Outcome]:
    """Exact outcome distribution of our operative playing (clue, number)."""
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
        weights = _pick_weights(pool, sides, utility, decay)
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
    """Match win probability after this action; the opponent moves next.

    Only the future turns use the generic progress model; this turn is valued from
    the action's own predicted outcomes.
    """
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


def required_overlap(intended: int, number: int) -> int:
    """At least one intended word, and a third of the comparable ones (1, 1, 1, 2)."""
    return (min(intended, number) + 2) // 3


def target_consistency(
    ranking: Sequence[RankedWord], intended: Sequence[str], number: int
) -> tuple[int, bool]:
    """(overlap, passed): intended words among the operative's first ``number`` picks.

    Without generator targets (none recorded) there is nothing to compare: it passes.
    """
    predicted = {r.word for r in ranking[:number]}
    overlap = len(predicted & set(intended))
    return overlap, overlap >= required_overlap(len(intended), number)


def action_numbers(
    intended: int,
    ranking: Sequence[RankedWord],
    sides: Mapping[str, Side],
    race: RaceState,
    utility: ClueUtility = DEFAULT_UTILITY,
) -> list[int]:
    """Numbers worth trying for a clue: 1 up to its intended size or its listed friendly
    words, whichever is larger, capped at min(max_number, our cards left).

    Each becomes a separate action; the smaller ones are the fallbacks of the larger.
    """
    listed = sum(1 for r in ranking if sides.get(r.word) is Side.FRIENDLY)
    cap = min(utility.max_number, race.ours)
    return list(range(1, max(1, min(cap, max(intended, listed))) + 1))


def assess_action(
    clue: str,
    number: int,
    intended: Sequence[str],
    ranking: Sequence[RankedWord],
    sides: Mapping[str, Side],
    race: RaceState,
    utility: ClueUtility = DEFAULT_UTILITY,
    *,
    verified: bool = False,
) -> ClueAssessment:
    """Score exactly the action (clue, number); the number is never changed here."""
    if not 1 <= number <= min(utility.max_number, race.ours):
        raise ValueError(f"Illegal clue number {number} with {race.ours} cards left")
    ranking = tuple(r for r in ranking if r.fit in utility.pick_weight)
    intended = tuple(intended)
    overlap, consistent = target_consistency(ranking, intended, number)

    def result(
        dist: Sequence[Outcome], win: float, p_first: float, ok: bool, veto: bool, why: str
    ) -> ClueAssessment:
        misses = {
            side: sum(o.probability for o in dist if o.ended_by is side)
            for side in (Side.ASSASSIN, Side.ENEMY, Side.NEUTRAL)
        }
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
            miss_chances=MappingProxyType(misses),
            hits=likely_hits(ranking, sides, number, utility.race) if ranking and not veto else (),
            acceptable=ok,
            vetoed=veto,
            reason=why,
            target_overlap=overlap,
            target_consistent=consistent,
        )

    if not ranking:
        return result((), 0.0, 0.0, False, False, "blind operative connects no word")
    assassin = next((r for r in ranking if sides[r.word] is Side.ASSASSIN), None)
    if assassin is not None and assassin.fit is Fit.STRONG:
        return result((), 0.0, 0.0, False, True, f"assassin {assassin.word} is a strong match")
    decay = utility.verified_position_decay if verified else utility.position_decay
    weights = _pick_weights(ranking, sides, utility, decay)
    p_first = sum(
        w for r, w in zip(ranking, weights, strict=True) if sides[r.word] is Side.FRIENDLY
    ) / sum(weights)
    dist = outcomes(ranking, sides, number, utility, verified=verified)
    win = win_probability(dist, race, utility.race)
    top = ranking[0]
    if sides[top.word] is not Side.FRIENDLY:
        why = f"operative's top pick {top.word} is {sides[top.word].value}"
        return result(dist, win, p_first, False, False, why)
    if not consistent:
        predicted = [r.word for r in ranking[:number]]
        why = (
            f"target mismatch: predicted {predicted} share {overlap} of the "
            f"{required_overlap(len(intended), number)} needed with intended {list(intended)}"
        )
        return result(dist, win, p_first, False, False, why)
    why = f"first guess ours {p_first:.0%}"
    if assassin is not None:
        why += f"; assassin {assassin.word} listed as possible (priced as a loss)"
    return result(dist, win, p_first, True, False, why)


def assess_actions(
    clue: str,
    intended: Sequence[str],
    ranking: Sequence[RankedWord],
    sides: Mapping[str, Side],
    race: RaceState,
    utility: ClueUtility = DEFAULT_UTILITY,
    *,
    numbers: Sequence[int] | None = None,
    verified: bool = False,
) -> list[ClueAssessment]:
    """Score every action of a clue; ``numbers`` fixes the action set (verification)."""
    if numbers is None:
        numbers = action_numbers(len(intended), ranking, sides, race, utility)
    return [
        assess_action(clue, n, intended, ranking, sides, race, utility, verified=verified)
        for n in numbers
    ]


# Win probabilities closer than one percentage point are a tie: the race prior and the
# ranking noise make the model no more precise than that, and a bigger number that
# wins only through rare paths or "the turn was ending anyway" guesses would overstate
# the clue (sending the operative into words it would otherwise leave alone and leaving
# a phantom unresolved count in public memory).
WIN_TIE = 0.01


def _key(a: ClueAssessment) -> tuple[float, int, float]:
    return (round(a.win_probability, 6), -a.number, a.expected_value)


def _pick(pool: Sequence[ClueAssessment]) -> ClueAssessment:
    top = max(a.win_probability for a in pool)
    near = [a for a in pool if a.win_probability >= top - WIN_TIE]
    # Among near-ties: the smaller, less exposed number, then the higher win.
    return min(near, key=lambda a: (a.number, -a.win_probability, -a.expected_value))


def select_clue(assessments: Sequence[ClueAssessment]) -> ClueAssessment | None:
    """Highest win probability among acceptable actions (near-ties go to the smaller
    number); else the least bad non-vetoed one; never a vetoed one."""
    acceptable = [a for a in assessments if a.acceptable]
    pool = acceptable or [a for a in assessments if not a.vetoed and a.ranking]
    if not pool:
        return None
    return _pick(pool)


def prune_dominated(assessments: Sequence[ClueAssessment]) -> list[ClueAssessment]:
    """Drop (clue, n) when a smaller number of the SAME clue is as good: win within
    WIN_TIE and less than 0.05 extra expected friendly cards. Such an action could never
    be selected (select_clue prefers the smaller number on a near-tie), so pruning only
    removes redundant work and log noise. Numbers are never changed, only dropped.

    Only an action at least as acceptable dominates: target consistency differs by
    number (MOTION 1 can miss its targets while MOTION 2 reaches one), and select_clue
    never prefers an unacceptable action to an acceptable one."""
    kept: list[ClueAssessment] = []
    by_clue: dict[str, list[ClueAssessment]] = {}
    for a in assessments:
        by_clue.setdefault(a.clue, []).append(a)
    for actions in by_clue.values():
        survivors: list[ClueAssessment] = []
        for a in sorted(actions, key=lambda x: x.number):
            dominated = any(
                b.win_probability >= a.win_probability - WIN_TIE
                and a.expected_hits - b.expected_hits < 0.05
                and (b.acceptable or not a.acceptable)
                for b in survivors
            )
            if not dominated:
                survivors.append(a)
        kept += survivors
    return kept


def best_by_size(assessments: Sequence[ClueAssessment]) -> dict[int, ClueAssessment]:
    """Best acceptable action for each clue number (diagnostics and explanations)."""
    best: dict[int, ClueAssessment] = {}
    for a in assessments:
        if a.acceptable and (a.number not in best or _key(a) > _key(best[a.number])):
            best[a.number] = a
    return dict(sorted(best.items()))


def explain_choice(
    choice: ClueAssessment, assessments: Sequence[ClueAssessment], race: RaceState
) -> str:
    """Why the choice won, in race terms."""
    context = {
        "critical": (f"opponent has {race.their_finish_chance:.0%} estimated finish-next chance"),
        "behind": "we are behind in the race",
        "ahead": "we are ahead, so risk costs more than tempo",
        "even": "the race is close",
        "last card": "one card left",
    }[race.pressure]
    if choice.finish_chance >= 0.5:
        context += f"; this action finishes the game with {choice.finish_chance:.0%} chance"
    sizes = best_by_size(assessments)
    others = [a for n, a in sizes.items() if n != choice.number]
    if others:
        runner = max(others, key=_key)
        verb = "would leave" if runner.number < choice.number else "only reaches"
        context += (
            f"; best {runner.number}-card {runner.clue} {verb} win "
            f"{runner.win_probability:.3f} vs {choice.win_probability:.3f}"
        )
    return context
