"""Offline decision harness: feed fixed boards to the real agents on local Ollama.

For a spymaster case it shows every candidate, its colour-blind ranking (annotated
with the key: F friendly, E enemy, N neutral, A assassin), the simulated outcome,
the selected clue, and then plays the REAL operative on that clue with the hidden
colours revealing each guess (a full offline turn). For an operative case it shows
the ranking, the decision and why. Every result records call count and latency.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from codenames_ai.agents.association import format_ranking
from codenames_ai.agents.base import LLMCall
from codenames_ai.agents.clue_scoring import WIN_TIE, Side, best_by_size, side_of
from codenames_ai.agents.operative import OperativeAgent
from codenames_ai.agents.spymaster import Candidate, SpymasterAgent
from codenames_ai.domain.enums import CardColor, Team
from codenames_ai.domain.models import (
    Card,
    Clue,
    PublicClueMemory,
    PublicGameState,
    SpymasterGameState,
)
from codenames_ai.llm.ollama import OllamaClient

from .cases import OperativeCase, SpymasterCase


@dataclass
class CaseResult:
    name: str
    kind: str
    purpose: str
    checks: dict[str, bool] = field(default_factory=dict)
    calls: list[LLMCall] = field(default_factory=list)
    seconds: float = 0.0
    lines: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and all(self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "passed": self.passed,
            "checks": self.checks,
            "error": self.error,
            "seconds": round(self.seconds, 2),
            "calls": [
                {"purpose": c.purpose, "seconds": round(c.seconds, 2), "ok": c.ok}
                for c in self.calls
            ],
            **self.data,
        }


def _labels(cards: tuple[Card, ...], case_team) -> dict[str, str]:
    return {
        c.word: side_of(c.color, case_team).value[0].upper()
        for c in cards
        if not c.revealed and c.color is not None
    }


def _reveal(cards: tuple[Card, ...], word: str) -> tuple[tuple[Card, ...], Card]:
    card = next(c for c in cards if c.word == word)
    shown = card.model_copy(update={"revealed": True})
    return tuple(shown if c.index == card.index else c for c in cards), shown


async def run_spymaster_case(case: SpymasterCase, llm: OllamaClient) -> CaseResult:
    result = CaseResult(case.name, "spymaster", case.purpose)
    cards = case.board.cards(case.name)
    labels = _labels(cards, case.team)
    spymaster = SpymasterAgent(case.team, llm)
    spymaster.memory = list(case.memory)
    # A remembered clue was given earlier, so it is also a used clue (as in a real game).
    spymaster.used_clues = {m.clue.casefold() for m in case.memory}
    given = (
        [Candidate(clue, tuple(targets), "given") for clue, targets in case.candidates]
        if case.candidates is not None
        else None
    )
    started = time.perf_counter()
    try:
        decision = await spymaster.choose_clue(
            SpymasterGameState(team=case.team, cards=cards), candidates=given
        )
    except ValueError as exc:
        result.error = str(exc)
        trace = spymaster.last_trace
        result.calls = list(trace.calls) if trace else []
        result.seconds = time.perf_counter() - started
        return result
    spy_seconds = time.perf_counter() - started
    trace = spymaster.last_trace
    assert trace is not None and trace.selected is not None
    chosen = trace.selected
    result.calls.extend(trace.calls)
    lines = result.lines
    lines.append(
        f"spymaster: {len(trace.calls)} calls ({', '.join(c.purpose for c in trace.calls)}), "
        f"{spy_seconds:.1f}s" + (" (fallback round used)" if trace.fallback_round else "")
    )
    assert trace.race is not None
    lines.append(f"race: {trace.race.describe()}")
    for clue, problem in trace.rejected:
        lines.append(f"  dropped {clue!r}: {problem}")
    for word, kinds in trace.words.items():
        lines.append(f"  assoc {word}: themes={kinds['themes']} specific={kinds['specific']}")
    sizes = trace.generated_sizes
    lines.append(
        "generated: "
        + " ".join(f"n{n}={k}" for n, k in sizes.items())
        + (
            f"  (focused regeneration: {trace.focused_regeneration})"
            if trace.focused_regeneration
            else ""
        )
    )
    by_clue: dict[str, list] = {}
    for a in trace.assessments:
        by_clue.setdefault(a.clue, []).append(a)
    screened_ranking = {a.clue: a.ranking for a in trace.screened}
    order = sorted(by_clue, key=lambda c: -max(a.win_probability for a in by_clue[c]))
    candidates = {c.clue: c for c in trace.candidates}
    for clue_word in order:
        actions = by_clue[clue_word]
        cand = candidates.get(clue_word)
        verified = actions[0].verified
        lines.append(
            f"  {clue_word} intended={list(cand.targets) if cand else '?'} "
            f"({cand.source if cand else ''}{': ' + cand.connection if cand and cand.connection else ''})"
        )
        lines.append(f"      screen: {format_ranking(screened_ranking.get(clue_word, ()), labels)}")
        if verified:
            lines.append(f"      VERIFY: {format_ranking(actions[0].ranking, labels)}")
        for a in sorted(actions, key=lambda a: a.number):
            mark = "*" if a is chosen else " "
            verdict = "ok" if a.acceptable else ("VETO" if a.vetoed else "no")
            lines.append(f"     {mark}{a.clue} {a.number} {verdict:4} {a.metrics()} | {a.reason}")
    best = best_by_size(trace.assessments)
    lines.append(
        "best by size: "
        + "; ".join(
            f"n{n} {a.clue} {a.number} win={a.win_probability:.3f}"
            + ("" if a.verified else " (screened)")
            for n, a in best.items()
        )
    )
    lines.append(
        f"selected: {decision.word} {decision.number}; expected hits {list(chosen.hits)}"
        + (f"  [{trace.note}]" if trace.note else "")
    )
    lines.append(f"reason: {trace.explanation}")

    # Real operative round trip on the chosen clue; colours only reveal outcomes.
    operative = OperativeAgent(case.team, llm)
    clue = Clue(word=decision.word, number=decision.number)
    guesses: list[tuple[str, str]] = []
    first_ranking = None
    won = False
    max_guesses = decision.number + 1
    op_started = time.perf_counter()
    # While the turn lasts every earlier guess was friendly (a miss ends it), so the
    # guess index is also the friendly-hit count.
    for made in range(max_guesses):
        memory = PublicClueMemory(
            clue=clue.word,
            number=clue.number,
            guesses_made=made,
            friendly_hits=made,
            unresolved_count=clue.number - made,
            words_guessed=tuple(w for w, _ in guesses),
        )
        # Fixture boards give BLUE 9 cards, so BLUE moved first (public information).
        state = PublicGameState(
            team=case.team,
            cards=cards,
            clue=clue,
            clue_history=(memory,),
            starting_team=Team.BLUE,
        )
        choice = await operative.choose_guesses(state, guesses_made=made, max_guesses=max_guesses)
        op_trace = operative.last_trace
        assert op_trace is not None
        result.calls.extend(op_trace.calls)
        if first_ranking is None:
            first_ranking = op_trace.ranking
        if choice.end_turn:
            lines.append(
                f"operative: END TURN ({op_trace.decision.reason if op_trace.decision else ''})"
            )
            break
        word = next(c.word for c in cards if c.index == choice.indices[0])
        cards, shown = _reveal(cards, word)
        side = side_of(shown.color, case.team)  # type: ignore[arg-type]
        guesses.append((word, side.value))
        lines.append(
            f"operative: {word} -> {side.value} ({op_trace.decision.reason if op_trace.decision else ''})"
            f" | {format_ranking(op_trace.ranking)}{' (reused)' if op_trace.reused_ranking else ''}"
        )
        if side is not Side.FRIENDLY:
            break
        if not any(side_of(c.color, case.team) is Side.FRIENDLY for c in cards if not c.revealed):  # type: ignore[arg-type]
            won = True
            lines.append("operative: revealed our last card -> WIN")
            break
        if made + 1 >= clue.number:
            break  # no old clues in this case: the controller ends guessing here
    result.seconds = time.perf_counter() - started
    lines.append(f"operative turn: {time.perf_counter() - op_started:.1f}s")
    if chosen.verified and first_ranking is not None:
        same = [(r.word, r.fit) for r in chosen.ranking] == [(r.word, r.fit) for r in first_ranking]
        lines.append(f"verified ranking identical to operative's: {same}")

    hit_words = [w for w, s in guesses if s == Side.FRIENDLY.value]
    result.checks = {
        f"number in {case.min_number}..{case.max_number}": case.min_number
        <= decision.number
        <= case.max_number,
        "selected clue acceptable": chosen.acceptable,
        "operative first guess friendly": bool(guesses) and guesses[0][1] == Side.FRIENDLY.value,
        "operative hit no enemy/assassin": all(
            s not in (Side.ENEMY.value, Side.ASSASSIN.value) for _, s in guesses
        ),
    }
    exact_match = [(r.word, r.fit) for r in chosen.ranking] == [
        (r.word, r.fit) for r in first_ranking or ()
    ]
    if chosen.verified:
        # The operative must follow the verified plan. (Exact ranking equality is
        # recorded separately: Ollama's prompt cache can flip a near-tie late in the
        # list even for a byte-identical request.)
        planned = list(chosen.hits)
        result.checks["operative follows verified plan"] = [w for w, _ in guesses][
            : len(planned)
        ] == planned[: len(guesses)] and bool(guesses)
    # Candidate identity: verification re-scores the same (clue, number) actions only.
    screened_actions: dict[str, set[int]] = {}
    for a in trace.screened:
        screened_actions.setdefault(a.clue, set()).add(a.number)
    verified_actions: dict[str, set[int]] = {}
    for a in trace.verified:
        verified_actions.setdefault(a.clue, set()).add(a.number)
    result.checks["verification preserved every (clue, number)"] = all(
        numbers == screened_actions.get(clue_word)
        for clue_word, numbers in verified_actions.items()
    )
    result.checks["selected action verified"] = chosen.verified
    top = max(
        (a.win_probability for a in trace.assessments if a.verified and a.acceptable), default=0
    )
    result.checks["selected action has the best verified win (within tie)"] = (
        not chosen.acceptable or chosen.win_probability >= top - WIN_TIE - 1e-9
    )
    if case.min_multi_generated and case.candidates is None:
        multi = sum(k for n, k in trace.generated_sizes.items() if n >= 2)
        result.checks[f"generated >= {case.min_multi_generated} multi-word plans"] = (
            multi >= case.min_multi_generated
        )
    if case.min_largest_generated and case.candidates is None:
        largest = max((n for n, k in trace.generated_sizes.items() if k), default=0)
        result.checks[f"attempted a {case.min_largest_generated}+-word plan"] = (
            largest >= case.min_largest_generated
        )
    if case.reject_clue is not None:
        result.checks[f"does not select {case.reject_clue}"] = decision.word != case.reject_clue
    if case.expect_clue is not None:
        result.checks[f"selects {case.expect_clue}"] = decision.word == case.expect_clue
    if case.expect_win:
        result.checks["wins this turn"] = won
    if case.target_pool is not None:
        result.checks["expected hits within target pool"] = (
            bool(chosen.hits) and set(chosen.hits) <= case.target_pool
        )
    if case.avoid:
        # Listed after all our words is harmless (the operative stops first); listed
        # before them, or guessed, is not.
        ahead_of_ours: set[str] = set()
        for item in chosen.ranking:
            if labels.get(item.word) == "F":
                break
            ahead_of_ours.add(item.word)
        result.checks["avoid words never guessed nor ranked above ours"] = not (
            case.avoid & (ahead_of_ours | {w for w, _ in guesses})
        )
    result.data = {
        "clue": decision.word,
        "number": decision.number,
        "expected_hits": list(chosen.hits),
        "candidates": len(trace.assessments),
        "acceptable": sum(a.acceptable for a in trace.assessments),
        "fallback_round": trace.fallback_round,
        "race": trace.race.describe(),
        "win_probability": round(chosen.win_probability, 4),
        "finish_chance": round(chosen.finish_chance, 4),
        "reason": trace.explanation,
        "operative_guesses": guesses,
        "operative_friendly_hits": len(hit_words),
        "won": won,
        "verified": chosen.verified,
        "exact_ranking_match": exact_match,
        "generated_sizes": trace.generated_sizes,
        "focused_regeneration": trace.focused_regeneration,
        "best_by_size": {
            n: [a.clue, a.number, round(a.win_probability, 4), a.verified] for n, a in best.items()
        },
    }
    return result


async def run_operative_case(case: OperativeCase, llm: OllamaClient) -> CaseResult:
    result = CaseResult(case.name, "operative", case.purpose)
    cards = case.board.cards(case.name)
    state = PublicGameState(
        team=case.team,
        cards=cards,
        clue=Clue(word=case.clue, number=case.number),
        clue_history=case.history,
        bonus_only=case.bonus_only,
        starting_team=Team.BLUE,
    )
    operative = OperativeAgent(case.team, llm)
    started = time.perf_counter()
    try:
        choice = await operative.choose_guesses(
            state, guesses_made=case.guesses_made, max_guesses=case.number + 1
        )
    except ValueError as exc:
        result.error = str(exc)
        return result
    result.seconds = time.perf_counter() - started
    trace = operative.last_trace
    assert trace is not None and trace.decision is not None
    result.calls = list(trace.calls)
    word = None if choice.end_turn else next(c.word for c in cards if c.index == choice.indices[0])
    color = next((c.color for c in cards if c.word == word), None)
    memory = "; ".join(
        f"{m.clue} {m.number} -> {m.unresolved_count} unresolved" for m in case.history
    )
    result.lines += [
        (
            f"clue {case.clue} {case.number}, guesses made {case.guesses_made}, "
            f"bonus only {case.bonus_only}"
        ),
        f"public memory: {memory or '-'}",
        f"ranking: {format_ranking(trace.ranking, show_clue=case.bonus_only)}",
        f"decision: {word or 'END TURN'}"
        + (f" for {choice.source_clue}" if word else "")
        + f" ({trace.decision.reason})"
        + (
            f" -> actually {color.value}"
            if color is not None and color != CardColor.UNKNOWN
            else ""
        ),
        f"{len(trace.calls)} calls, {result.seconds:.1f}s",
    ]
    if case.expect_guess is None:
        result.checks["ends turn"] = choice.end_turn
    else:
        result.checks["guesses an expected word"] = word in case.expect_guess
        if case.expect_source is not None:
            result.checks[f"credits old clue {case.expect_source}"] = (
                choice.source_clue == case.expect_source
            )
    result.data = {
        "decision": word or "end_turn",
        "source_clue": choice.source_clue,
        "reason": trace.decision.reason,
        "ranking": [(r.word, r.fit.value, r.clue) for r in trace.ranking],
    }
    return result
