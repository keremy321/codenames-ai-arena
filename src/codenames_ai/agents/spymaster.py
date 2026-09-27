"""Spymaster: the model proposes clues; our simulated operative and the race decide.

Normal cost is three qwen3 calls per clue:
1. generate: associations for each friendly word (specific and broad), then a few
   multi-target plans, with the race situation in the prompt. Code adds exact
   association overlaps and precise single-target clues from the same lists.
2. screen: one batched colour-blind ranking for up to 10 candidates.
3. verify: the best screened candidate is ranked again with the operative's exact
   single-clue request, which reproduces the operative's own ranking. A second or
   third finalist is verified only if the first disappoints.
Every candidate is scored locally by clue_scoring: simulated operative outcomes,
valued by match win probability (race.py); no model call is spent on strategy. Only
if no screened candidate is acceptable, one more screen runs on reserve single-target
ideas. Nothing bypasses the colour-blind check; there are no open-ended retries.
"""

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from codenames_ai.domain.enums import CardColor, Role, Team
from codenames_ai.domain.models import ClueDecision, SpymasterGameState
from codenames_ai.llm.ollama import OllamaClient, OllamaResponseError

from .association import (
    RANKER_TEMPERATURE,
    RANKING_SYSTEM,
    RankedWord,
    canonical_clue,
    format_ranking,
    parse_rankings,
    parse_single,
    ranking_schema,
    ranking_user,
    single_clue_request,
)
from .base import Agent, LLMCall, describe_calls, timed_chat
from .clue_scoring import (
    DEFAULT_UTILITY,
    ClueAssessment,
    ClueUtility,
    Side,
    assess_clue,
    explain_choice,
    select_clue,
    side_of,
)
from .race import RaceState

logger = logging.getLogger(__name__)

POOL_SIZE = 10
MAX_VERIFY = 3
# Batched screening misjudges multi-word clues most (it over- or under-lists
# competitors), so a finalist whose screened win probability is within this margin of
# the best verified clue is verified too, up to MAX_VERIFY. Each check is ~1-2 s.
VERIFY_MARGIN = 0.10
FINAL_CHECKS = 2
ASSOCIATIONS_PER_KIND = 3
GENERATOR_TEMPERATURE = 0.4
_CLUE_PATTERN = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*")

GUIDANCE = {
    "last card": "Only one of your words is left: give a precise clue for it.",
    "critical": (
        "The opponent will probably win on its next turn. Search hard for a clue that "
        "connects as many of your remaining words as possible, ideally all of them; a "
        "one-word clue now probably loses the game."
    ),
    "behind": (
        "You are behind. A series of one-word clues will likely lose the race: actively "
        "look for strong two- and three-word connections."
    ),
    "even": "The race is close: prefer clues that safely connect two or three of your words.",
    "ahead": (
        "You are ahead: prefer safe, efficient clues and take no unnecessary risk; the "
        "assassin and opponent words matter more than speed."
    ),
}


def _is_board_derivative(clue: str, board_words: set[str]) -> bool:
    clue = clue.casefold()
    for word in board_words:
        for part in re.findall(r"\w+", word):
            if clue == part:
                return True
            forms = {part + suffix for suffix in ("s", "es", "ed", "ing", "er", "est", "ly")}
            if part.endswith("e"):
                forms.update(part[:-1] + suffix for suffix in ("ing", "ed"))
            if part.endswith("y"):
                forms.add(part[:-1] + "ies")
            if len(part) >= 3 and part[-1] not in "aeiou":
                forms.update(part + part[-1] + suffix for suffix in ("ing", "ed", "er"))
            if clue in forms:
                return True
            # Compounds (SNOWMAN for SNOW, or SNOW for SNOWMAN) leak the word itself.
            if len(part) >= 4 and (clue.startswith(part) or clue.endswith(part)):
                return True
            if len(clue) >= 4 and (part.startswith(clue) or part.endswith(clue)):
                return True
    return False


def clue_problem(clue: str, board_words: set[str]) -> str | None:
    """Local rule check; returns why a clue word is illegal, or None."""
    if not _CLUE_PATTERN.fullmatch(clue) or len(clue) > 24:
        return "not a single plain word"
    if _is_board_derivative(clue, board_words):
        return "repeats, contains, or derives from a board word"
    return None


def _stem(word: str) -> str:
    word = word.casefold().strip()
    for suffix in ("ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


@dataclass(frozen=True)
class Candidate:
    clue: str
    targets: tuple[str, ...]
    source: str  # model | overlap | specific | given


@dataclass(frozen=True)
class PrivateClue:
    """Spymaster-only memory: what a past clue was expected to find."""

    clue: str
    number: int
    intended: tuple[str, ...]
    expected: tuple[str, ...]


@dataclass
class SpymasterTrace:
    team: Team
    race: RaceState | None = None
    calls: list[LLMCall] = field(default_factory=list)
    associations: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    screened: list[ClueAssessment] = field(default_factory=list)
    verified: list[ClueAssessment] = field(default_factory=list)
    assessments: list[ClueAssessment] = field(default_factory=list)  # final estimates
    selected: ClueAssessment | None = None
    fallback_round: bool = False
    explanation: str = ""
    note: str = ""


@dataclass(frozen=True)
class _Board:
    friendly: tuple[str, ...]
    enemy: tuple[str, ...]
    neutral: tuple[str, ...]
    assassin: tuple[str, ...]
    unrevealed: tuple[str, ...]  # card-index order, exactly as the operative lists them
    all_words: frozenset[str]
    sides: Mapping[str, Side]


class SpymasterAgent(Agent):
    def __init__(
        self, team: Team, llm: OllamaClient, *, utility: ClueUtility = DEFAULT_UTILITY
    ) -> None:
        super().__init__(team, Role.SPYMASTER, llm)
        self.utility = utility
        self.used_clues: set[str] = set()
        self.memory: list[PrivateClue] = []
        self.last_trace: SpymasterTrace | None = None

    def _board(self, state: SpymasterGameState) -> _Board:
        if type(state) is not SpymasterGameState or state.team != self.team:
            raise ValueError("Spymaster requires its own team's SpymasterGameState")
        if len(state.cards) != 25 or any(
            card.color in (None, CardColor.UNKNOWN) for card in state.cards if not card.revealed
        ):
            raise ValueError("Spymaster board is incomplete or has unknown hidden colors")
        sides = {
            card.word: side_of(card.color, self.team)  # type: ignore[arg-type]
            for card in sorted(state.cards, key=lambda c: c.index)
            if not card.revealed
        }

        def of(side: Side) -> tuple[str, ...]:
            return tuple(word for word, s in sides.items() if s is side)

        if not of(Side.FRIENDLY):
            raise ValueError(
                "No unrevealed friendly cards; reread game state before requesting a clue"
            )
        if not of(Side.ENEMY):
            raise ValueError("Opponent has no cards left; the game should be over")
        return _Board(
            friendly=of(Side.FRIENDLY),
            enemy=of(Side.ENEMY),
            neutral=of(Side.NEUTRAL),
            assassin=of(Side.ASSASSIN),
            unrevealed=tuple(sides),
            all_words=frozenset(card.word.casefold() for card in state.cards),
            sides=sides,
        )

    def missed_targets(self, state: SpymasterGameState) -> list[tuple[str, tuple[str, ...]]]:
        """Private: earlier clues whose expected words are still unrevealed and ours."""
        ours = {
            card.word
            for card in state.cards
            if not card.revealed and card.color == CardColor(self.team.value)
        }
        missed = []
        for record in self.memory:
            words = tuple(word for word in record.expected if word in ours)
            if words:
                missed.append((record.clue, words))
        return missed

    async def choose_clue(
        self, state: SpymasterGameState, *, candidates: Sequence[Candidate] | None = None
    ) -> ClueDecision:
        """Pick a clue. ``candidates`` replaces generation (offline evaluation only)."""
        board = self._board(state)
        race = RaceState.of(len(board.friendly), len(board.enemy), self.utility.race)
        trace = SpymasterTrace(self.team, race)
        self.last_trace = trace
        logger.info("[%s] race: %s", self.team, race.describe())
        if candidates is None:
            pool = await self._candidates(board, race, trace, self.missed_targets(state))
        else:
            given = [(c.clue, c.targets, "given") for c in candidates]
            pool = self._prepare(board, trace, given)
        first, reserve = pool[:POOL_SIZE], pool[POOL_SIZE:]
        screened = await self._screen(board, race, trace, first)
        if not any(a.acceptable for a in screened) and reserve:
            trace.fallback_round = True
            logger.info("[%s] no acceptable clue; screening reserve clues", self.team)
            screened += await self._screen(board, race, trace, reserve[:POOL_SIZE])
        trace.screened = screened
        trace.assessments = await self._verify(board, race, trace, screened)
        choice = select_clue(trace.assessments)
        if choice is None:
            trace.note = "every candidate was invalid, unassessed, or vetoed"
            logger.error("[%s] no safe clue: %s", self.team, describe_calls(trace.calls))
            raise ValueError(f"No safe clue after {len(trace.calls)} Ollama calls: {trace.note}")
        if not choice.acceptable:
            trace.note = "no acceptable clue; using the least bad non-vetoed candidate"
            logger.warning("[%s] %s", self.team, trace.note)
        trace.selected = choice
        trace.explanation = explain_choice(choice, trace.assessments, race)
        self.used_clues.add(choice.clue.casefold())
        self.memory.append(PrivateClue(choice.clue, choice.number, choice.intended, choice.hits))
        logger.info(
            "[%s] selected %s %d (%s; %s); expected %s; intended %s; reason: %s; %s",
            self.team,
            choice.clue,
            choice.number,
            "verified" if choice.verified else "screened only",
            choice.metrics(),
            list(choice.hits),
            list(choice.intended),
            trace.explanation,
            describe_calls(trace.calls),
        )
        return ClueDecision(word=choice.clue, number=choice.number)

    async def _candidates(
        self,
        board: _Board,
        race: RaceState,
        trace: SpymasterTrace,
        missed: Sequence[tuple[str, tuple[str, ...]]],
    ) -> list[Candidate]:
        """One generation call (one retry only if its JSON is unusable)."""
        reply: dict[str, Any] | None = None
        for attempt in range(2):
            try:
                reply = await self._generate(board, race, trace, missed)
                break
            except OllamaResponseError as exc:
                logger.warning("[%s] generation attempt %d failed: %s", self.team, attempt + 1, exc)
        if reply is None:
            return []
        friendly = {word.casefold(): word for word in board.friendly}
        associations: dict[str, dict[str, list[str]]] = {}
        raw_assoc = reply.get("associations")
        for word in board.friendly:
            entry = raw_assoc.get(word) if isinstance(raw_assoc, dict) else None
            entry = entry if isinstance(entry, dict) else {}
            associations[word] = {
                kind: [str(a).strip() for a in entry.get(kind, []) if isinstance(a, str)]
                for kind in ("specific", "broad")
            }
        trace.associations = associations

        ordered: list[tuple[str, Sequence[str], str]] = []
        raw_plans = reply.get("candidates")
        for plan in raw_plans if isinstance(raw_plans, list) else []:
            if not isinstance(plan, dict):
                continue
            targets = [
                friendly[t.casefold()]
                for t in plan.get("targets", [])
                if isinstance(t, str) and t.casefold() in friendly
            ]
            ordered.append((str(plan.get("clue", "")).strip(), targets, "model"))
        # Exact overlaps between association lists: a shared idea for 2-3 of our words.
        by_stem: dict[str, tuple[str, list[str]]] = {}
        for word, kinds in associations.items():
            for assoc in kinds["broad"] + kinds["specific"]:
                _, words = by_stem.setdefault(_stem(assoc), (assoc, []))
                if word not in words:
                    words.append(word)
        overlaps = sorted(
            ((spelled, words) for spelled, words in by_stem.values() if len(words) >= 2),
            key=lambda item: -len(item[1]),
        )
        ordered += [(clue, targets, "overlap") for clue, targets in overlaps]
        # Precise single-target clues: first choices first, then second choices...
        for rank in range(ASSOCIATIONS_PER_KIND):
            for word, kinds in associations.items():
                if rank < len(kinds["specific"]):
                    ordered.append((kinds["specific"][rank], [word], "specific"))
        return self._prepare(board, trace, ordered)

    def _prepare(
        self,
        board: _Board,
        trace: SpymasterTrace,
        ordered: Sequence[tuple[str, Sequence[str], str]],
    ) -> list[Candidate]:
        """Legal, distinct, never-used clues in canonical spelling; multi-target first."""
        friendly = {word.casefold(): word for word in board.friendly}
        candidates: list[Candidate] = []
        # A repeated clue word would make the team's public clue memory ambiguous.
        seen: set[str] = {_stem(clue) for clue in self.used_clues}
        board_words = set(board.all_words)
        for clue, raw_targets, source in ordered:
            targets = tuple(
                dict.fromkeys(
                    friendly[t.casefold()] for t in raw_targets if t.casefold() in friendly
                )
            )
            problem = clue_problem(clue, board_words)
            if problem is None and not targets:
                problem = "no valid friendly targets"
            if problem is None and _stem(clue) in seen:
                continue  # quietly skip duplicates of a better-ranked candidate
            if problem is not None:
                trace.rejected.append((clue or "<empty>", f"{source}: {problem}"))
                continue
            seen.add(_stem(clue))
            candidates.append(
                Candidate(canonical_clue(clue), targets[: self.utility.max_number], source)
            )
        trace.candidates = candidates
        logger.info(
            "[%s] %d candidates (%s); %d dropped",
            self.team,
            len(candidates),
            ", ".join(
                f"{sum(c.source == src for c in candidates)} {src}"
                for src in ("model", "overlap", "specific", "given")
                if any(c.source == src for c in candidates)
            ),
            len(trace.rejected),
        )
        # Multi-target ideas first, then precise singles; singles fill the reserve.
        return sorted(candidates, key=lambda c: (len(c.targets) == 1, c.source == "specific"))

    async def _generate(
        self,
        board: _Board,
        race: RaceState,
        trace: SpymasterTrace,
        missed: Sequence[tuple[str, tuple[str, ...]]],
    ) -> dict[str, Any]:
        association_list = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": ASSOCIATIONS_PER_KIND,
            "maxItems": ASSOCIATIONS_PER_KIND,
        }
        properties: dict[str, Any] = {
            "associations": {
                "type": "object",
                "properties": {
                    word: {
                        "type": "object",
                        "properties": {"specific": association_list, "broad": association_list},
                        "required": ["specific", "broad"],
                        "additionalProperties": False,
                    }
                    for word in board.friendly
                },
                "required": list(board.friendly),
                "additionalProperties": False,
            }
        }
        multi = len(board.friendly) >= 2
        if multi:
            properties["candidates"] = {
                "type": "array",
                "minItems": 3,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "targets": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(board.friendly)},
                            "minItems": 2,
                            "maxItems": min(self.utility.max_number, len(board.friendly)),
                        },
                        "clue": {"type": "string"},
                    },
                    "required": ["targets", "clue"],
                    "additionalProperties": False,
                },
            }
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        system = (
            f"You are the {self.team.value.upper()} spymaster in Codenames, racing the other "
            "team to reveal all of your words first. A clue is ONE real English dictionary "
            "word (no spaces, no made-up compounds) that is not a board word, part of one, or "
            "a form of one. First, for each of your words give single-word clue ideas: "
            "'specific' ideas point to that word alone; 'broad' ideas are categories or "
            "themes that could also cover other words."
        )
        if multi:
            system += (
                " Then propose clues that connect 2 or 3 of your words through one real, "
                "common association (category, synonym, well-known phrase, shared property). "
                "Never force a link because words are yours."
            )
        system += (
            " Avoid any idea that also fits an opponent word, a neutral word, or above all "
            "the assassin."
        )
        lines = [
            f"YOUR words: {', '.join(board.friendly)}",
            f"OPPONENT words: {', '.join(board.enemy) or '-'}",
            f"NEUTRAL words: {', '.join(board.neutral) or '-'}",
            f"ASSASSIN (never hint at it): {', '.join(board.assassin) or '-'}",
            f"RACE: you have {race.ours} words left, the opponent has {race.theirs}. "
            + GUIDANCE[race.pressure],
        ]
        if 2 <= race.ours <= self.utility.max_number:
            lines.append(
                f"One clue connecting all {race.ours} of your remaining words would win the "
                "game now; include such a candidate if any real link exists."
            )
        if missed:
            lines.append(
                "Earlier clues of yours whose words are still unfound (a new clue may link "
                "these words again): "
                + "; ".join(f"{clue} -> {', '.join(words)}" for clue, words in missed)
            )
        if self.used_clues:
            lines.append(f"Clues already used: {', '.join(sorted(self.used_clues))}")
        return await timed_chat(
            self.llm,
            trace.calls,
            "generate",
            system=system,
            user="\n".join(lines),
            schema=schema,
            num_predict=900,
            temperature=GENERATOR_TEMPERATURE,
        )

    def _log(self, stage: str, board: _Board, candidate: Candidate, a: ClueAssessment) -> None:
        labels = {w: s.value[0].upper() for w, s in board.sides.items()}
        logger.info(
            "[%s]   %s %s %d (%s, intended %s): %s | %s | %s",
            self.team,
            stage,
            a.clue,
            a.number,
            candidate.source,
            list(candidate.targets),
            a.metrics(),
            "ok" if a.acceptable else f"rejected: {a.reason}",
            format_ranking(a.ranking, labels),
        )

    async def _screen(
        self,
        board: _Board,
        race: RaceState,
        trace: SpymasterTrace,
        candidates: Sequence[Candidate],
    ) -> list[ClueAssessment]:
        if not candidates:
            return []
        rankings = await self._batched_rankings(board, trace, candidates)
        if rankings is None:
            return []  # unassessed candidates are never used
        assessments = []
        for candidate in candidates:
            ranking = rankings.get(candidate.clue, ())
            assessment = assess_clue(
                candidate.clue, candidate.targets, ranking, board.sides, race, self.utility
            )
            assessments.append(assessment)
            self._log("screen", board, candidate, assessment)
        return assessments

    async def _verify(
        self,
        board: _Board,
        race: RaceState,
        trace: SpymasterTrace,
        screened: Sequence[ClueAssessment],
    ) -> list[ClueAssessment]:
        """Re-rank the most promising candidates with the operative's exact request.

        The screened (batched) ranking is only an estimate: batching and clue spelling
        change qwen3's answer. The verified ranking is what the operative will see.
        """
        finalists = sorted(
            (a for a in screened if not a.vetoed and a.ranking),
            key=lambda a: (a.acceptable, a.win_probability),
            reverse=True,
        )
        final = {a.clue: a for a in screened}
        for position, screened_one in enumerate(finalists[:MAX_VERIFY]):
            verified = await self._verify_one(board, race, trace, screened_one)
            if verified is None:
                continue
            final[verified.clue] = verified
            best = select_clue([v for v in trace.verified if v.acceptable])
            upcoming = finalists[position + 1] if position + 1 < len(finalists) else None
            # Stop once no unverified finalist could plausibly beat the verified best.
            if best is not None and (
                upcoming is None or upcoming.win_probability < best.win_probability - VERIFY_MARGIN
            ):
                break
        # Never submit a clue whose ranking was only screened: if the best estimate is
        # still unverified (the verified finalists disappointed), verify it and re-select.
        for _ in range(FINAL_CHECKS):
            choice = select_clue(list(final.values()))
            if choice is None or choice.verified:
                break
            verified = await self._verify_one(board, race, trace, choice)
            if verified is None:
                break
            final[verified.clue] = verified
        return list(final.values())

    async def _verify_one(
        self, board: _Board, race: RaceState, trace: SpymasterTrace, screened: ClueAssessment
    ) -> ClueAssessment | None:
        request = single_clue_request(screened.clue, board.unrevealed)
        try:
            reply = await timed_chat(self.llm, trace.calls, "verify", **request)
        except OllamaResponseError as exc:
            logger.warning("[%s] verifying %s failed: %s", self.team, screened.clue, exc)
            return None
        verified = assess_clue(
            screened.clue,
            screened.intended,
            parse_single(reply, screened.clue, board.unrevealed),
            board.sides,
            race,
            self.utility,
            verified=True,
        )
        trace.verified.append(verified)
        candidate = next(c for c in trace.candidates if c.clue == verified.clue)
        self._log("verify", board, candidate, verified)
        return verified

    async def _batched_rankings(
        self, board: _Board, trace: SpymasterTrace, candidates: Sequence[Candidate]
    ) -> dict[str, tuple[RankedWord, ...]] | None:
        # Colour-blind on purpose: only unrevealed words and clue words, never the key
        # or the intended targets, so the ranking predicts what our operative will do.
        clues = [c.clue for c in candidates]
        try:
            reply = await timed_chat(
                self.llm,
                trace.calls,
                "screen",
                system=RANKING_SYSTEM,
                user=ranking_user(board.unrevealed, clues),
                schema=ranking_schema(clues, board.unrevealed),
                num_predict=60 + 60 * len(clues),
                temperature=RANKER_TEMPERATURE,
            )
        except OllamaResponseError as exc:
            logger.warning("[%s] screening failed: %s", self.team, exc)
            return None
        return parse_rankings(reply, clues, board.unrevealed)
