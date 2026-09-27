"""Spymaster: search target groups, keep each (clue, number) action intact, pick the
action with the highest match win probability.

Per clue the normal cost is 3-6 qwen3 calls:
1. generate: per friendly word two themes (categories, functions, properties) and two
   specific single-word ideas, then 4-8 target GROUPS (2 up to min(4, our cards) words)
   with a clue and a short private connection. The race situation sets the search
   order; in a critical endgame it asks for an all-remaining group first.
   Code adds exact theme overlaps (multi-word) and specific single-word fallbacks.
2. focused regeneration, at most once and only if multi-card plans are nearly absent
   when behind/critical, or a critical endgame has no all-in attempt.
3. screen: one batched colour-blind ranking; multi-word clues fill most of the pool,
   single-word fallbacks the rest.
4. verify (1-3 short calls, +2 at most for the final choice): the operative's exact
   single-clue request. It re-scores every action of that clue with the SAME numbers.
Every action is scored locally (clue_scoring + race); no call is spent on strategy.
Nothing bypasses the colour-blind check, and there are no open-ended retries.
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
    assess_actions,
    best_by_size,
    explain_choice,
    select_clue,
    side_of,
)
from .race import RaceState

logger = logging.getLogger(__name__)

POOL_SIZE = 10  # clue words per screening call
POOL_SINGLES = 3  # single-word fallbacks in the first screen when multi-word ideas exist
MAX_VERIFY = 3
# Batched screening misjudges multi-word clues most (it over- or under-lists
# competitors), so a finalist whose screened win probability is within this margin of
# the best verified action is verified too, up to MAX_VERIFY. Each check is ~1-2 s.
VERIFY_MARGIN = 0.10
FINAL_CHECKS = 2
THEMES_PER_WORD = 2
SPECIFIC_PER_WORD = 2
GENERATOR_TEMPERATURE = 0.4
# Real clue words longer than this are rare; glued phrases (MUSICALINSTRUMENT) are not.
MAX_CLUE_LETTERS = 13
_CLUE_PATTERN = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*")
# Enforced by the JSON grammar for every generated idea and clue: one word, no spaces,
# no internal capitals. Without it qwen3 answers with phrases ("Erupting mountain"), and
# with a looser pattern it glues them into CamelCase ("EruptingMountain").
WORD_SCHEMA = {"type": "string", "pattern": "^[A-Za-z][a-z]*(-[a-z]+)?$"}

GUIDANCE = {
    "last card": "Only one of your words is left: give precise clues for it.",
    "critical": (
        "The opponent is very likely to win on its next turn. Search aggressively for a "
        "clue that can finish ALL of your remaining words now. If no credible clue covers "
        "all of them, find the strongest group one smaller, then smaller again. A safe "
        "one-word clue is strategically poor unless no credible multi-word option exists."
    ),
    "behind": (
        "You are behind. A series of one-word clues will lose the race: spend your effort "
        "on genuine two-, three- and (if natural) four-word groups."
    ),
    "even": "The race is close: look first for clues that safely connect two or three words.",
    "ahead": (
        "You are ahead: prefer safe, coherent groups of two or three; take no unnecessary "
        "risk with the assassin or opponent words."
    ),
}

STYLE = (
    "Good groups share ONE natural concept: a category, a function, a physical property, "
    "a cultural association, an action, or a domain. Bad clues: generic words like "
    "OBJECT, THING or ITEM; forcing unrelated words together; a link that only works "
    "because you know which words are yours; invented compounds. Prefer a smaller coherent "
    "group over a larger fake one, and never pad a group with an unrelated word."
)


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
    if not _CLUE_PATTERN.fullmatch(clue):
        return "not a single plain word"
    if len(clue) > MAX_CLUE_LETTERS:
        return "too long; likely a glued compound"
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
    """A clue word with the friendly words it was generated for."""

    clue: str
    targets: tuple[str, ...]
    source: str  # model | focus | overlap | specific | given
    connection: str = ""  # private generation note; never sent to any ranking request


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
    words: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    generated_sizes: dict[int, int] = field(default_factory=dict)
    focused_regeneration: str = ""  # why the one extra generation ran, if it did
    screened: list[ClueAssessment] = field(default_factory=list)
    verified: list[ClueAssessment] = field(default_factory=list)
    assessments: list[ClueAssessment] = field(default_factory=list)  # final estimates
    selected: ClueAssessment | None = None
    fallback_round: bool = False
    explanation: str = ""
    note: str = ""

    @property
    def associations(self) -> dict[str, dict[str, list[str]]]:  # harness compatibility
        return self.words


@dataclass(frozen=True)
class _Board:
    friendly: tuple[str, ...]
    enemy: tuple[str, ...]
    neutral: tuple[str, ...]
    assassin: tuple[str, ...]
    unrevealed: tuple[str, ...]  # card-index order, exactly as the operative lists them
    all_words: frozenset[str]
    sides: Mapping[str, Side]


def size_counts(candidates: Sequence[Candidate], max_number: int) -> dict[int, int]:
    counts = {n: 0 for n in range(1, max_number + 1)}
    for c in candidates:
        counts[min(len(c.targets), max_number)] += 1
    return counts


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

    def _max_group(self, race: RaceState) -> int:
        return min(self.utility.max_number, race.ours)

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
            given = [(c.clue, c.targets, "given", c.connection) for c in candidates]
            pool = self._prepare(board, trace, given)
        trace.generated_sizes = size_counts(trace.candidates, self.utility.max_number)
        logger.info(
            "[%s] generated: %s",
            self.team,
            " ".join(f"n{n}={k}" for n, k in trace.generated_sizes.items()),
        )
        first, reserve = self._split_pool(pool)
        screened = await self._screen(board, race, trace, first)
        if not any(a.acceptable for a in screened) and reserve:
            trace.fallback_round = True
            logger.info("[%s] no acceptable action; screening reserve clues", self.team)
            screened += await self._screen(board, race, trace, reserve[:POOL_SIZE])
        trace.screened = screened
        trace.assessments = await self._verify(board, race, trace, screened)
        choice = select_clue(trace.assessments)
        if choice is None:
            trace.note = "every candidate was invalid, unassessed, or vetoed"
            logger.error("[%s] no safe clue: %s", self.team, describe_calls(trace.calls))
            raise ValueError(f"No safe clue after {len(trace.calls)} Ollama calls: {trace.note}")
        if not choice.acceptable:
            trace.note = "no acceptable action; using the least bad non-vetoed candidate"
            logger.warning("[%s] %s", self.team, trace.note)
        trace.selected = choice
        trace.explanation = explain_choice(choice, trace.assessments, race)
        self.used_clues.add(choice.clue.casefold())
        self.memory.append(PrivateClue(choice.clue, choice.number, choice.intended, choice.hits))
        sizes = best_by_size(trace.assessments)
        logger.info(
            "[%s] best by size: %s",
            self.team,
            "; ".join(
                f"n{n} {a.clue} {a.number} win={a.win_probability:.3f}"
                f"{'' if a.verified else ' (screened)'}"
                for n, a in sizes.items()
            )
            or "-",
        )
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

    @staticmethod
    def _split_pool(pool: Sequence[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
        """First screen: multi-word clues first, with a few single-word fallbacks."""
        multi = [c for c in pool if len(c.targets) > 1]
        singles = [c for c in pool if len(c.targets) == 1]
        n_singles = min(len(singles), max(POOL_SINGLES, POOL_SIZE - len(multi)))
        first = multi[: POOL_SIZE - n_singles] + singles[:n_singles]
        chosen = {c.clue for c in first}
        return first, [c for c in pool if c.clue not in chosen]

    async def _candidates(
        self,
        board: _Board,
        race: RaceState,
        trace: SpymasterTrace,
        missed: Sequence[tuple[str, tuple[str, ...]]],
    ) -> list[Candidate]:
        """One generation call (one retry only if its JSON is unusable), plus at most one
        focused regeneration when multi-word plans are missing where they matter."""
        reply: dict[str, Any] | None = None
        for attempt in range(2):
            try:
                reply = await self._generate(board, race, trace, missed, focus=False)
                break
            except OllamaResponseError as exc:
                logger.warning("[%s] generation attempt %d failed: %s", self.team, attempt + 1, exc)
        if reply is None:
            return []
        ordered = self._plans(board, trace, reply, "model")
        pool = self._prepare(board, trace, ordered)
        # At most ONE extra generation: a retry if nothing legal came back, otherwise a
        # focused multi-word search when it matters.
        reason = (
            "no legal clue in the first generation" if not pool else self._needs_focus(race, pool)
        )
        if reason:
            trace.focused_regeneration = reason
            logger.info("[%s] focused multi-word regeneration: %s", self.team, reason)
            try:
                extra = await self._generate(board, race, trace, missed, focus=True)
                ordered += self._plans(board, trace, extra, "focus")
                pool = self._prepare(board, trace, ordered)
            except OllamaResponseError as exc:
                logger.warning("[%s] focused regeneration failed: %s", self.team, exc)
        return pool

    def _needs_focus(self, race: RaceState, pool: Sequence[Candidate]) -> str:
        multi = [c for c in pool if len(c.targets) > 1]
        if race.ours >= 4 and race.pressure in ("behind", "critical") and len(multi) < 2:
            return (
                f"{race.pressure} with {race.ours} cards left but {len(multi)} multi-word plan(s)"
            )
        all_in = self._max_group(race)
        if (
            race.pressure == "critical"
            and all_in >= 3
            and not any(len(c.targets) >= all_in for c in pool)
        ):
            return f"critical endgame without a {all_in}-word attempt"
        return ""

    def _plans(
        self, board: _Board, trace: SpymasterTrace, reply: Mapping[str, Any], source: str
    ) -> list[tuple[str, Sequence[str], str, str]]:
        friendly = {word.casefold(): word for word in board.friendly}
        ordered: list[tuple[str, Sequence[str], str, str]] = []
        raw_groups = reply.get("groups")
        for plan in raw_groups if isinstance(raw_groups, list) else []:
            if not isinstance(plan, dict):
                continue
            targets = [
                friendly[t.casefold()]
                for t in plan.get("targets", [])
                if isinstance(t, str) and t.casefold() in friendly
            ]
            connection = str(plan.get("connection", ""))[:80]
            ordered.append((str(plan.get("clue", "")).strip(), targets, source, connection))
        raw_words = reply.get("words")
        if not isinstance(raw_words, dict):
            return ordered
        words: dict[str, dict[str, list[str]]] = {}
        for word in board.friendly:
            entry = raw_words.get(word)
            entry = entry if isinstance(entry, dict) else {}
            words[word] = {
                kind: [str(a).strip() for a in entry.get(kind, []) if isinstance(a, str)]
                for kind in ("themes", "specific")
            }
        trace.words = words
        # Exact theme overlaps: a shared idea for 2+ of our words (multi-word by nature).
        by_stem: dict[str, tuple[str, list[str]]] = {}
        for word, kinds in words.items():
            for idea in kinds["themes"] + kinds["specific"]:
                _, members = by_stem.setdefault(_stem(idea), (idea, []))
                if word not in members:
                    members.append(word)
        for spelled, members in sorted(by_stem.values(), key=lambda item: -len(item[1])):
            if len(members) >= 2:
                ordered.append((spelled, members, "overlap", "shared theme"))
        # Specific single-word fallbacks: first choices first, then second choices.
        for rank in range(SPECIFIC_PER_WORD):
            for word, kinds in words.items():
                if rank < len(kinds["specific"]):
                    ordered.append((kinds["specific"][rank], [word], "specific", ""))
        return ordered

    def _prepare(
        self,
        board: _Board,
        trace: SpymasterTrace,
        ordered: Sequence[tuple[str, Sequence[str], str, str]],
    ) -> list[Candidate]:
        """Legal, distinct, never-used clues in canonical spelling; multi-word first.

        A clue word proposed more than once keeps its largest coherent target group.
        """
        friendly = {word.casefold(): word for word in board.friendly}
        cap = min(self.utility.max_number, len(board.friendly))
        used = {_stem(clue) for clue in self.used_clues}
        board_words = set(board.all_words)
        by_stem: dict[str, Candidate] = {}
        trace.rejected = []
        for clue, raw_targets, source, connection in ordered:
            targets = tuple(
                dict.fromkeys(
                    friendly[t.casefold()] for t in raw_targets if t.casefold() in friendly
                )
            )[:cap]
            problem = clue_problem(clue, board_words)
            if problem is None and not targets:
                problem = "no valid friendly targets"
            if problem is None and _stem(clue) in used:
                problem = "already used by this team"
            if problem is not None:
                trace.rejected.append((clue or "<empty>", f"{source}: {problem}"))
                continue
            key = _stem(clue)
            if key not in by_stem or len(targets) > len(by_stem[key].targets):
                by_stem[key] = Candidate(canonical_clue(clue), targets, source, connection)
        candidates = sorted(
            by_stem.values(),
            key=lambda c: (len(c.targets) == 1, c.source in ("specific", "overlap")),
        )
        trace.candidates = candidates
        return candidates

    async def _generate(
        self,
        board: _Board,
        race: RaceState,
        trace: SpymasterTrace,
        missed: Sequence[tuple[str, tuple[str, ...]]],
        *,
        focus: bool,
    ) -> dict[str, Any]:
        cap = self._max_group(race)
        idea_list = {"type": "array", "items": WORD_SCHEMA}
        properties: dict[str, Any] = {}
        if not focus or not trace.candidates:
            properties["words"] = {
                "type": "object",
                "properties": {
                    word: {
                        "type": "object",
                        "properties": {
                            "themes": {
                                **idea_list,
                                "minItems": THEMES_PER_WORD,
                                "maxItems": THEMES_PER_WORD,
                            },
                            "specific": {
                                **idea_list,
                                "minItems": SPECIFIC_PER_WORD,
                                "maxItems": SPECIFIC_PER_WORD,
                            },
                        },
                        "required": ["themes", "specific"],
                        "additionalProperties": False,
                    }
                    for word in board.friendly
                },
                "required": list(board.friendly),
                "additionalProperties": False,
            }
        if cap >= 2:
            properties["groups"] = {
                "type": "array",
                "minItems": 3 if focus else 4,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "targets": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(board.friendly)},
                            "minItems": 2,
                            "maxItems": cap,
                        },
                        "clue": WORD_SCHEMA,
                        "connection": {"type": "string"},
                    },
                    "required": ["targets", "clue", "connection"],
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
            "a form of one. " + STYLE
        )
        if "words" in properties:
            system += (
                " First, for each of your words give two themes (categories, functions, "
                "properties, domains it belongs to) and two specific clue ideas that point "
                "to that word alone."
            )
        if cap >= 2:
            system += (
                " Then find GROUPS of your words that share one defensible concept: look at "
                f"pairs, triples{' and groups of four' if cap >= 4 else ''}, and give each "
                "group a clue and a short connection (at most six words, for your notes). "
                "Most of your effort should go into these groups."
            )
        system += (
            " Avoid any clue that also fits an opponent word, a neutral word, or above all "
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
        if race.pressure == "critical" and cap >= 2:
            order = ", then ".join(f"a credible {n}-word clue" for n in range(cap, 1, -1))
            lines.append(f"Search order: {order}; a one-word clue only as a last resort.")
        elif 2 <= race.ours <= self.utility.max_number:
            lines.append(
                f"One clue connecting all {race.ours} of your remaining words would win the "
                "game now; include such a group if any real link exists."
            )
        if focus and not trace.candidates:
            lines.append(
                "Your earlier suggestions were not usable single words. Every idea and clue "
                "must be ONE real dictionary word."
            )
        elif focus:
            lines.append(
                "Your earlier suggestions had too few multi-word groups. Give ONLY groups: "
                f"the most coherent pairs, triples{' and fours' if cap >= 4 else ''} you can "
                "find. Do not repeat these clues: "
                + (", ".join(c.clue for c in trace.candidates) or "-")
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
            "generate-focus" if focus else "generate",
            system=system,
            user="\n".join(lines),
            schema=schema,
            num_predict=700 if focus else 1000,
            temperature=GENERATOR_TEMPERATURE,
        )

    def _log(self, stage: str, board: _Board, a: ClueAssessment) -> None:
        labels = {w: s.value[0].upper() for w, s in board.sides.items()}
        logger.info(
            "[%s]   %s %s %d (intended %s): %s | %s | %s",
            self.team,
            stage,
            a.clue,
            a.number,
            list(a.intended),
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
        assessments: list[ClueAssessment] = []
        for candidate in candidates:
            actions = assess_actions(
                candidate.clue,
                candidate.targets,
                rankings.get(candidate.clue, ()),
                board.sides,
                race,
                self.utility,
            )
            for action in actions:
                self._log("screen", board, action)
            assessments += actions
        return assessments

    async def _verify(
        self,
        board: _Board,
        race: RaceState,
        trace: SpymasterTrace,
        screened: Sequence[ClueAssessment],
    ) -> list[ClueAssessment]:
        """Re-rank the most promising clues with the operative's exact request.

        The verified ranking replaces the screened one for EVERY action of that clue,
        with exactly the same numbers: (MOTION, 3) stays (MOTION, 3), better or worse.
        """
        final = {a.action: a for a in screened}
        verified_clues: set[str] = set()

        def finalists() -> list[ClueAssessment]:
            return sorted(
                (a for a in final.values() if not a.vetoed and a.ranking and not a.verified),
                key=lambda a: (a.acceptable, a.win_probability),
                reverse=True,
            )

        for _ in range(MAX_VERIFY):
            pending = finalists()
            if not pending:
                break
            await self._verify_clue(board, race, trace, pending[0].clue, final, verified_clues)
            best = select_clue([a for a in final.values() if a.verified and a.acceptable])
            upcoming = finalists()
            # Stop once no unverified action could plausibly beat the verified best.
            if best is not None and (
                not upcoming or upcoming[0].win_probability < best.win_probability - VERIFY_MARGIN
            ):
                break
        # Never submit an action whose ranking was only screened.
        for _ in range(FINAL_CHECKS):
            choice = select_clue(list(final.values()))
            if choice is None or choice.verified or choice.clue in verified_clues:
                break
            await self._verify_clue(board, race, trace, choice.clue, final, verified_clues)
        return list(final.values())

    async def _verify_clue(
        self,
        board: _Board,
        race: RaceState,
        trace: SpymasterTrace,
        clue: str,
        final: dict[tuple[str, int], ClueAssessment],
        verified_clues: set[str],
    ) -> None:
        verified_clues.add(clue)
        actions = [a for a in final.values() if a.clue == clue]
        request = single_clue_request(clue, board.unrevealed)
        try:
            reply = await timed_chat(self.llm, trace.calls, "verify", **request)
        except OllamaResponseError as exc:
            logger.warning("[%s] verifying %s failed: %s", self.team, clue, exc)
            return
        numbers = [a.number for a in actions]
        rescored = assess_actions(
            clue,
            actions[0].intended,
            parse_single(reply, clue, board.unrevealed),
            board.sides,
            race,
            self.utility,
            numbers=numbers,
            verified=True,
        )
        for before, after in zip(actions, rescored, strict=True):
            # Candidate identity is (clue, number): verification may reject an action
            # but must never turn it into a different one.
            if after.action != before.action:
                raise AssertionError(f"verification changed {before.action} to {after.action}")
            final[after.action] = after
            trace.verified.append(after)
            self._log(f"VERIFY {after.clue} {after.number}:", board, after)

    async def _batched_rankings(
        self, board: _Board, trace: SpymasterTrace, candidates: Sequence[Candidate]
    ) -> dict[str, tuple[RankedWord, ...]] | None:
        # Colour-blind on purpose: only unrevealed words and clue words, never the key,
        # the intended targets, or the private connection notes.
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
