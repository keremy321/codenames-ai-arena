"""Operative: one colour-blind ranking per clue per turn, then a deterministic policy.

The ranking call is the same prompt and schema the spymaster uses for its blind
check, so the spymaster's simulation predicts this agent. guess_policy decides
guess or stop from the tiers and how many words are still needed; there are no
model confidence numbers. After each reveal the controller rereads the board and
asks again; the ranking for the same clue in the same turn is reused minus the
revealed words, so a turn normally costs one qwen3 call (one more if the optional
old-clue bonus guess is considered).

Only PublicGameState is accepted and it is re-sanitized: no hidden colours, no
spymaster targets, no DOM data reach the prompt.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from codenames_ai.domain.enums import CardColor, Role, Team
from codenames_ai.domain.models import GuessDecision, PublicGameState
from codenames_ai.llm.ollama import OllamaClient, OllamaResponseError

from .association import (
    RANKER_TEMPERATURE,
    RANKING_SYSTEM,
    SINGLE_NUM_PREDICT,
    Fit,
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
from .guess_policy import PolicyDecision, decide_bonus, decide_current
from .race import RacePosition

logger = logging.getLogger(__name__)


@dataclass
class OperativeTrace:
    team: Team
    mode: str
    clue: str
    guesses_made: int
    calls: list[LLMCall] = field(default_factory=list)
    ranking: tuple[RankedWord, ...] = ()
    reused_ranking: bool = False
    unresolved: dict[str, int] = field(default_factory=dict)
    decision: PolicyDecision | None = None


class OperativeAgent(Agent):
    def __init__(self, team: Team, llm: OllamaClient) -> None:
        super().__init__(team, Role.OPERATIVE, llm)
        self._cache: tuple[tuple[str, int], frozenset[str], tuple[RankedWord, ...]] | None = None
        self.last_trace: OperativeTrace | None = None

    def prompt_state(self, state: PublicGameState) -> str:
        if type(state) is not PublicGameState or state.team != self.team:
            raise ValueError("Operative requires its own team's PublicGameState")
        # Rebuild from a plain dump: model_copy(update=...) can bypass the sanitizer, so
        # validation runs again and strips any hidden colour before prompts are built.
        return PublicGameState.model_validate(state.model_dump()).model_dump_json()

    async def choose_guesses(
        self,
        state: PublicGameState,
        *,
        guesses_made: int = 0,
        max_guesses: int | None = None,
    ) -> GuessDecision:
        safe = PublicGameState.model_validate_json(self.prompt_state(state))
        if safe.clue is None:
            raise ValueError("Operative cannot act without a readable clue")
        if max_guesses is None:
            max_guesses = safe.clue.number + 1
        if not 0 <= guesses_made < max_guesses:
            raise ValueError("Operative guess count is outside the legal turn limit")
        index = {card.word: card.index for card in safe.cards if not card.revealed}
        mode = "bonus" if safe.bonus_only else "current"
        trace = OperativeTrace(self.team, mode, safe.clue.word, guesses_made)
        self.last_trace = trace
        if not index:
            decision = PolicyDecision(None, "no unrevealed words")
        elif safe.bonus_only:
            decision = await self._bonus(safe, trace)
        else:
            decision = await self._current(safe, guesses_made, trace)
        trace.decision = decision
        logger.info(
            "[%s] operative %s clue %s %d, guess %d/%d: %s (%s) | ranking %s%s | %s",
            self.team,
            mode,
            safe.clue.word,
            safe.clue.number,
            guesses_made + 1,
            max_guesses,
            decision.word or "END TURN",
            decision.reason,
            format_ranking(trace.ranking, show_clue=mode == "bonus"),
            " (reused)" if trace.reused_ranking else "",
            describe_calls(trace.calls),
        )
        clue_words = (
            tuple(r.word for r in trace.ranking if r.fit is Fit.STRONG) if mode == "current" else ()
        )
        if decision.word is None:
            return GuessDecision(end_turn=True, clue_words=clue_words)
        return GuessDecision(
            indices=(index[decision.word],),
            source_clue=decision.clue or safe.clue.word,
            clue_words=clue_words,
        )

    async def _current(
        self, safe: PublicGameState, guesses_made: int, trace: OperativeTrace
    ) -> PolicyDecision:
        assert safe.clue is not None
        clue = safe.clue.word
        unrevealed = _unrevealed(safe)
        key = (canonical_clue(clue), safe.clue.number)
        ranking: tuple[RankedWord, ...] | None = None
        if guesses_made > 0 and self._cache is not None:
            cached_key, cached_words, cached_ranking = self._cache
            if cached_key == key and set(unrevealed) <= cached_words:
                ranking = tuple(r for r in cached_ranking if r.word in unrevealed)
                trace.reused_ranking = True
        if ranking is None:
            # Byte-identical to the spymaster's verification of this clue.
            request = single_clue_request(clue, unrevealed)
            reply = await self._call(trace, "rank", request, required=guesses_made == 0)
            ranking = parse_single(reply, clue, unrevealed)
            self._cache = (key, frozenset(unrevealed), ranking)
        trace.ranking = ranking
        # While still guessing, every earlier guess this turn was ours (a miss ends it).
        remaining = max(safe.clue.number - guesses_made, 0)
        return decide_current(
            ranking,
            remaining,
            first_guess_required=guesses_made == 0,
            position=public_position(safe),
        )

    async def _bonus(self, safe: PublicGameState, trace: OperativeTrace) -> PolicyDecision:
        assert safe.clue is not None
        current = safe.clue.word.casefold()
        old = [
            entry
            for entry in safe.clue_history
            if entry.unresolved_count > 0 and entry.clue.casefold() != current
        ]
        if not old:
            return PolicyDecision(None, "no unresolved old clue")
        unresolved: dict[str, int] = {}
        for entry in old:  # the latest entry wins if a clue word was repeated
            unresolved[entry.clue] = entry.unresolved_count
        trace.unresolved = dict(unresolved)
        found = {
            card.word
            for card in safe.cards
            if card.revealed and card.color == CardColor(self.team.value)
        }
        context = [
            f"- {entry.clue}: {entry.unresolved_count} of {entry.number} still unfound"
            + (
                f" (found for it: {', '.join(w for w in entry.words_guessed if w in found)})"
                if any(w in found for w in entry.words_guessed)
                else ""
            )
            for entry in old
        ]
        unrevealed = _unrevealed(safe)
        clues = list(unresolved)
        request = {
            "system": RANKING_SYSTEM,
            "user": (
                "These earlier clues from your spymaster still have unfound words:\n"
                + "\n".join(context)
                + f"\nYour team's words already found: {', '.join(sorted(found)) or '-'}\n"
                + ranking_user(unrevealed, clues)
            ),
            "schema": ranking_schema(clues, unrevealed),
            "num_predict": SINGLE_NUM_PREDICT,
            "temperature": RANKER_TEMPERATURE,
        }
        reply = await self._call(trace, "rank-bonus", request, required=False)
        # Each clue keeps its own list: competition within a clue must stay visible even
        # when a word is also listed under another clue.
        rankings = parse_rankings(reply, clues, unrevealed)
        trace.ranking = tuple(item for clue in clues for item in rankings.get(clue, ()))
        return decide_bonus(rankings, unresolved)

    async def _call(
        self, trace: OperativeTrace, purpose: str, request: dict[str, Any], *, required: bool
    ) -> dict[str, Any]:
        # One corrective retry at most, and only when a guess is mandatory.
        for attempt in range(2 if required else 1):
            try:
                return await timed_chat(self.llm, trace.calls, purpose, **request)
            except OllamaResponseError as exc:
                logger.warning("[%s] invalid ranking (attempt %d): %s", self.team, attempt + 1, exc)
        if required:
            raise ValueError("Ollama failed to rank words for a mandatory guess")
        return {}


def public_position(state: PublicGameState) -> RacePosition | None:
    """Remaining card counts as on the public score display, or None if unknown."""
    if state.starting_team is None or len(state.cards) != 25:
        return None
    totals = {
        CardColor(state.starting_team.value): 9,
        CardColor(state.starting_team.other.value): 8,
        CardColor.NEUTRAL: 7,
        CardColor.ASSASSIN: 1,
    }
    left = dict(totals)
    for card in state.cards:
        if card.revealed and card.color in left:
            left[card.color] -= 1
    ours = CardColor(state.team.value)
    theirs = CardColor(state.team.other.value)
    if min(left.values()) < 0 or left[ours] <= 0 or left[theirs] <= 0:
        return None
    return RacePosition(
        left[ours], left[theirs], left[CardColor.NEUTRAL], left[CardColor.ASSASSIN] > 0
    )


def _unrevealed(state: PublicGameState) -> list[str]:
    """Board order is the card index, on every session and for the spymaster too."""
    return [card.word for card in sorted(state.cards, key=lambda c: c.index) if not card.revealed]
