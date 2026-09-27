"""Explicit human/debug decisions only. No random guesses or model strategy."""

import asyncio
from collections import deque

from codenames_ai.domain.models import (
    ClueDecision,
    GuessDecision,
    PublicGameState,
    SpymasterGameState,
)


class ManualSpymasterAgent:
    async def choose_clue(self, state: SpymasterGameState) -> ClueDecision:
        value = await asyncio.to_thread(input, f"{state.team} clue and count (e.g. ORBIT 2): ")
        word, count = value.strip().rsplit(maxsplit=1)
        return ClueDecision(word=word, number=int(count))


class ManualOperativeAgent:
    async def choose_guesses(
        self, state: PublicGameState, *, guesses_made: int = 0, max_guesses: int = 0
    ) -> GuessDecision:
        safe = PublicGameState.model_validate(state.model_dump())
        print(f"{safe.team} clue: {safe.clue}")
        print("Available: " + ", ".join(c.word for c in safe.cards if not c.revealed))
        value = (await asyncio.to_thread(input, "Guess one word, or /end: ")).strip()
        if value == "/end":
            return GuessDecision(end_turn=True)
        matches = [
            c for c in safe.cards if c.word.casefold() == value.casefold() and not c.revealed
        ]
        if len(matches) != 1:
            raise ValueError("Enter exactly one unrevealed board word")
        return GuessDecision(indices=(matches[0].index,))


class DebugSpymasterAgent:
    def __init__(self, decisions: list[ClueDecision]) -> None:
        self.decisions = deque(decisions)

    async def choose_clue(self, state: SpymasterGameState) -> ClueDecision:
        if not self.decisions:
            raise ValueError("No configured debug clue remains")
        return self.decisions.popleft()


class DebugOperativeAgent:
    def __init__(self, decisions: list[GuessDecision]) -> None:
        self.decisions = deque(decisions)

    async def choose_guesses(
        self, state: PublicGameState, *, guesses_made: int = 0, max_guesses: int = 0
    ) -> GuessDecision:
        if type(state) is not PublicGameState:
            raise ValueError("Operative accepts public state only")
        if not self.decisions:
            raise ValueError("No configured debug guess remains")
        return self.decisions.popleft()
