"""Sequential dispatch; each operative decision follows a fresh public read."""

import asyncio
import logging
import time
from typing import Any, Protocol

from codenames_ai.browser.arena import BrowserPlayer, wait_for_phase
from codenames_ai.browser.errors import BrowserIntegrationError, GameStateTimeoutError
from codenames_ai.domain.enums import CardColor, GamePhase, Role, Team
from codenames_ai.domain.models import (
    Card,
    ClueDecision,
    GuessDecision,
    PublicClueMemory,
    PublicGameState,
    SpymasterGameState,
)
from codenames_ai.recording import MatchRecorder

logger = logging.getLogger(__name__)


class SpymasterDecisionSource(Protocol):
    async def choose_clue(self, state: SpymasterGameState) -> ClueDecision: ...


class OperativeDecisionSource(Protocol):
    async def choose_guesses(
        self, state: PublicGameState, *, guesses_made: int, max_guesses: int
    ) -> GuessDecision: ...


class GameController:
    def __init__(
        self,
        players: dict[GamePhase, BrowserPlayer],
        spymasters: dict[GamePhase, SpymasterDecisionSource],
        operatives: dict[GamePhase, OperativeDecisionSource],
        *,
        timeout: float = 15,
        recorder: MatchRecorder | None = None,
    ) -> None:
        expected = {GamePhase(f"{team}_{role}") for team in ("blue", "red") for role in Role}
        if set(players) != expected:
            raise ValueError("Controller requires exactly four assigned players")
        for phase, player in players.items():
            if phase.value != f"{player.assignment.team}_{player.assignment.role}":
                raise ValueError("Player assignment does not match controller slot")
        if set(spymasters) != {GamePhase.BLUE_SPYMASTER, GamePhase.RED_SPYMASTER} or set(
            operatives
        ) != {GamePhase.BLUE_OPERATIVE, GamePhase.RED_OPERATIVE}:
            raise ValueError("Each role needs its own decision source")
        self.players = dict(players)
        self.spymasters = spymasters
        self.operatives = operatives
        self.observer = players[GamePhase.BLUE_OPERATIVE]
        self.timeout = timeout
        self._last_action: tuple[GamePhase, str] | None = None
        self._operative_turn: GamePhase | None = None
        self._guesses_made = 0
        self._friendly_guesses_this_turn = 0
        self.clue_history: dict[Team, list[PublicClueMemory]] = {team: [] for team in Team}
        self._current_clue_index: dict[Team, int | None] = {team: None for team in Team}
        self.starting_team: Team | None = None
        # Recording only: written after actions, never read by any decision.
        self.recorder = recorder
        self.winner: Team | None = None
        self._last_guess: tuple[Team, CardColor | None] | None = None
        self._last_phase: GamePhase | None = None

    def _record(self, type: str, **fields: Any) -> None:
        if self.recorder is not None:
            self.recorder.event(type, **fields)

    async def step(self) -> bool:
        # Browser stages of this step's guess, completed below and logged once.
        guess_timing: dict[str, Any] | None = None
        phase = await self._stable_phase()
        logger.info("%s TURN", phase.value.replace("_", " ").upper())
        if phase != self._last_phase and self._last_phase in self.players:
            ended = self.players[self._last_phase].assignment
            self._record("turn_ended", team=ended.team, role=ended.role)
        self._last_phase = phase
        if phase == GamePhase.GAME_OVER:
            return False
        if phase not in self.players:
            raise BrowserIntegrationError(
                f"Cannot dispatch phase={phase}; inspect or wait for a game"
            )
        if phase != self._operative_turn:
            self._operative_turn = phase if phase in self.operatives else None
            self._guesses_made = 0
            self._friendly_guesses_this_turn = 0
        player = self.players[phase]
        await wait_for_phase(player, phase, self.timeout)
        await player.verify()
        state = await player.state()
        # Public fingerprint: no spymaster secrets enter coordination state.
        public = PublicGameState(team=state.team, cards=state.cards, clue=state.clue)
        fingerprint = (phase, public.model_dump_json())
        if fingerprint == self._last_action:
            raise BrowserIntegrationError(
                "Unchanged state after action; refusing duplicate dispatch"
            )
        if player.assignment.role == Role.SPYMASTER:
            if type(state) is not SpymasterGameState:
                raise BrowserIntegrationError("Spymaster session returned the wrong state type")
            if self.starting_team is None and not any(c.revealed for c in state.cards):
                # The first clue on an untouched board is public: that team has 9 cards.
                self.starting_team = state.team
            decision = await self.spymasters[phase].choose_clue(state)
            # Recheck after potentially slow human/model inference.
            if await player.phase() != phase:
                raise BrowserIntegrationError("Turn changed while choosing a clue")
            await player.submit_clue(decision.word, decision.number)
            self._start_clue(state.team, decision.word, decision.number)
            self._record("clue", team=state.team, word=decision.word, number=decision.number)
        else:
            if type(state) is not PublicGameState:
                raise BrowserIntegrationError("Operative session returned non-public state")
            if state.clue is None:
                raise BrowserIntegrationError("Operative turn has no readable clue")
            max_guesses = state.clue.number + 1
            self._ensure_clue(state.team, state.clue.word, state.clue.number)
            self._reconcile(state.team, state.cards)
            bonus_available = (
                state.clue.number > 0
                and self._friendly_guesses_this_turn >= state.clue.number
                and self._guesses_made < max_guesses
                and self._has_unresolved_old_clue(state.team)
            )
            agent_state = PublicGameState(
                team=state.team,
                cards=state.cards,
                clue=state.clue,
                clue_history=tuple(self.clue_history[state.team]),
                bonus_only=bonus_available,
                starting_team=self.starting_team,
            )
            logger.info("Clue: %s %d", state.clue.word, state.clue.number)
            logger.info(
                "Public clue memory [%s]: %s%s",
                state.team,
                "; ".join(
                    f"{m.clue} {m.number} -> {m.unresolved_count} unresolved"
                    for m in self.clue_history[state.team]
                ),
                " (bonus guess available)" if bonus_available else "",
            )
            if (
                state.clue.number > 0
                and self._friendly_guesses_this_turn >= state.clue.number
                and not bonus_available
            ):
                if await player.phase() == phase:
                    await player.end_guessing()
            elif self._guesses_made >= max_guesses:
                logger.info(
                    "Legal guess limit reached (%d/%d); waiting for turn transition",
                    self._guesses_made,
                    max_guesses,
                )
                await self._finish_at_guess_limit(player, phase)
            else:
                decision = await self.operatives[phase].choose_guesses(
                    agent_state, guesses_made=self._guesses_made, max_guesses=max_guesses
                )
                if await player.phase() != phase:
                    raise BrowserIntegrationError("Turn changed while choosing a guess")
                if not bonus_available:
                    self._remember_reading(state.team, decision.clue_words)
                if (
                    bonus_available
                    and decision.indices
                    and self._old_clue_index(state.team, decision.source_clue) is None
                ):
                    logger.info("No valid old-clue bonus choice; ending guessing")
                    decision = GuessDecision(end_turn=True)
                if decision.end_turn and not decision.indices:
                    await player.end_guessing()
                elif len(decision.indices) == 1 and not decision.end_turn:
                    source_index = (
                        self._old_clue_index(state.team, decision.source_clue)
                        if bonus_available
                        else None
                    )
                    matches = [
                        c for c in public.cards if c.index == decision.indices[0] and not c.revealed
                    ]
                    if len(matches) != 1:
                        raise BrowserIntegrationError("Decision must select one unrevealed card")
                    result = await player.guess_card(matches[0].word)
                    guess_timing = {
                        "action": "guess",
                        "team": state.team,
                        "word": result.card.word,
                        "stages": list(result.stages),
                    }
                    lap = time.perf_counter()
                    self._guesses_made += 1
                    self._last_guess = (state.team, result.card.color)
                    self._record(
                        "guess",
                        team=state.team,
                        word=result.card.word,
                        result=guess_result(state.team, result.card.color),
                        bonus=source_index is not None,
                    )
                    self._record_guess(
                        state.team, result.card.word, result.card.color, source_index
                    )
                    if result.card.color == CardColor(state.team.value) and source_index is None:
                        self._friendly_guesses_this_turn += 1
                    decided = await self._game_decided()
                    lap = _lap(guess_timing, "game_decided_check", lap)
                    if decided:
                        self._record_timing(guess_timing)
                        # No bonus request, no End Guessing: the game is over.
                        self._last_action = fingerprint
                        await self._await_game_over()
                        return False
                    if result.phase == phase:
                        if self._guesses_made >= max_guesses:
                            await self._finish_at_guess_limit(player, phase)
                        elif (
                            state.clue.number > 0
                            and self._friendly_guesses_this_turn >= state.clue.number
                            and await player.phase() == phase
                            and not self._has_unresolved_old_clue(state.team)
                        ):
                            logger.info(
                                "Friendly clue target reached (%d/%d); ending guessing",
                                self._friendly_guesses_this_turn,
                                state.clue.number,
                            )
                            await player.end_guessing()
                    _lap(guess_timing, "turn_follow_up", lap)
                else:
                    raise BrowserIntegrationError(
                        "Choose one guess or end turn; batches are not executed"
                    )
        self._last_action = fingerprint
        synced = time.perf_counter()
        # Synchronize the source-of-truth observer after every successful action.
        try:
            async with asyncio.timeout(self.timeout):
                while True:
                    observed_phase = await self.observer.phase()
                    if observed_phase != phase:
                        break
                    observed = await self.observer.state()
                    comparable = PublicGameState(
                        team=public.team, cards=observed.cards, clue=observed.clue
                    )
                    if comparable.model_dump_json() != public.model_dump_json():
                        break
                    await asyncio.sleep(0.1)
        except TimeoutError as exc:
            raise GameStateTimeoutError("Observer did not receive the action result") from exc
        if guess_timing is not None:
            _lap(guess_timing, "observer_sync", synced)
            self._record_timing(guess_timing)
        return True

    def _record_timing(self, timing: dict[str, Any]) -> None:
        self._record("browser_action", total_ms=sum(s["ms"] for s in timing["stages"]), **timing)

    def _start_clue(self, team: Team, word: str, number: int) -> None:
        self.clue_history[team].append(
            PublicClueMemory(
                clue=word,
                number=number,
                guesses_made=0,
                friendly_hits=0,
                unresolved_count=number,
            )
        )
        self._current_clue_index[team] = len(self.clue_history[team]) - 1

    def _ensure_clue(self, team: Team, word: str, number: int) -> None:
        index = self._current_clue_index[team]
        if index is None:
            self._start_clue(team, word, number)
            return
        memory = self.clue_history[team][index]
        if (memory.clue.casefold(), memory.number) == (word.casefold(), number):
            if memory.clue != word:
                # Keep the site's spelling so operative source_clue lookups match.
                self.clue_history[team][index] = memory.model_copy(update={"clue": word})
            return
        if memory.guesses_made == 0 and index == len(self.clue_history[team]) - 1:
            # The site shows a different clue than the one recorded at submission;
            # replace the unused record instead of leaving a phantom unresolved clue.
            logger.warning(
                "Clue read back as %s %d, not %s %d", word, number, memory.clue, memory.number
            )
            self.clue_history[team][index] = PublicClueMemory(
                clue=word, number=number, guesses_made=0, friendly_hits=0, unresolved_count=number
            )
            return
        self._start_clue(team, word, number)

    def _old_clue_index(self, team: Team, clue: str | None) -> int | None:
        if clue is None:
            return None
        current = self._current_clue_index[team]
        for index in range(len(self.clue_history[team]) - 1, -1, -1):
            memory = self.clue_history[team][index]
            if (
                index != current
                and memory.unresolved_count > 0
                and memory.clue.casefold() == clue.casefold()
            ):
                return index
        return None

    def _has_unresolved_old_clue(self, team: Team) -> bool:
        current = self._current_clue_index[team]
        return any(
            index != current and memory.unresolved_count > 0
            for index, memory in enumerate(self.clue_history[team])
        )

    def _record_guess(
        self, team: Team, word: str, color: CardColor | None, source_index: int | None
    ) -> None:
        current_index = self._current_clue_index[team]
        assert current_index is not None
        current = self.clue_history[team][current_index]
        # A bonus guess uses one of this turn's guesses but belongs to the old clue.
        self.clue_history[team][current_index] = current.model_copy(
            update={
                "guesses_made": current.guesses_made + 1,
                "words_guessed": current.words_guessed + ((word,) if source_index is None else ()),
            }
        )
        if source_index is not None:
            previous = self.clue_history[team][source_index]
            self.clue_history[team][source_index] = previous.model_copy(
                update={
                    "guesses_made": previous.guesses_made + 1,
                    "words_guessed": previous.words_guessed + (word,),
                }
            )
        if color == CardColor(team.value):
            credited_index = current_index if source_index is None else source_index
            credited = self.clue_history[team][credited_index]
            self.clue_history[team][credited_index] = credited.model_copy(
                update={
                    "friendly_hits": credited.friendly_hits + 1,
                    "unresolved_count": max(credited.unresolved_count - 1, 0),
                }
            )

    def _reconcile(self, team: Team, cards: tuple[Card, ...]) -> None:
        """Bring unresolved counts in line with the public board.

        A friendly card revealed by any mechanism (a later clue, a bonus guess, or the
        opponent) counts as found for every clue whose public reading listed it: the
        clue's own guesses plus the team's strong words when it first ranked the clue.
        Counts only ever go down, and only public information is used.
        """
        found = {c.word for c in cards if c.revealed and c.color == CardColor(team.value)}
        for index, memory in enumerate(self.clue_history[team]):
            credited = found & set(memory.words_guessed + memory.associated)
            unresolved = min(memory.unresolved_count, max(memory.number - len(credited), 0))
            if unresolved != memory.unresolved_count:
                logger.info(
                    "Clue memory: %s %d now %d unresolved (found: %s)",
                    memory.clue,
                    memory.number,
                    unresolved,
                    ", ".join(sorted(credited)),
                )
                self.clue_history[team][index] = memory.model_copy(
                    update={"unresolved_count": unresolved}
                )

    def _remember_reading(self, team: Team, words: tuple[str, ...]) -> None:
        index = self._current_clue_index[team]
        if index is not None and words and not self.clue_history[team][index].associated:
            memory = self.clue_history[team][index]
            self.clue_history[team][index] = memory.model_copy(update={"associated": words})

    async def _game_decided(self) -> bool:
        """True once a team has no cards left or the assassin is revealed.

        Read from a spymaster session because the site's game-over screen can lag the
        final reveal; only the resulting yes/no is used, never passed to an operative.
        """
        state = await self.players[GamePhase.BLUE_SPYMASTER].state()
        if any(c.revealed and c.color == CardColor.ASSASSIN for c in state.cards):
            return True
        for team in Team:
            own = [c for c in state.cards if c.color == CardColor(team.value)]
            if own and all(c.revealed for c in own):
                return True
        return False

    async def _await_game_over(self) -> None:
        logger.info("Game decided by the last reveal; waiting for the game-over screen")
        try:
            async with asyncio.timeout(self.timeout):
                while await self.observer.phase() != GamePhase.GAME_OVER:
                    await asyncio.sleep(0.1)
        except TimeoutError:
            logger.warning("Game-over screen did not appear; stopping without further actions")

    async def run(self) -> None:
        while await self.step():
            continue
        self.winner = self._decided_winner()
        self._record("game_over", winner=self.winner)

    def _decided_winner(self) -> Team | None:
        """Only a reveal ends the game: the assassin loses it for the guessing team,
        otherwise the revealed card completed its own team's set."""
        if self._last_guess is None:
            return None
        team, color = self._last_guess
        if color == CardColor.ASSASSIN:
            return team.other
        if color in (CardColor.BLUE, CardColor.RED):
            return Team(color.value)
        return None

    async def _finish_at_guess_limit(self, player: BrowserPlayer, phase: GamePhase) -> None:
        try:
            async with asyncio.timeout(min(self.timeout, 5.0)):
                while await player.phase() == phase:
                    await asyncio.sleep(0.1)
            return
        except TimeoutError:
            pass
        if await player.phase() != phase:
            return
        if not await player.can_end_guessing():
            raise GameStateTimeoutError(
                "Legal guess limit reached, but the turn did not advance and End Guessing is unavailable"
            )
        logger.info("Turn remained active at the legal guess limit; using End Guessing")
        await player.end_guessing()
        if await player.phase() == phase:
            raise GameStateTimeoutError("End Guessing did not advance the turn")

    async def _stable_phase(self) -> GamePhase:
        # The site briefly changes panel and instruction in separate renders.
        # Those frames are UNKNOWN and must never dispatch an action.
        try:
            async with asyncio.timeout(self.timeout):
                while True:
                    phase = await self.observer.phase()
                    if phase not in (GamePhase.WAITING, GamePhase.UNKNOWN):
                        return phase
                    await asyncio.sleep(0.1)
        except TimeoutError as exc:
            raise GameStateTimeoutError("No actionable phase appeared") from exc


def guess_result(team: Team, color: CardColor | None) -> str:
    if color == CardColor(team.value):
        return "friendly"
    if color == CardColor(team.other.value):
        return "opponent"
    return color.value if color in (CardColor.NEUTRAL, CardColor.ASSASSIN) else "unknown"


def _lap(timing: dict[str, Any], stage: str, since: float) -> float:
    now = time.perf_counter()
    timing["stages"].append({"stage": stage, "ms": round((now - since) * 1000)})
    return now
