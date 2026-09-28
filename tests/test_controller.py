import asyncio
from dataclasses import dataclass, field

import pytest

from codenames_ai.agents.debug import DebugOperativeAgent, DebugSpymasterAgent
from codenames_ai.browser.actions import GuessResult
from codenames_ai.browser.errors import BrowserIntegrationError, GameStateTimeoutError
from codenames_ai.controller import GameController
from codenames_ai.domain.enums import CardColor, GamePhase, Role, Team
from codenames_ai.domain.models import (
    Card,
    Clue,
    ClueDecision,
    GuessDecision,
    PlayerAssignment,
    PublicGameState,
    SpymasterGameState,
)


@dataclass
class SharedGame:
    phase: GamePhase = GamePhase.BLUE_SPYMASTER
    cards: tuple[Card, ...] = (
        Card(index=0, word="MOON", color=CardColor.BLUE),
        Card(index=1, word="KING", color=CardColor.ASSASSIN),
    )
    clue: Clue | None = None
    calls: list[tuple[GamePhase, str]] = field(default_factory=list)
    fail: bool = False
    stale: bool = False
    auto_transition_after: int | None = None
    auto_transition_delay: float = 0
    confirmed_guesses: int = 0
    end_available: bool = True


class FakePlayer:
    def __init__(self, phase: GamePhase, game: SharedGame) -> None:
        team, role = phase.value.split("_")
        self.assignment = PlayerAssignment(team=Team(team), role=Role(role), nickname=phase.value)
        self.game = game
        self.slot = phase

    async def phase(self) -> GamePhase:
        return self.game.phase

    async def verify(self) -> None:
        assert self.game.phase == self.slot

    async def state(self) -> PublicGameState | SpymasterGameState:
        model = PublicGameState if self.assignment.role == Role.OPERATIVE else SpymasterGameState
        return model(team=self.assignment.team, cards=self.game.cards, clue=self.game.clue)

    def action(self, name: str) -> None:
        assert self.game.phase == self.slot  # no other role may act
        self.game.calls.append((self.slot, name))
        if self.game.fail:
            raise GameStateTimeoutError("Ambiguous server result")

    async def submit_clue(self, clue: str, number: int) -> None:
        self.action("clue")
        if not self.game.stale:
            self.game.clue = Clue(word=clue, number=number)
            self.game.phase = GamePhase(f"{self.assignment.team}_operative")

    async def guess_card(self, word: str) -> GuessResult:
        self.action("guess")
        card = next(c for c in self.game.cards if c.word == word)
        revealed = Card(index=card.index, word=word, color=card.color, revealed=True)
        self.game.cards = tuple(revealed if c.index == card.index else c for c in self.game.cards)
        self.game.confirmed_guesses += 1
        if self.game.confirmed_guesses == self.game.auto_transition_after:
            if self.game.auto_transition_delay:

                async def advance() -> None:
                    await asyncio.sleep(self.game.auto_transition_delay)
                    self.game.phase = GamePhase.RED_SPYMASTER

                asyncio.create_task(advance())
            else:
                self.game.phase = GamePhase.RED_SPYMASTER
        return GuessResult(revealed, self.game.phase, self.game.phase == self.slot)

    async def can_end_guessing(self) -> bool:
        return self.game.phase == self.slot and self.game.end_available

    async def end_guessing(self) -> None:
        self.action("end")
        self.game.phase = (
            GamePhase.RED_SPYMASTER
            if self.assignment.team == Team.BLUE
            else GamePhase.BLUE_SPYMASTER
        )


def controller(game: SharedGame) -> GameController:
    phases = [GamePhase(f"{team}_{role}") for team in Team for role in Role]
    players = {p: FakePlayer(p, game) for p in phases}
    spies = {
        p: DebugSpymasterAgent([ClueDecision(word="ORBIT", number=2)])
        for p in phases
        if p.value.endswith("spymaster")
    }
    ops = {
        p: DebugOperativeAgent([GuessDecision(indices=(0,)), GuessDecision(end_turn=True)])
        for p in phases
        if p.value.endswith("operative")
    }
    return GameController(players, spies, ops, timeout=0.2)


@pytest.mark.parametrize(
    "phase",
    [
        GamePhase.BLUE_SPYMASTER,
        GamePhase.RED_SPYMASTER,
        GamePhase.BLUE_OPERATIVE,
        GamePhase.RED_OPERATIVE,
    ],
)
async def test_only_active_player_dispatches(phase: GamePhase) -> None:
    game = SharedGame(phase=phase, clue=Clue(word="ORBIT", number=2))
    await controller(game).step()
    assert len(game.calls) == 1
    assert game.calls[0][0] == phase


async def test_reconsider_after_each_reveal_and_do_not_share_secrets() -> None:
    game = SharedGame()
    control = controller(game)
    seen: list[PublicGameState] = []

    class RecordingOperative:
        async def choose_guesses(
            self, state: PublicGameState, *, guesses_made: int, max_guesses: int
        ) -> GuessDecision:
            assert type(state) is PublicGameState
            assert state.cards[1].color is None
            assert "assassin" not in state.model_dump_json()
            assert (guesses_made, max_guesses) == (len(seen), 3)
            seen.append(state)
            return GuessDecision(indices=(0,)) if len(seen) == 1 else GuessDecision(end_turn=True)

    control.operatives[GamePhase.BLUE_OPERATIVE] = RecordingOperative()
    await control.step()  # Spy knows assassin; operative never receives that state.
    await control.step()  # First operative decision sees no revealed cards.
    await control.step()  # Second operative decision sees the new public reveal.
    assert [s.cards[0].revealed for s in seen] == [False, True]
    assert game.calls == [
        (GamePhase.BLUE_SPYMASTER, "clue"),
        (GamePhase.BLUE_OPERATIVE, "guess"),
        (GamePhase.BLUE_OPERATIVE, "end"),
    ]


@pytest.mark.parametrize("clue_number,max_guesses", [(1, 2), (2, 3)])
async def test_operative_limit_ends_turn_without_extra_agent_call(
    clue_number: int, max_guesses: int
) -> None:
    game = SharedGame(
        phase=GamePhase.BLUE_OPERATIVE,
        clue=Clue(word="HERO", number=clue_number),
        cards=tuple(
            Card(index=index, word=f"WORD{index}", color=CardColor.NEUTRAL) for index in range(4)
        ),
    )
    control = controller(game)

    class AlwaysGuess:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        async def choose_guesses(
            self, state: PublicGameState, *, guesses_made: int, max_guesses: int
        ) -> GuessDecision:
            self.calls.append((guesses_made, max_guesses))
            return GuessDecision(indices=(guesses_made,))

    agent = AlwaysGuess()
    control.operatives[GamePhase.BLUE_OPERATIVE] = agent
    for _ in range(max_guesses):
        await control.step()
    assert agent.calls == [(index, max_guesses) for index in range(max_guesses)]
    assert game.calls == [
        *[(GamePhase.BLUE_OPERATIVE, "guess")] * max_guesses,
        (GamePhase.BLUE_OPERATIVE, "end"),
    ]
    assert game.phase == GamePhase.RED_SPYMASTER


async def test_maximum_guesses_auto_transition_without_end_click() -> None:
    game = SharedGame(
        phase=GamePhase.BLUE_OPERATIVE,
        clue=Clue(word="VEHICLE", number=2),
        cards=tuple(
            Card(index=index, word=word, color=CardColor.NEUTRAL)
            for index, word in enumerate(("TAXI", "DRONE", "TEXAS"))
        ),
        auto_transition_after=3,
        auto_transition_delay=0.1,
    )
    control = controller(game)
    control.operatives[GamePhase.BLUE_OPERATIVE] = DebugOperativeAgent(
        [GuessDecision(indices=(index,)) for index in range(3)]
    )
    for _ in range(3):
        await control.step()
    assert game.phase == GamePhase.RED_SPYMASTER
    assert game.calls == [(GamePhase.BLUE_OPERATIVE, "guess")] * 3
    assert control._guesses_made == 3


@pytest.mark.parametrize("clue_number", [1, 2])
async def test_friendly_target_ends_turn_without_bonus_qwen_call(clue_number: int) -> None:
    game = SharedGame(
        phase=GamePhase.BLUE_OPERATIVE,
        clue=Clue(word="DRINK", number=clue_number),
        cards=tuple(
            Card(index=index, word=f"WORD{index}", color=CardColor.BLUE)
            for index in range(clue_number + 1)
        ),
    )
    control = controller(game)

    class RecordingOperative:
        def __init__(self) -> None:
            self.calls = 0

        async def choose_guesses(
            self, state: PublicGameState, *, guesses_made: int, max_guesses: int
        ) -> GuessDecision:
            self.calls += 1
            assert guesses_made == self.calls - 1
            assert max_guesses == clue_number + 1
            return GuessDecision(indices=(guesses_made,))

    agent = RecordingOperative()
    control.operatives[GamePhase.BLUE_OPERATIVE] = agent
    for _ in range(clue_number):
        await control.step()
    assert agent.calls == clue_number
    assert game.calls == [
        *[(GamePhase.BLUE_OPERATIVE, "guess")] * clue_number,
        (GamePhase.BLUE_OPERATIVE, "end"),
    ]
    assert game.phase == GamePhase.RED_SPYMASTER


def test_clue_memory_is_public_and_separate_by_team() -> None:
    control = controller(SharedGame())
    control._start_clue(Team.BLUE, "ANIMAL", 3)
    control._start_clue(Team.RED, "MUSIC", 2)
    control._record_guess(Team.BLUE, "DOG", CardColor.BLUE, None)
    blue = control.clue_history[Team.BLUE][0]
    red = control.clue_history[Team.RED][0]
    assert (blue.guesses_made, blue.friendly_hits, blue.unresolved_count) == (1, 1, 2)
    assert blue.words_guessed == ("DOG",)
    assert (red.guesses_made, red.friendly_hits, red.unresolved_count) == (0, 0, 2)
    assert "targets" not in blue.model_dump_json()


async def test_completed_current_clue_allows_one_old_clue_bonus_and_preserves_history() -> None:
    game = SharedGame(
        phase=GamePhase.BLUE_OPERATIVE,
        clue=Clue(word="WATER", number=2),
        cards=(
            Card(index=0, word="RIVER", color=CardColor.BLUE),
            Card(index=1, word="SEA", color=CardColor.BLUE),
            Card(index=2, word="CAT", color=CardColor.BLUE),
            Card(index=3, word="DOG", color=CardColor.BLUE, revealed=True),
            Card(index=4, word="OAK", color=CardColor.BLUE),
        ),
    )
    control = controller(game)
    control._start_clue(Team.BLUE, "ANIMAL", 3)
    previous = control.clue_history[Team.BLUE][0]
    control.clue_history[Team.BLUE][0] = previous.model_copy(
        update={
            "guesses_made": 1,
            "friendly_hits": 1,
            "unresolved_count": 2,
            "words_guessed": ("DOG",),
        }
    )
    seen_bonus: list[bool] = []

    class RememberingOperative:
        async def choose_guesses(
            self, state: PublicGameState, *, guesses_made: int, max_guesses: int
        ) -> GuessDecision:
            assert "targets" not in state.model_dump_json()
            seen_bonus.append(state.bonus_only)
            if state.bonus_only:
                assert state.clue_history[0].unresolved_count == 2
                return GuessDecision(indices=(2,), source_clue="ANIMAL")
            return GuessDecision(indices=(guesses_made,), source_clue="WATER")

    control.operatives[GamePhase.BLUE_OPERATIVE] = RememberingOperative()
    for _ in range(3):
        await control.step()
    assert seen_bonus == [False, False, True]
    assert game.calls == [(GamePhase.BLUE_OPERATIVE, "guess")] * 3 + [
        (GamePhase.BLUE_OPERATIVE, "end")
    ]
    assert control.clue_history[Team.BLUE][0].unresolved_count == 1
    assert control.clue_history[Team.BLUE][0].words_guessed == ("DOG", "CAT")
    assert control.clue_history[Team.BLUE][0].guesses_made == 2
    assert control.clue_history[Team.BLUE][1].friendly_hits == 2
    game.phase = GamePhase.BLUE_SPYMASTER
    await control.step()
    assert control.clue_history[Team.BLUE][0].unresolved_count == 1
    assert len(control.clue_history[Team.BLUE]) == 3

    class LaterOperative:
        async def choose_guesses(
            self, state: PublicGameState, *, guesses_made: int, max_guesses: int
        ) -> GuessDecision:
            assert state.clue_history[0].clue == "ANIMAL"
            assert state.clue_history[0].unresolved_count == 1
            return GuessDecision(end_turn=True)

    control.operatives[GamePhase.BLUE_OPERATIVE] = LaterOperative()
    await control.step()


async def test_guess_limit_does_not_click_unavailable_end_action() -> None:
    game = SharedGame(
        phase=GamePhase.BLUE_OPERATIVE,
        clue=Clue(word="HERO", number=0),
        cards=(
            Card(index=0, word="MOON", color=CardColor.BLUE),
            Card(index=1, word="SUN", color=CardColor.BLUE),
        ),
        end_available=False,
    )
    control = controller(game)
    with pytest.raises(GameStateTimeoutError, match="End Guessing is unavailable"):
        await control.step()
    assert game.calls == [(GamePhase.BLUE_OPERATIVE, "guess")]


async def test_no_batch_execution() -> None:
    game = SharedGame(phase=GamePhase.BLUE_OPERATIVE, clue=Clue(word="ORBIT", number=2))
    control = controller(game)
    control.operatives[game.phase] = DebugOperativeAgent([GuessDecision(indices=(0, 1))])
    with pytest.raises(BrowserIntegrationError, match="batches"):
        await control.step()
    assert not game.calls


async def test_action_error_stops_run_without_retry() -> None:
    game = SharedGame(fail=True)
    with pytest.raises(GameStateTimeoutError):
        await controller(game).run()
    assert len(game.calls) == 1


async def test_observer_transition_timeout() -> None:
    game = SharedGame(stale=True)
    with pytest.raises(GameStateTimeoutError, match="Observer"):
        await controller(game).step()
    assert len(game.calls) == 1


async def test_game_over_stops_without_actions() -> None:
    game = SharedGame(phase=GamePhase.GAME_OVER)
    await controller(game).run()
    assert not game.calls


async def test_unknown_phase_times_out_without_actions() -> None:
    game = SharedGame(phase=GamePhase.UNKNOWN)
    with pytest.raises(GameStateTimeoutError):
        await controller(game).run()
    assert not game.calls


async def test_turn_changes_during_decision() -> None:
    game = SharedGame()
    control = controller(game)

    class SlowSpy:
        async def choose_clue(self, state: SpymasterGameState) -> ClueDecision:
            game.phase = GamePhase.RED_SPYMASTER
            return ClueDecision(word="ORBIT", number=2)

    control.spymasters[GamePhase.BLUE_SPYMASTER] = SlowSpy()
    with pytest.raises(BrowserIntegrationError, match="Turn changed"):
        await control.step()
    assert not game.calls


def test_clue_read_back_in_other_case_does_not_create_phantom_memory() -> None:
    control = controller(SharedGame())
    control._start_clue(Team.BLUE, "armor", 1)
    control._ensure_clue(Team.BLUE, "ARMOR", 1)
    history = control.clue_history[Team.BLUE]
    assert [(m.clue, m.unresolved_count) for m in history] == [("ARMOR", 1)]
    assert control._old_clue_index(Team.BLUE, "armor") is None


def test_unused_clue_record_is_replaced_by_what_the_site_shows() -> None:
    control = controller(SharedGame())
    control._start_clue(Team.BLUE, "ANIMAL", 3)
    control._record_guess(Team.BLUE, "DOG", CardColor.BLUE, None)
    control._start_clue(Team.BLUE, "St. Patrick", 2)
    control._ensure_clue(Team.BLUE, "PATRICK", 2)
    history = control.clue_history[Team.BLUE]
    assert [(m.clue, m.unresolved_count) for m in history] == [("ANIMAL", 2), ("PATRICK", 2)]
    # Once a clue has guesses, a different clue is a genuinely new turn.
    control._record_guess(Team.BLUE, "CAT", CardColor.BLUE, None)
    control._ensure_clue(Team.BLUE, "PET", 1)
    assert [m.clue for m in control.clue_history[Team.BLUE]] == ["ANIMAL", "PATRICK", "PET"]


async def test_real_operative_turn_with_bonus_uses_two_model_calls() -> None:
    from codenames_ai.agents.operative import OperativeAgent

    class Scripted:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def chat_json(self, **kwargs: object) -> dict:
            self.calls.append(kwargs)
            assert "assassin" not in str(kwargs).casefold()
            if len(self.calls) == 1:
                return {
                    "rankings": [{"clue": "WATER", "strong": ["RIVER", "SEA"], "possible": []}],
                    "best_guess": "RIVER",
                }
            return {"rankings": [{"clue": "ANIMAL", "strong": ["CAT"], "possible": []}]}

    game = SharedGame(
        phase=GamePhase.BLUE_OPERATIVE,
        clue=Clue(word="WATER", number=2),
        cards=(
            Card(index=0, word="RIVER", color=CardColor.BLUE),
            Card(index=1, word="SEA", color=CardColor.BLUE),
            Card(index=2, word="CAT", color=CardColor.BLUE),
            Card(index=3, word="DOG", color=CardColor.BLUE, revealed=True),
            Card(index=4, word="SKULL", color=CardColor.ASSASSIN),
            Card(index=5, word="OAK", color=CardColor.BLUE),
        ),
    )
    control = controller(game)
    control._start_clue(Team.BLUE, "ANIMAL", 3)
    control._record_guess(Team.BLUE, "DOG", CardColor.BLUE, None)
    llm = Scripted()
    control.operatives[GamePhase.BLUE_OPERATIVE] = OperativeAgent(Team.BLUE, llm)  # type: ignore[arg-type]
    for _ in range(3):
        await control.step()
    # Two current-clue hits, one legal bonus guess (N+1 = 3), then the limit ends the turn.
    assert game.calls == [(GamePhase.BLUE_OPERATIVE, "guess")] * 3 + [
        (GamePhase.BLUE_OPERATIVE, "end")
    ]
    assert game.phase == GamePhase.RED_SPYMASTER
    revealed = [c.word for c in game.cards if c.revealed]
    assert set(revealed) == {"RIVER", "SEA", "CAT", "DOG"}
    assert len(llm.calls) == 2  # one ranking for WATER, one for the optional bonus
    animal, water = control.clue_history[Team.BLUE]
    assert (animal.unresolved_count, animal.words_guessed) == (1, ("DOG", "CAT"))
    assert (water.friendly_hits, water.unresolved_count) == (2, 0)


async def test_game_over_after_last_card_skips_bonus_and_model() -> None:
    from codenames_ai.agents.operative import OperativeAgent

    class Scripted:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def chat_json(self, **kwargs: object) -> dict:
            self.calls.append(kwargs)
            return {
                "rankings": [{"clue": "WATER", "strong": ["RIVER", "SEA"], "possible": []}],
                "best_guess": "RIVER",
            }

    game = SharedGame(
        phase=GamePhase.BLUE_OPERATIVE,
        clue=Clue(word="WATER", number=2),
        cards=(
            Card(index=0, word="RIVER", color=CardColor.BLUE),
            Card(index=1, word="SEA", color=CardColor.BLUE),
            Card(index=2, word="DOG", color=CardColor.BLUE, revealed=True),
            Card(index=3, word="CAT", color=CardColor.RED),
            Card(index=4, word="SKULL", color=CardColor.ASSASSIN),
        ),
    )
    control = controller(game)
    control._start_clue(Team.BLUE, "ANIMAL", 3)  # unresolved old clue: a bonus would apply
    llm = Scripted()
    control.operatives[GamePhase.BLUE_OPERATIVE] = OperativeAgent(Team.BLUE, llm)  # type: ignore[arg-type]
    assert await control.step()  # RIVER
    # SEA is our last card. The site still shows the operative turn (game-over lag),
    # yet the controller must stop without a bonus request or End Guessing.
    assert not await control.step()
    assert game.calls == [(GamePhase.BLUE_OPERATIVE, "guess")] * 2
    assert len(llm.calls) == 1


def test_friendly_reveal_under_later_clue_resolves_old_clue() -> None:
    # Live bug: SOUND 2 found DRUM; RADIO was later found under another clue, but the
    # old memory still said "SOUND -> 1 unresolved" and drove a bogus bonus guess.
    control = controller(SharedGame())
    control._start_clue(Team.BLUE, "SOUND", 2)
    control._remember_reading(Team.BLUE, ("DRUM", "RADIO"))
    control._record_guess(Team.BLUE, "DRUM", CardColor.BLUE, None)
    control._start_clue(Team.BLUE, "MUSIC", 1)
    control._record_guess(Team.BLUE, "RADIO", CardColor.BLUE, None)
    cards = (
        Card(index=0, word="DRUM", color=CardColor.BLUE, revealed=True),
        Card(index=1, word="RADIO", color=CardColor.BLUE, revealed=True),
        Card(index=2, word="BALL"),
    )
    assert control.clue_history[Team.BLUE][0].unresolved_count == 1
    control._reconcile(Team.BLUE, cards)
    assert control.clue_history[Team.BLUE][0].unresolved_count == 0
    assert not control._has_unresolved_old_clue(Team.BLUE)


def test_opponent_revealing_our_card_also_resolves_memory() -> None:
    control = controller(SharedGame())
    control._start_clue(Team.BLUE, "ANIMAL", 2)
    control._remember_reading(Team.BLUE, ("CAT", "DOG"))
    control._remember_reading(Team.BLUE, ("IGNORED",))  # first reading is kept
    cards = (Card(index=0, word="DOG", color=CardColor.BLUE, revealed=True),)
    control._reconcile(Team.BLUE, cards)
    memory = control.clue_history[Team.BLUE][0]
    assert (memory.unresolved_count, memory.associated) == (1, ("CAT", "DOG"))
    # A revealed enemy or neutral card never counts; counts never go back up.
    control._reconcile(Team.BLUE, (Card(index=1, word="CAT", color=CardColor.RED, revealed=True),))
    assert control.clue_history[Team.BLUE][0].unresolved_count == 1


async def test_starting_team_is_recorded_and_shared_publicly() -> None:
    game = SharedGame(
        cards=(
            Card(index=0, word="MOON", color=CardColor.BLUE),
            Card(index=1, word="SUN", color=CardColor.BLUE),
            Card(index=2, word="KING", color=CardColor.ASSASSIN),
        )
    )
    control = controller(game)
    seen: list[PublicGameState] = []

    class Recording:
        async def choose_guesses(
            self, state: PublicGameState, *, guesses_made: int, max_guesses: int
        ) -> GuessDecision:
            seen.append(state)
            return GuessDecision(indices=(0,))

    control.operatives[GamePhase.BLUE_OPERATIVE] = Recording()
    await control.step()  # BLUE gives the first clue on an untouched board
    await control.step()
    assert control.starting_team == Team.BLUE
    assert seen[0].starting_team == Team.BLUE
    assert "assassin" not in seen[0].model_dump_json()


@pytest.mark.parametrize(
    "guess,result,winner",
    [(0, "friendly", Team.BLUE), (1, "assassin", Team.RED)],
)
async def test_game_events_are_recorded_after_actions(
    tmp_path, guess: int, result: str, winner: Team
) -> None:
    import json

    from codenames_ai.llm.base import LLMConfig
    from codenames_ai.recording import MatchRecorder

    recorder = MatchRecorder(
        tmp_path,
        room_url="https://codenames.game/room/test",
        players={"blue_operative": LLMConfig(provider="ollama", model="qwen3:14b")},
    )
    game = SharedGame()
    control = controller(game)
    control.recorder = recorder
    control.operatives[GamePhase.BLUE_OPERATIVE] = DebugOperativeAgent(
        [GuessDecision(indices=(guess,))]
    )
    await control.run()  # clue, then one decisive reveal
    assert control.winner == winner
    lines = [json.loads(line) for line in recorder.events_path.read_text("utf-8").splitlines()]
    assert [(e["type"], e.get("team")) for e in lines] == [
        ("clue", "blue"),
        ("turn_ended", "blue"),
        ("guess", "blue"),
        ("browser_action", "blue"),
        ("game_over", None),
    ]
    assert (lines[0]["word"], lines[0]["number"]) == ("ORBIT", 2)
    assert lines[1]["role"] == "spymaster"
    assert (lines[2]["word"], lines[2]["result"]) == (game.cards[guess].word, result)
    timing = lines[3]
    assert (timing["action"], timing["word"]) == ("guess", game.cards[guess].word)
    assert [stage["stage"] for stage in timing["stages"]] == ["game_decided_check"]
    assert timing["total_ms"] == sum(stage["ms"] for stage in timing["stages"])
    assert lines[4]["winner"] == winner


async def test_one_browser_action_event_per_guess_with_controller_stages(tmp_path) -> None:
    import json

    from codenames_ai.recording import MatchRecorder

    recorder = MatchRecorder(tmp_path, room_url="https://codenames.game/r/test", players={})
    game = SharedGame(
        cards=(
            Card(index=0, word="MOON", color=CardColor.BLUE),
            Card(index=1, word="SUN", color=CardColor.BLUE),
            Card(index=2, word="KING", color=CardColor.ASSASSIN),
        )
    )
    control = controller(game)
    control.recorder = recorder
    await control.step()  # clue: no browser_action (guesses only)
    await control.step()  # MOON: friendly, game continues
    lines = [json.loads(line) for line in recorder.events_path.read_text("utf-8").splitlines()]
    (timing,) = [e for e in lines if e["type"] == "browser_action"]
    assert [s["stage"] for s in timing["stages"]] == [
        "game_decided_check",
        "turn_follow_up",
        "observer_sync",
    ]
    assert lines.index(timing) == len(lines) - 1  # written after the observer caught up
