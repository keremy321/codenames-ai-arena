import json

import httpx
import pytest

from codenames_ai.agents.association import RANKING_SYSTEM, single_clue_request
from codenames_ai.agents.operative import OperativeAgent
from codenames_ai.agents.spymaster import (
    MAX_VERIFY,
    POOL_SIZE,
    Candidate,
    SpymasterAgent,
    clue_problem,
)
from codenames_ai.domain.enums import CardColor, Team
from codenames_ai.domain.models import (
    Card,
    Clue,
    PublicClueMemory,
    PublicGameState,
    SpymasterGameState,
)
from codenames_ai.llm.ollama import OllamaClient, OllamaResponseError


class ScriptedLLM:
    def __init__(self, *responses: dict | Exception) -> None:
        self.responses = iter(responses)
        self.calls: list[dict] = []

    async def chat_json(self, **kwargs: object) -> dict:
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


Ranking = tuple[list[int], list[int]]


class RoutedLLM:
    """Answers spymaster requests by kind: generate, screen (batched), verify (single).

    ``verify`` defaults to the screened ranking, as when batching changes nothing.
    """

    def __init__(
        self,
        generate: dict | Exception,
        screen: dict[str, Ranking] | Exception,
        verify: dict[str, Ranking] | None = None,
    ) -> None:
        self.generate, self.screen, self.verify = generate, screen, verify or {}
        self.calls: list[dict] = []

    @property
    def kinds(self) -> list[str]:
        return [_kind(call) for call in self.calls]

    async def chat_json(self, **kwargs: object) -> dict:
        self.calls.append(kwargs)
        kind = _kind(kwargs)
        if kind == "generate":
            answer = self.generate
        else:
            clues = _clues(kwargs)
            if isinstance(self.screen, Exception) and kind == "screen":
                answer = self.screen
            else:
                source = self.screen if kind == "screen" else {**self.screen, **self.verify}
                answer = {"rankings": [_entry(c, source.get(c.upper(), ([], []))) for c in clues]}
                if kind == "verify":
                    answer["best_guess"] = "WORD0"
        if isinstance(answer, Exception):
            raise answer
        return answer


def _kind(call: dict) -> str:
    props = call["schema"]["properties"]
    if "associations" in props:
        return "generate"
    return "verify" if "best_guess" in props else "screen"


def _clues(call: dict) -> list[str]:
    return call["schema"]["properties"]["rankings"]["items"]["properties"]["clue"]["enum"]


def _entry(clue: str, ranking: Ranking) -> dict:
    strong, possible = ranking
    return {
        "clue": clue,
        "strong": [f"WORD{i}" for i in strong],
        "possible": [f"WORD{i}" for i in possible],
    }


# WORD0-8 blue, WORD9-16 red, WORD17-23 neutral, WORD24 assassin. ``revealed``
# changes the race: e.g. revealing WORD9-15 leaves RED one card.
def spy_state(revealed: frozenset[int] = frozenset()) -> SpymasterGameState:
    return SpymasterGameState(
        team=Team.BLUE,
        cards=[
            Card(
                index=i,
                word=f"WORD{i}",
                revealed=i in revealed,
                color=(
                    CardColor.BLUE
                    if i < 9
                    else CardColor.RED
                    if i < 17
                    else CardColor.NEUTRAL
                    if i < 24
                    else CardColor.ASSASSIN
                ),
            )
            for i in range(25)
        ],
    )


def plans(*items: tuple[str, list[str]], associations: dict | None = None) -> dict:
    return {
        "associations": associations or {},
        "candidates": [{"targets": targets, "clue": clue} for clue, targets in items],
    }


async def test_spymaster_calls_and_colour_blind_checks() -> None:
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        props = payload["format"]["properties"]
        if "associations" in props:
            answer = plans(("cosmos", ["WORD0", "WORD1"]))
        else:
            answer = {
                "rankings": [{"clue": "COSMOS", "strong": ["WORD0", "WORD1"], "possible": []}]
            }
            if "best_guess" in props:
                answer["best_guess"] = "WORD0"
        return httpx.Response(200, json={"message": {"content": json.dumps(answer)}})

    async with OllamaClient(transport=httpx.MockTransport(respond)) as llm:
        agent = SpymasterAgent(Team.BLUE, llm)
        clue = await agent.choose_clue(spy_state(frozenset({3})))
    # Canonical spelling everywhere: submitted, screened, verified, and later ranked.
    assert (clue.word, clue.number) == ("COSMOS", 2)
    assert set(clue.model_dump()) == {"word", "number"}
    assert len(requests) == 3
    generator, screen, verify = requests
    targets = generator["format"]["properties"]["candidates"]["items"]["properties"]["targets"]
    assert targets["items"]["enum"] == [f"WORD{i}" for i in range(9) if i != 3]
    user = generator["messages"][1]["content"]
    assert "ASSASSIN (never hint at it): WORD24" in user
    assert "RACE: you have 8 words left, the opponent has 8" in user
    for check in (screen, verify):
        text = json.dumps(check["messages"])
        for leak in ("OPPONENT", "ASSASSIN", "NEUTRAL", "YOUR words", "targets", "RACE"):
            assert leak not in text
        assert "WORD3" not in check["messages"][1]["content"]
        assert check["messages"][0]["content"] == RANKING_SYSTEM
        assert check["options"]["temperature"] == 0.0
    assert agent.last_trace.selected.verified


async def test_verification_is_byte_identical_to_the_operative_request() -> None:
    spy_llm = RoutedLLM(plans(("orbit", ["WORD0"])), {"ORBIT": ([0], [])})
    decision = await SpymasterAgent(Team.BLUE, spy_llm).choose_clue(spy_state())
    verify = next(c for c in spy_llm.calls if _kind(c) == "verify")
    # The operative sees the clue as the site shows it and the board in its own order.
    public = PublicGameState(
        team=Team.BLUE,
        cards=list(reversed(spy_state().cards)),
        clue=Clue(word=decision.word.lower(), number=decision.number),
    )
    op_llm = ScriptedLLM(
        {"rankings": [{"clue": "ORBIT", "strong": ["WORD0"]}], "best_guess": "WORD0"}
    )
    await OperativeAgent(Team.BLUE, op_llm).choose_guesses(public)
    assert op_llm.calls[0] == verify
    assert verify == single_clue_request("Orbit", [f"WORD{i}" for i in range(25)])


async def test_verified_ranking_overrides_misleading_screen() -> None:
    # The batched screen says RADIO (ours); the operative's own request says STADIUM.
    llm = RoutedLLM(
        plans(("WAVE", ["WORD0"]), ("SIGNAL", ["WORD1"])),
        {"WAVE": ([0], []), "SIGNAL": ([1], [17])},
        verify={"WAVE": ([18, 0], [])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    assert (await agent.choose_clue(spy_state())).word == "SIGNAL"
    wave = next(a for a in agent.last_trace.assessments if a.clue == "WAVE")
    assert wave.verified and not wave.acceptable and "neutral" in wave.reason
    assert llm.kinds == ["generate", "screen", "verify", "verify"]


async def test_verified_two_card_clue_beats_equally_safe_one() -> None:
    llm = RoutedLLM(
        plans(("SOLO", ["WORD2"]), ("PAIR", ["WORD0", "WORD1"])),
        {"SOLO": ([2], []), "PAIR": ([0, 1], [])},
    )
    clue = await SpymasterAgent(Team.BLUE, llm).choose_clue(spy_state())
    assert (clue.word, clue.number) == ("PAIR", 2)
    assert llm.kinds == ["generate", "screen", "verify"]


async def test_verified_three_card_clue_is_possible() -> None:
    llm = RoutedLLM(plans(("TRIO", ["WORD0", "WORD1", "WORD2"])), {"TRIO": ([0, 1, 2], [])})
    assert (await SpymasterAgent(Team.BLUE, llm).choose_clue(spy_state())).number == 3


async def test_friendly_target_does_not_save_clue_that_points_elsewhere() -> None:
    llm = RoutedLLM(
        plans(("ODDLINK", ["WORD0"]), ("PLAIN", ["WORD5"])),
        {"ODDLINK": ([17], [0]), "PLAIN": ([5], [])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    assert (await agent.choose_clue(spy_state())).word == "PLAIN"
    rejected = next(a for a in agent.last_trace.assessments if a.clue == "ODDLINK")
    assert not rejected.acceptable and "neutral" in rejected.reason


async def test_assassin_match_vetoes_even_large_coverage() -> None:
    llm = RoutedLLM(
        plans(("BIG", ["WORD0", "WORD1", "WORD2"]), ("SMALL", ["WORD3"])),
        {"BIG": ([0, 1, 2], [24]), "SMALL": ([3], [])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    assert (await agent.choose_clue(spy_state())).word == "SMALL"
    assert next(a for a in agent.last_trace.assessments if a.clue == "BIG").vetoed


async def test_assassin_in_verified_ranking_vetoes_screened_winner() -> None:
    llm = RoutedLLM(
        plans(("BIG", ["WORD0", "WORD1"]), ("SMALL", ["WORD3"])),
        {"BIG": ([0, 1], []), "SMALL": ([3], [])},
        verify={"BIG": ([0, 1], [24])},
    )
    assert (await SpymasterAgent(Team.BLUE, llm).choose_clue(spy_state())).word == "SMALL"


async def test_number_never_exceeds_likely_plan() -> None:
    llm = RoutedLLM(plans(("PARTIAL", ["WORD0", "WORD1", "WORD2"])), {"PARTIAL": ([0, 1], [2, 17])})
    agent = SpymasterAgent(Team.BLUE, llm)
    assert (await agent.choose_clue(spy_state())).number == 2
    assert agent.memory[-1].expected == ("WORD0", "WORD1")
    assert agent.memory[-1].intended == ("WORD0", "WORD1", "WORD2")


RED_ON_ONE = frozenset(range(9, 16))  # RED has only WORD16 left
BLUE_ON_TWO = frozenset(range(2, 9))  # BLUE has only WORD0, WORD1 left


async def test_match_point_prefers_finishing_clue_over_perfect_single() -> None:
    # ours=2, theirs=1: a certain single card leaves RED a very likely winning turn.
    llm = RoutedLLM(
        plans(("PAIR", ["WORD0", "WORD1"]), associations={}),
        {"PAIR": ([0, 1], [17]), "SOLO": ([0], [])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    given = [Candidate("SOLO", ("WORD0",), "given"), Candidate("PAIR", ("WORD0", "WORD1"), "given")]
    clue = await agent.choose_clue(spy_state(RED_ON_ONE | BLUE_ON_TWO), candidates=given)
    assert (clue.word, clue.number) == ("PAIR", 2)
    trace = agent.last_trace
    assert trace.race.pressure == "critical"
    assert trace.selected.finish_chance > 0.9
    assert "opponent can likely finish next turn" in trace.explanation


async def test_when_ahead_a_safe_single_beats_a_risky_triple() -> None:
    # ours=5, theirs=8: three listed friendly words, but two opponent words are strong too.
    ahead = frozenset({5, 6, 7, 8})
    llm = RoutedLLM(
        plans(),
        {"RISKY": ([9, 0, 10, 1, 2], []), "SAFE": ([3], [])},
    )
    given = [
        Candidate("RISKY", ("WORD0", "WORD1", "WORD2"), "given"),
        Candidate("SAFE", ("WORD3",), "given"),
    ]
    agent = SpymasterAgent(Team.BLUE, llm)
    assert (await agent.choose_clue(spy_state(ahead), candidates=given)).word == "SAFE"
    assert agent.last_trace.race.pressure == "ahead"


async def test_generator_receives_race_guidance() -> None:
    llm = RoutedLLM(plans(("PAIR", ["WORD0", "WORD1"])), {"PAIR": ([0, 1], [])})
    await SpymasterAgent(Team.BLUE, llm).choose_clue(spy_state(RED_ON_ONE | BLUE_ON_TWO))
    user = llm.calls[0]["user"]
    assert "you have 2 words left, the opponent has 1" in user
    assert "will probably win on its next turn" in user
    assert "connecting all 2 of your remaining words would win" in user


async def test_last_card_generation_asks_only_for_associations() -> None:
    llm = RoutedLLM(
        {"associations": {"WORD0": {"specific": ["lone"], "broad": []}}},
        {"LONE": ([0], [])},
    )
    clue = await SpymasterAgent(Team.BLUE, llm).choose_clue(spy_state(frozenset(range(1, 9))))
    assert (clue.word, clue.number) == ("LONE", 1)
    assert "candidates" not in llm.calls[0]["schema"]["properties"]


async def test_reserve_single_target_round_uses_screen_not_generation() -> None:
    associations = {
        f"WORD{i}": {"specific": [f"pick{c}{v}" for v in "abc"], "broad": []}
        for i, c in enumerate("klmnopqrs")
    }
    first_round = {}
    llm = RoutedLLM(plans(associations=associations), {})

    async def chat_json(**kwargs: object) -> dict:
        llm.calls.append(kwargs)
        kind = _kind(kwargs)
        if kind == "generate":
            return plans(associations=associations)
        clues = _clues(kwargs)
        if not first_round:  # every first-round clue points at an enemy word
            first_round["clues"] = clues
            assert len(clues) == POOL_SIZE
            return {"rankings": [{"clue": c, "strong": ["WORD9"]} for c in clues]}
        return {"rankings": [_entry(c, ([4], [])) for c in clues], "best_guess": "WORD4"}

    llm.chat_json = chat_json  # type: ignore[method-assign]
    agent = SpymasterAgent(Team.BLUE, llm)  # type: ignore[arg-type]
    clue = await agent.choose_clue(spy_state())
    # Equal screened estimates: each is verified in turn, never more than MAX_VERIFY.
    assert llm.kinds[:3] == ["generate", "screen", "screen"]
    assert set(llm.kinds[3:]) == {"verify"} and 1 <= len(llm.kinds[3:]) <= MAX_VERIFY
    assert [c["temperature"] for c in llm.calls][:3] == [0.4, 0.0, 0.0]
    assert agent.last_trace.fallback_round
    second = _clues(llm.calls[2])
    assert not set(first_round["clues"]) & set(second)
    assert clue.number == 1 and clue.word == second[0]


async def test_least_bad_clue_when_nothing_is_acceptable() -> None:
    llm = RoutedLLM(
        plans(("MEH", ["WORD0"]), ("WORSE", ["WORD1"])),
        {"MEH": ([17], [0]), "WORSE": ([9], [1])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    clue = await agent.choose_clue(spy_state())
    assert clue.word == "MEH"
    assert "least bad" in agent.last_trace.note


async def test_all_vetoed_raises_instead_of_risking_assassin() -> None:
    llm = RoutedLLM(plans(("DOOM", ["WORD0"])), {"DOOM": ([0], [24])})
    with pytest.raises(ValueError, match="No safe clue after 2 Ollama calls"):
        await SpymasterAgent(Team.BLUE, llm).choose_clue(spy_state())


async def test_screen_failure_never_uses_unchecked_clue() -> None:
    llm = RoutedLLM(plans(("ORBIT", ["WORD0"])), OllamaResponseError("bad JSON"))
    with pytest.raises(ValueError, match="No safe clue"):
        await SpymasterAgent(Team.BLUE, llm).choose_clue(spy_state())
    assert llm.kinds == ["generate", "screen"]


async def test_malformed_generation_retries_once() -> None:
    calls = {"n": 0}
    llm = RoutedLLM(plans(("ORBIT", ["WORD0"])), {"ORBIT": ([0], [])})
    original = llm.chat_json

    async def flaky(**kwargs: object) -> dict:
        if _kind(kwargs) == "generate" and not calls["n"]:
            calls["n"] += 1
            llm.calls.append(kwargs)
            raise OllamaResponseError("truncated")
        return await original(**kwargs)

    llm.chat_json = flaky  # type: ignore[method-assign]
    assert (await SpymasterAgent(Team.BLUE, llm).choose_clue(spy_state())).word == "ORBIT"
    assert llm.kinds == ["generate", "generate", "screen", "verify"]


async def test_board_derivative_is_dropped_locally_without_retry() -> None:
    llm = RoutedLLM(
        plans(("WORD0s", ["WORD0"]), ("two words", ["WORD1"]), ("ORBIT", ["WORD2"])),
        {"ORBIT": ([2], [])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    assert (await agent.choose_clue(spy_state())).word == "ORBIT"
    assert llm.kinds == ["generate", "screen", "verify"]
    assert {clue for clue, _ in agent.last_trace.rejected} == {"WORD0s", "two words"}


async def test_private_missed_targets_feed_the_next_generation() -> None:
    llm = RoutedLLM(
        plans(("ORBIT", ["WORD0", "WORD1"])),
        {"ORBIT": ([0, 1], []), "COSMOS": ([1], [])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    await agent.choose_clue(spy_state())
    llm.generate = plans(("COSMOS", ["WORD1"]))
    await agent.choose_clue(spy_state(frozenset({0})))
    second_generation = [c for c in llm.calls if _kind(c) == "generate"][1]
    prompt = second_generation["user"]
    assert "ORBIT -> WORD1" in prompt and "ORBIT -> WORD0" not in prompt
    assert "Clues already used: orbit" in prompt
    # Colour-blind checks never see private memory.
    assert all("ORBIT" not in c["user"] for c in llm.calls[-2:])


async def test_spymaster_never_repeats_its_own_clue() -> None:
    llm = RoutedLLM(plans(("ORBIT", ["WORD0"]), ("COSMOS", ["WORD1"])), {"COSMOS": ([1], [])})
    agent = SpymasterAgent(Team.BLUE, llm)
    agent.used_clues.add("orbit")
    assert (await agent.choose_clue(spy_state())).word == "COSMOS"
    assert _clues(llm.calls[1]) == ["COSMOS"]


@pytest.mark.parametrize(
    "clue,problem",
    [
        ("SNOWMAN", True),  # contains board word SNOW
        ("SNO", False),
        ("snowing", True),
        ("ICE AGE", True),
        ("ice-cold", False),
        ("B4", True),
        ("frost", False),
    ],
)
def test_clue_rules(clue: str, problem: bool) -> None:
    assert (clue_problem(clue, {"snow", "moon"}) is not None) is problem


async def test_spymaster_zero_friendly_does_not_call_model() -> None:
    llm = ScriptedLLM()
    with pytest.raises(ValueError, match="reread game state"):
        await SpymasterAgent(Team.BLUE, llm).choose_clue(spy_state(frozenset(range(9))))
    assert not llm.calls


async def test_spymaster_refuses_incomplete_board() -> None:
    llm = ScriptedLLM()
    with pytest.raises(ValueError, match="incomplete"):
        await SpymasterAgent(Team.BLUE, llm).choose_clue(
            SpymasterGameState(
                team=Team.BLUE, cards=[Card(index=0, word="MOON", color=CardColor.UNKNOWN)]
            )
        )


def public_state(
    *, bonus: bool = False, clue: str = "SPACE", number: int = 2, revealed: tuple = ()
) -> PublicGameState:
    words = ["MOON", "STAR", "DOG", "HORSE", "PIANO"]
    return PublicGameState(
        team=Team.BLUE,
        cards=[
            Card(
                index=i,
                word=w,
                revealed=w in revealed,
                # A hidden colour on an unrevealed card must be stripped before any prompt.
                color=CardColor.ASSASSIN if w == "MOON" else CardColor.BLUE,
            )
            for i, w in enumerate(words)
        ],
        clue=Clue(word=clue, number=number),
        clue_history=(
            PublicClueMemory(
                clue="ANIMAL",
                number=3,
                guesses_made=1,
                friendly_hits=1,
                unresolved_count=2,
                words_guessed=("CAT",),
            ),
            PublicClueMemory(
                clue=clue,
                number=number,
                guesses_made=number if bonus else 0,
                friendly_hits=number if bonus else 0,
                unresolved_count=0 if bonus else number,
            ),
        ),
        bonus_only=bonus,
    )


def op_rank(clue: str, strong: list[str], possible: list[str] = (), best: str | None = None):
    reply = {"rankings": [{"clue": clue, "strong": strong, "possible": list(possible)}]}
    if best is not None:
        reply["best_guess"] = best
    return reply


async def test_operative_prompt_is_public_only_and_matches_blind_check_format() -> None:
    llm = ScriptedLLM(op_rank("SPACE", ["STAR", "MOON"], best="STAR"))
    choice = await OperativeAgent(Team.BLUE, llm).choose_guesses(public_state())
    assert choice.indices == (1,) and choice.source_clue == "SPACE"
    call = llm.calls[0]
    text = json.dumps(call)
    assert "assassin" not in text.casefold()
    assert "ANIMAL" not in call["user"]  # old clues only matter for the bonus guess
    assert "CAT" not in text  # nor private or unrelated history
    assert call["system"] == RANKING_SYSTEM
    words = call["schema"]["properties"]["rankings"]["items"]["properties"]["strong"]["items"]
    assert words["enum"] == ["MOON", "STAR", "DOG", "HORSE", "PIANO"]


async def test_operative_rejects_spymaster_state() -> None:
    with pytest.raises(ValueError, match="PublicGameState"):
        await OperativeAgent(Team.BLUE, ScriptedLLM()).choose_guesses(spy_state())  # type: ignore[arg-type]


async def test_operative_reuses_ranking_within_turn_and_reranks_next_turn() -> None:
    llm = ScriptedLLM(
        op_rank("SPACE", ["STAR", "MOON"], ["DOG"], best="STAR"),
        op_rank("SPACE", ["MOON"], best="MOON"),
    )
    agent = OperativeAgent(Team.BLUE, llm)
    assert (await agent.choose_guesses(public_state())).indices == (1,)
    second = await agent.choose_guesses(public_state(revealed=("STAR",)), guesses_made=1)
    assert second.indices == (0,) and agent.last_trace.reused_ranking
    assert len(llm.calls) == 1
    await agent.choose_guesses(public_state(revealed=("STAR",)))  # new turn: fresh ranking
    assert len(llm.calls) == 2


async def test_operative_stops_when_strong_matches_are_ambiguous() -> None:
    llm = ScriptedLLM(op_rank("SPACE", ["STAR", "MOON", "DOG"], best="STAR"))
    agent = OperativeAgent(Team.BLUE, llm)
    await agent.choose_guesses(public_state())
    decision = await agent.choose_guesses(public_state(revealed=("STAR",)), guesses_made=1)
    assert decision.end_turn and "ambiguous" in agent.last_trace.decision.reason


async def test_operative_stops_without_strong_evidence() -> None:
    llm = ScriptedLLM(op_rank("SPACE", ["STAR"], ["MOON"], best="STAR"))
    agent = OperativeAgent(Team.BLUE, llm)
    await agent.choose_guesses(public_state())
    assert (await agent.choose_guesses(public_state(revealed=("STAR",)), guesses_made=1)).end_turn


async def test_operative_forced_first_guess_uses_best_guess() -> None:
    llm = ScriptedLLM(op_rank("SPACE", [], best="PIANO"))
    assert (await OperativeAgent(Team.BLUE, llm).choose_guesses(public_state())).indices == (4,)


async def test_operative_mandatory_guess_retries_once_then_raises() -> None:
    llm = ScriptedLLM(OllamaResponseError("bad"), OllamaResponseError("bad"))
    with pytest.raises(ValueError, match="mandatory guess"):
        await OperativeAgent(Team.BLUE, llm).choose_guesses(public_state())
    assert len(llm.calls) == 2


async def test_operative_bonus_uses_unambiguous_old_clue_only() -> None:
    llm = ScriptedLLM(
        {"rankings": [{"clue": "ANIMAL", "strong": ["HORSE", "DOG"], "possible": ["PIANO"]}]}
    )
    agent = OperativeAgent(Team.BLUE, llm)
    choice = await agent.choose_guesses(public_state(bonus=True), guesses_made=2, max_guesses=3)
    assert choice.indices == (3,) and choice.source_clue == "ANIMAL"
    call = llm.calls[0]
    clue_enum = call["schema"]["properties"]["rankings"]["items"]["properties"]["clue"]["enum"]
    assert clue_enum == ["ANIMAL"]  # never the finished current clue
    assert "ANIMAL: 2 of 3 still unfound" in call["user"]
    assert "intended" not in call["user"] and "assassin" not in json.dumps(call).casefold()


async def test_operative_bonus_ambiguous_keeps_bonus_unused() -> None:
    llm = ScriptedLLM(
        {"rankings": [{"clue": "ANIMAL", "strong": ["HORSE", "DOG", "MOON"], "possible": []}]}
    )
    choice = await OperativeAgent(Team.BLUE, llm).choose_guesses(
        public_state(bonus=True), guesses_made=2, max_guesses=3
    )
    assert choice.end_turn


async def test_operative_bonus_failure_ends_turn_without_retry() -> None:
    llm = ScriptedLLM(OllamaResponseError("bad"))
    choice = await OperativeAgent(Team.BLUE, llm).choose_guesses(
        public_state(bonus=True), guesses_made=2, max_guesses=3
    )
    assert choice.end_turn and len(llm.calls) == 1


async def test_operative_guess_count_must_be_legal() -> None:
    with pytest.raises(ValueError, match="legal turn limit"):
        await OperativeAgent(Team.BLUE, ScriptedLLM()).choose_guesses(
            public_state(), guesses_made=3, max_guesses=3
        )


def race_public_state(guesses_made: int) -> PublicGameState:
    """BLUE moved first (9 cards); 8 BLUE and 7 RED revealed: one card left each."""
    cards = []
    for i in range(25):
        color = (
            CardColor.BLUE
            if i < 9
            else CardColor.RED
            if i < 17
            else CardColor.NEUTRAL
            if i < 24
            else CardColor.ASSASSIN
        )
        revealed = i < 7 or 9 <= i < 16 or (guesses_made and i == 7)
        cards.append(Card(index=i, word=f"W{i}", color=color, revealed=revealed))
    return PublicGameState(
        team=Team.BLUE, cards=cards, clue=Clue(word="PAIR", number=2), starting_team=Team.BLUE
    )


def test_public_position_uses_only_public_counts() -> None:
    from codenames_ai.agents.operative import public_position

    state = race_public_state(0)
    position = public_position(state)
    assert (position.ours, position.theirs, position.neutral, position.assassin_hidden) == (
        2,
        1,
        7,
        True,
    )
    assert public_position(state.model_copy(update={"starting_team": None})) is None
    # Hidden colours were stripped: the counts come from revealed cards and the start.
    assert all(c.color is None for c in state.cards if not c.revealed)


async def test_operative_presses_on_at_match_point_with_public_counts() -> None:
    reply = {"rankings": [{"clue": "PAIR", "strong": ["W7", "W8", "W17"]}], "best_guess": "W7"}
    agent = OperativeAgent(Team.BLUE, ScriptedLLM(reply))
    await agent.choose_guesses(race_public_state(0))
    # One of our cards left, opponent on one card: stopping at 2 strong for 1 hands
    # them the game, so the operative takes the coin flip instead.
    decision = await agent.choose_guesses(race_public_state(1), guesses_made=1)
    assert decision.indices == (8,)
    assert "race favours guessing" in agent.last_trace.decision.reason


async def test_underrated_multi_word_clue_is_verified_within_margin() -> None:
    # The batched screen wrongly lists an enemy word for PAIR, putting it just below
    # SOLO. Within the verification margin it is checked with the operative's own
    # request, which shows the clean pair, and the pair wins.
    llm = RoutedLLM(
        plans(("PAIR", ["WORD0", "WORD1"]), ("SOLO", ["WORD2"])),
        {"PAIR": ([0, 9, 1], []), "SOLO": ([2], [])},
        verify={"PAIR": ([0, 1], [])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    clue = await agent.choose_clue(spy_state())
    assert (clue.word, clue.number) == ("PAIR", 2)
    assert llm.kinds == ["generate", "screen", "verify", "verify"]
    assert [v.clue for v in agent.last_trace.verified] == ["SOLO", "PAIR"]


async def test_selected_clue_is_always_verified() -> None:
    # Three finalists disappoint on verification; the fourth, only screened so far, is
    # now the best estimate, so it is verified before being submitted.
    names = ["AONE", "BTWO", "CTHREE", "DFOUR"]
    llm = RoutedLLM(
        plans(*[(n, [f"WORD{i}", f"WORD{i + 4}"]) for i, n in enumerate(names)]),
        {"AONE": ([0, 4], []), "BTWO": ([1, 5], []), "CTHREE": ([2, 6], []), "DFOUR": ([3], [])},
        verify={name: ([17], []) for name in names[:3]},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    clue = await agent.choose_clue(spy_state())
    assert clue.word == "DFOUR" and agent.last_trace.selected.verified
    assert llm.kinds == ["generate", "screen"] + ["verify"] * 4
