import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from codenames_ai.agents.association import RANKING_SYSTEM, single_clue_request
from codenames_ai.agents.base import Agent
from codenames_ai.agents.operative import OperativeAgent
from codenames_ai.agents.spymaster import (
    MAX_VERIFY,
    POOL_SINGLES,
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
from codenames_ai.llm.anthropic_client import AnthropicClient
from codenames_ai.llm.base import LLMClient, LLMConfig, LLMResponseError
from codenames_ai.llm.ollama import OllamaClient, OllamaResponseError
from codenames_ai.llm.openai_client import OpenAIClient
from codenames_ai.recording import MatchRecorder, RecordedSpymaster


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
        generate: dict | Exception | list[dict],
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
            answer = self.generate.pop(0) if isinstance(self.generate, list) else self.generate
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
    if "words" in props or "groups" in props:
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


def plans(*items: tuple[str, list[str]], words: dict | None = None) -> dict:
    return {
        "words": words or {},
        "groups": [
            {"targets": targets, "clue": clue, "connection": "test link"} for clue, targets in items
        ],
    }


async def test_spymaster_calls_and_colour_blind_checks() -> None:
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        props = payload["format"]["properties"]
        if "groups" in props:
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
    targets = generator["format"]["properties"]["groups"]["items"]["properties"]["targets"]
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
        {"BIG": ([0, 1, 2, 24], []), "SMALL": ([3], [])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    assert (await agent.choose_clue(spy_state())).word == "SMALL"
    big = [a for a in agent.last_trace.assessments if a.clue == "BIG"]
    assert big and all(a.vetoed for a in big)


async def test_assassin_in_verified_ranking_vetoes_screened_winner() -> None:
    llm = RoutedLLM(
        plans(("BIG", ["WORD0", "WORD1"]), ("SMALL", ["WORD3"])),
        {"BIG": ([0, 1], []), "SMALL": ([3], [])},
        verify={"BIG": ([0, 1, 24], [])},
    )
    assert (await SpymasterAgent(Team.BLUE, llm).choose_clue(spy_state())).word == "SMALL"


async def test_each_number_is_its_own_action_and_verification_keeps_it() -> None:
    # Live bug: MOTION 3 screened well, verification re-derived it as MOTION 1, and a
    # weaker single won. Now every (clue, number) is scored and verified as itself.
    llm = RoutedLLM(
        plans(("MOTION", ["WORD0", "WORD1", "WORD2"]), ("SALAD", ["WORD3"])),
        {"MOTION": ([0, 1, 2], []), "SALAD": ([3], [])},
        verify={"MOTION": ([0], [9])},  # the operative's own reading is much weaker
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    await agent.choose_clue(spy_state())
    trace = agent.last_trace
    screened = {a.action for a in trace.screened if a.clue == "MOTION"}
    verified = {a.action for a in trace.verified if a.clue == "MOTION"}
    assert screened == verified == {("MOTION", 1), ("MOTION", 2), ("MOTION", 3)}
    motion3 = next(a for a in trace.verified if a.action == ("MOTION", 3))
    assert motion3.verified and motion3.number == 3
    # The choice is simply the best verified action, whatever its size.
    best = max(a.win_probability for a in trace.assessments if a.verified and a.acceptable)
    assert trace.selected.win_probability == pytest.approx(best)


async def test_ride_two_is_never_mutated_to_ride_one() -> None:
    llm = RoutedLLM(
        plans(("RIDE", ["WORD0", "WORD1"])),
        {"RIDE": ([0, 1], [])},
        verify={"RIDE": ([0, 1, 17], [])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    await agent.choose_clue(spy_state())
    for a in agent.last_trace.verified:
        assert a.action in {x.action for x in agent.last_trace.screened}


RED_ON_ONE = frozenset(range(9, 16))  # RED has only WORD16 left
BLUE_ON_TWO = frozenset(range(2, 9))  # BLUE has only WORD0, WORD1 left


async def test_match_point_prefers_finishing_clue_over_perfect_single() -> None:
    # ours=2, theirs=1: a certain single card leaves RED a very likely winning turn.
    llm = RoutedLLM(
        plans(("PAIR", ["WORD0", "WORD1"])),
        {"PAIR": ([0, 1], [17]), "SOLO": ([0], [])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    given = [Candidate("SOLO", ("WORD0",), "given"), Candidate("PAIR", ("WORD0", "WORD1"), "given")]
    clue = await agent.choose_clue(spy_state(RED_ON_ONE | BLUE_ON_TWO), candidates=given)
    assert (clue.word, clue.number) == ("PAIR", 2)
    trace = agent.last_trace
    assert trace.race.pressure == "critical"
    assert trace.selected.finish_chance > 0.9
    assert "85% estimated finish-next chance" in trace.explanation
    assert "best 1-card SOLO" in trace.explanation


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
    assert "very likely to win on its next turn" in user
    assert "Search order: a credible 2-word clue" in user


async def test_last_card_generation_asks_only_for_associations() -> None:
    llm = RoutedLLM(
        {"words": {"WORD0": {"themes": [], "specific": ["lone"]}}},
        {"LONE": ([0], [])},
    )
    clue = await SpymasterAgent(Team.BLUE, llm).choose_clue(spy_state(frozenset(range(1, 9))))
    assert (clue.word, clue.number) == ("LONE", 1)
    assert "groups" not in llm.calls[0]["schema"]["properties"]


async def test_reserve_single_target_round_uses_screen_not_generation() -> None:
    associations = {
        f"WORD{i}": {"themes": [], "specific": [f"pick{c}{v}" for v in "ab"]}
        for i, c in enumerate("klmnopqrs")
    }
    first_round = {}
    llm = RoutedLLM(plans(words=associations), {})

    async def chat_json(**kwargs: object) -> dict:
        llm.calls.append(kwargs)
        kind = _kind(kwargs)
        if kind == "generate":
            return plans(words=associations)
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
    # Every reserve clue is ranked to WORD4; only the one generated for WORD4 is consistent.
    assert clue.number == 1 and clue.word in second
    assert agent.last_trace.selected.intended == ("WORD4",)


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
    llm = RoutedLLM(plans(("DOOM", ["WORD0"])), {"DOOM": ([0, 24], [])})
    with pytest.raises(ValueError, match="No safe clue after 2 LLM calls"):
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
        ("international", False),  # 13 letters: still allowed
        ("musicalinstrument", True),  # a glued compound
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
    assert [v.action for v in agent.last_trace.verified] == [
        ("SOLO", 1),
        ("PAIR", 1),
        ("PAIR", 2),
    ]


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


BEHIND = frozenset(range(9, 14))  # RED has 3 left, BLUE 9: behind
CRITICAL_FOUR = frozenset(range(4, 9)) | frozenset(range(9, 16))  # ours=4, theirs=1


def singles_only() -> dict:
    words = {
        f"WORD{i}": {"themes": [], "specific": [f"solo{c}"]} for i, c in enumerate("klmnopqrs")
    }
    return {"words": words, "groups": []}


async def test_generator_searches_groups_up_to_four_and_keeps_notes_private() -> None:
    llm = RoutedLLM(
        plans(("QUAD", ["WORD0", "WORD1", "WORD2", "WORD3"])),
        {"QUAD": ([0, 1, 2, 3], [])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    clue = await agent.choose_clue(spy_state())
    group = llm.calls[0]["schema"]["properties"]["groups"]["items"]["properties"]["targets"]
    assert (group["minItems"], group["maxItems"]) == (2, 4)
    assert "groups of four" in llm.calls[0]["system"]
    assert clue.number <= 4 and agent.last_trace.generated_sizes[4] == 1
    for call in llm.calls[1:]:
        assert "test link" not in json.dumps(call)  # private connection notes


async def test_first_screen_is_mostly_multi_word() -> None:
    words = {
        f"WORD{i}": {"themes": [], "specific": [f"solo{c}", f"alt{c}"]}
        for i, c in enumerate("klmnopqrs")
    }
    groups = [(f"pair{c}", [f"WORD{i}", f"WORD{i + 1}"]) for i, c in enumerate("abcdefg")]
    llm = RoutedLLM({**plans(*groups), "words": words}, {})
    agent = SpymasterAgent(Team.BLUE, llm)
    with pytest.raises(ValueError):  # the fake ranks nothing; only the pool matters here
        await agent.choose_clue(spy_state())
    first_screen = _clues(llm.calls[1])
    singles = [c for c in first_screen if c.startswith(("SOLO", "ALT"))]
    assert len(first_screen) == POOL_SIZE and len(singles) == POOL_SINGLES
    assert agent.last_trace.generated_sizes[1] == 18 and agent.last_trace.generated_sizes[2] == 7


async def test_focused_regeneration_runs_once_when_behind_without_groups() -> None:
    llm = RoutedLLM(
        [singles_only(), plans(("PAIR", ["WORD0", "WORD1"]))],
        {"PAIR": ([0, 1], []), "SOLOK": ([0], [])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    clue = await agent.choose_clue(spy_state(BEHIND))
    assert agent.last_trace.race.pressure == "behind"
    assert llm.kinds.count("generate") == 2
    assert "Give ONLY groups" in llm.calls[1]["user"]
    assert "multi-word plan" in agent.last_trace.focused_regeneration
    assert (clue.word, clue.number) == ("PAIR", 2)


async def test_focused_regeneration_never_loops() -> None:
    llm = RoutedLLM([singles_only(), {"groups": []}], {"SOLOK": ([0], [])})
    agent = SpymasterAgent(Team.BLUE, llm)
    await agent.choose_clue(spy_state(BEHIND))
    assert llm.kinds.count("generate") == 2


async def test_no_focused_regeneration_in_an_even_race() -> None:
    llm = RoutedLLM(singles_only(), {"SOLOK": ([0], [])})
    agent = SpymasterAgent(Team.BLUE, llm)
    await agent.choose_clue(spy_state())
    assert llm.kinds.count("generate") == 1 and not agent.last_trace.focused_regeneration


async def test_critical_endgame_searches_all_in_first() -> None:
    llm = RoutedLLM(
        [
            plans(("DUO", ["WORD0", "WORD1"]), ("DUET", ["WORD2", "WORD3"])),
            plans(("QUAD", ["WORD0", "WORD1", "WORD2", "WORD3"])),
        ],
        {"DUO": ([0, 1], []), "DUET": ([2, 3], []), "QUAD": ([0, 1, 2, 3], [17])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    clue = await agent.choose_clue(spy_state(CRITICAL_FOUR))
    first = llm.calls[0]["user"]
    assert "Search order: a credible 4-word clue, then a credible 3-word clue" in first
    assert "without a 4-word attempt" in agent.last_trace.focused_regeneration
    assert (clue.word, clue.number) == ("QUAD", 4)
    assert agent.last_trace.selected.finish_chance > 0.5


async def test_generator_grammar_forces_single_words() -> None:
    llm = RoutedLLM(plans(("ORBIT", ["WORD0", "WORD1"])), {"ORBIT": ([0, 1], [])})
    await SpymasterAgent(Team.BLUE, llm).choose_clue(spy_state())
    props = llm.calls[0]["schema"]["properties"]
    pattern = props["groups"]["items"]["properties"]["clue"]["pattern"]
    assert (
        pattern
        == props["words"]["properties"]["WORD0"]["properties"]["specific"]["items"]["pattern"]
    )
    import re

    assert re.fullmatch(pattern, "Magma") and re.fullmatch(pattern, "ice-cold")
    for bad in ("Erupting mountain", "EruptingMountain", "B4", ""):
        assert not re.fullmatch(pattern, bad)


async def test_unusable_first_generation_gets_one_retry() -> None:
    phrases = {"words": {"WORD0": {"themes": [], "specific": ["two words"]}}, "groups": []}
    llm = RoutedLLM([phrases, plans(("LONE", ["WORD0"]))], {"LONE": ([0], [])})
    agent = SpymasterAgent(Team.BLUE, llm)
    clue = await agent.choose_clue(spy_state(frozenset(range(1, 9))))
    assert clue.word == "LONE"
    assert llm.kinds.count("generate") == 2
    assert "not usable single words" in llm.calls[1]["user"]
    assert "words" in llm.calls[1]["schema"]["properties"]


async def test_one_request_verifies_every_number_of_a_word() -> None:
    llm = RoutedLLM(plans(("PLANT", ["WORD0", "WORD1", "WORD2"])), {"PLANT": ([0, 1, 2], [])})
    agent = SpymasterAgent(Team.BLUE, llm)
    await agent.choose_clue(spy_state())
    assert llm.kinds == ["generate", "screen", "verify"]
    assert {a.action for a in agent.last_trace.verified} == {
        ("PLANT", 1),
        ("PLANT", 2),
        ("PLANT", 3),
    }
    assert agent.last_trace.verified_words == ["PLANT"]


async def test_normal_turn_verifies_at_most_two_words_and_logs_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    names = ["PAIRA", "PAIRB", "PAIRC", "PAIRD", "PAIRE"]
    llm = RoutedLLM(
        plans(*[(n, [f"WORD{i}", f"WORD{i + 1}"]) for i, n in enumerate(names)]),
        {n: ([i, i + 1], []) for i, n in enumerate(names)},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    await agent.choose_clue(spy_state())
    trace = agent.last_trace
    assert llm.kinds.count("verify") <= MAX_VERIFY
    assert len(trace.finalists) <= MAX_VERIFY and trace.finalists[0][1] == [1, 2]
    assert trace.expanded_sizes[1] == 5 and trace.expanded_sizes[2] == 5
    assert trace.seconds > 0
    for line in ("raw plans: n1=0 n2=5", "expanded actions: n1=5 n2=5", "finalists: PAIRA[1,2]",
                 "verified clue words:", "llm calls:", "total:"):  # fmt: skip
        assert line in caplog.text


async def test_generator_asks_for_compact_json() -> None:
    llm = RoutedLLM(plans(("ORBIT", ["WORD0", "WORD1"])), {"ORBIT": ([0, 1], [])})
    await SpymasterAgent(Team.BLUE, llm).choose_clue(spy_state())
    assert "compact single-line JSON" in llm.calls[0]["system"]
    # The ranking prompts stay byte-identical to the operative's: no extra instruction.
    assert all("compact" not in call["system"] for call in llm.calls[1:])


def test_agents_depend_on_the_shared_llm_interface() -> None:
    import codenames_ai.agents.operative as operative_module
    import codenames_ai.agents.spymaster as spymaster_module

    assert Agent.__dataclass_fields__["llm"].type is LLMClient
    for module in (operative_module, spymaster_module):
        assert not hasattr(module, "OllamaClient")


async def test_operative_retries_any_provider_response_error() -> None:
    llm = ScriptedLLM(LLMResponseError("bad"), op_rank("SPACE", ["STAR"], best="STAR"))
    choice = await OperativeAgent(Team.BLUE, llm).choose_guesses(public_state())
    assert choice.indices == (1,) and len(llm.calls) == 2


def _hosted(provider: str, answer: dict, requests: list[dict]) -> LLMClient:
    import anthropic
    import httpx2
    import openai

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        text = json.dumps(answer)
        if provider == "openai":
            message = {"role": "assistant", "content": text, "refusal": None}
            return httpx2.Response(
                200,
                json={
                    "id": "c",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "m",
                    "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                },
            )
        return httpx2.Response(
            200,
            json={
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    if provider == "openai":
        return OpenAIClient("m", client=openai.AsyncOpenAI(api_key="k", http_client=http))
    return AnthropicClient("m", client=anthropic.AsyncAnthropic(api_key="k", http_client=http))


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_operative_plays_on_hosted_providers_with_public_prompt(provider: str) -> None:
    requests: list[dict] = []
    llm = _hosted(provider, op_rank("SPACE", ["STAR", "MOON"], best="STAR"), requests)
    choice = await OperativeAgent(Team.BLUE, llm).choose_guesses(public_state())
    assert choice.indices == (1,) and choice.source_clue == "SPACE"
    (sent,) = requests
    assert "assassin" not in json.dumps(sent).casefold()  # no hidden key reaches any provider


# --- clue naturalness and target consistency -------------------------------------


async def test_friendly_guess_with_zero_target_overlap_is_not_a_clean_verification(
    tmp_path: Path,
) -> None:
    # Generated for WORD0; the blind operative would pick WORD1 (also ours).
    llm = RoutedLLM(plans(("HUGGABLE", ["WORD0"])), {"HUGGABLE": ([1], [])})
    agent = SpymasterAgent(Team.BLUE, llm)
    rec = MatchRecorder(tmp_path, room_url="https://codenames.game/r/x", players={})
    clue = await RecordedSpymaster(agent, rec, LLMConfig(provider="ollama", model="m")).choose_clue(
        spy_state()
    )
    trace = agent.last_trace
    assert clue.word == "HUGGABLE"  # nothing better: least bad, never an illegal clue
    assert trace.selected.verified and not trace.selected.acceptable
    assert (trace.selected.target_overlap, trace.selected.target_consistent) == (0, False)
    assert "least bad" in trace.note and trace.selection_mode == "fallback"
    assert trace.regenerated == []  # the retry repeated HUGGABLE: nothing fresh to check
    # The one extra generation was spent looking for a clue that means what it says.
    assert llm.kinds == ["generate", "screen", "verify", "generate"]
    assert trace.focused_regeneration == "no verified clue leads to its intended words"
    retry = llm.calls[3]["user"]
    assert "Do not repeat these clues: HUGGABLE" in retry
    (event,) = map(json.loads, rec.events_path.read_text("utf-8").splitlines())
    selection = event["selection"]
    assert (selection["verification_ran"], selection["verification_passed"]) == (True, False)
    assert selection["target_overlap"] == 0 and selection["target_precision"] == 0.0
    assert selection["target_consistency_passed"] is False
    assert selection["generator_targets"] == ["WORD0"]
    assert selection["expected_guesses"] == ["WORD1"]
    assert event["regeneration"] == trace.focused_regeneration
    assert event["selection_mode"] == "fallback"


async def test_consistency_failure_regenerates_once_and_picks_the_fresh_clue() -> None:
    llm = RoutedLLM(
        [plans(("HUGGABLE", ["WORD0"])), plans(("HUGGABLE", ["WORD0"]), ("PILLOW", ["WORD0"]))],
        {"HUGGABLE": ([1], []), "PILLOW": ([0], [])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    clue = await agent.choose_clue(spy_state())
    assert (clue.word, clue.number) == ("PILLOW", 1)
    assert llm.kinds == ["generate", "screen", "verify", "generate", "screen", "verify"]
    assert _clues(llm.calls[4]) == ["PILLOW"]  # only fresh clues are screened again
    selected = agent.last_trace.selected
    assert selected.verified and selected.acceptable and selected.target_overlap == 1
    assert [c.purpose for c in agent.last_trace.calls] == [
        "generate",
        "screen",
        "verify",
        "generate-retry",
        "screen-retry",
        "verify-retry",
    ]
    assert agent.last_trace.regenerated == ["PILLOW"]
    assert agent.last_trace.selection_mode == "regenerated_verified"


async def test_phrase_clues_are_dropped_before_ranking_and_never_submitted(
    tmp_path: Path,
) -> None:
    llm = RoutedLLM(
        plans(("WHATYOUHUG", ["WORD0"]), ("AIRMOVEMENT", ["WORD1"]), ("ORBIT", ["WORD2"])),
        {"WHATYOUHUG": ([0], []), "AIRMOVEMENT": ([1], []), "ORBIT": ([2], [])},
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    rec = MatchRecorder(tmp_path, room_url="https://codenames.game/r/x", players={})
    recorded = RecordedSpymaster(agent, rec, LLMConfig(provider="ollama", model="m"))
    assert (await recorded.choose_clue(spy_state())).word == "ORBIT"
    assert all("WHATYOUHUG" not in _clues(c) for c in llm.calls[1:])
    reasons = dict(agent.last_trace.rejected)
    assert reasons["WHATYOUHUG"] == "model: unnatural: sentence fragment (what + you + hug)"
    assert reasons["AIRMOVEMENT"].startswith("model: unnatural: glued noun phrase")
    (event,) = map(json.loads, rec.events_path.read_text("utf-8").splitlines())
    logged = {r["clue"]: r["reason"] for r in event["rejected_clues"]}
    assert logged["WHATYOUHUG"] == reasons["WHATYOUHUG"]
    assert event["selection"]["target_consistency_passed"] is True
    assert event["selection_mode"] == "verified" and event["regeneration"] is None


async def test_only_illegal_candidates_raise_instead_of_submitting_one() -> None:
    llm = ScriptedLLM()
    with pytest.raises(ValueError, match="No safe clue"):
        await SpymasterAgent(Team.BLUE, llm).choose_clue(
            spy_state(), candidates=[Candidate("WHATYOUHUG", ("WORD0",), "given")]
        )
    assert not llm.calls  # nothing legal to rank


async def test_fallback_skips_illegal_even_if_it_scores_best() -> None:
    # A clue that slipped past preparation (e.g. rules tightened between turns) is still
    # never submitted: the final guard re-checks the rules before choosing.
    llm = RoutedLLM(plans(("ORBIT", ["WORD0"])), {"ORBIT": ([9], [])})
    agent = SpymasterAgent(Team.BLUE, llm)
    original = agent._verify

    async def smuggle(*args: object) -> list:
        final = await original(*args)  # type: ignore[arg-type]
        good = final[0]
        return [*final, replace(good, clue="WHATYOUHUG", win_probability=0.99)]

    agent._verify = smuggle  # type: ignore[method-assign]
    clue = await agent.choose_clue(spy_state())
    assert clue.word == "ORBIT"  # least bad legal clue, not the illegal "best" one


async def test_live_tiny_case_keeps_the_verified_acceptable_number() -> None:
    # Live: PAIR 1 missed its targets, PAIR 2 (verified, acceptable, best) was pruned by
    # it, which triggered a pointless retry and submitted the screened-only TINY.
    llm = RoutedLLM(
        plans(("PAIR", ["WORD0", "WORD1"]), ("TINY", ["WORD2"])),
        {"PAIR": ([0, 1], []), "TINY": ([2, 9, 10], [])},
        {"PAIR": ([3], [0])},  # verified: WORD3 (ours, not intended) first
    )
    agent = SpymasterAgent(Team.BLUE, llm)
    clue = await agent.choose_clue(spy_state())
    trace = agent.last_trace
    assert (clue.word, clue.number) == ("PAIR", 2)
    assert trace.selected.verified and trace.selected.acceptable
    assert trace.selection_mode == "verified" and not trace.focused_regeneration
    assert [c.purpose for c in trace.calls] == ["generate", "screen", "verify"]


def retry_llm(retry_plans: dict, verify_fails: set[str] = frozenset()) -> RoutedLLM:
    """HUGGABLE is generated for WORD0 but ranked to WORD1; the retry offers ``retry_plans``.
    Verifying a clue in ``verify_fails`` returns unusable JSON."""
    llm = RoutedLLM(
        [plans(("HUGGABLE", ["WORD0"])), retry_plans],
        {"HUGGABLE": ([1], []), "PILLOW": ([0], []), "CUSHION": ([1], [])},
    )
    routed = llm.chat_json

    async def chat_json(**kwargs: object) -> dict:
        if _kind(kwargs) == "verify" and set(_clues(kwargs)) & verify_fails:
            llm.calls.append(kwargs)
            raise LLMResponseError("truncated")
        return await routed(**kwargs)

    llm.chat_json = chat_json  # type: ignore[method-assign]
    return llm


async def test_unverified_retry_clue_is_an_explicit_fallback_not_a_verified_pick() -> None:
    llm = retry_llm(plans(("PILLOW", ["WORD0"])), verify_fails={"PILLOW"})
    agent = SpymasterAgent(Team.BLUE, llm)
    clue = await agent.choose_clue(spy_state())
    trace = agent.last_trace
    purposes = [c.purpose for c in trace.calls]
    assert purposes[3:] == ["generate-retry", "screen-retry", "verify-retry"]
    assert not trace.calls[-1].ok  # the retry verification ran and failed
    # PILLOW looked fine when screened, but it was never verified.
    assert clue.word == "PILLOW" and not trace.selected.verified
    assert trace.selection_mode == "fallback"
    assert trace.note == "no verified action is acceptable; using a screened-only estimate"


async def test_retry_clue_that_fails_verification_follows_the_fallback() -> None:
    # CUSHION (for WORD0) is also ranked to WORD1: verified, but inconsistent.
    llm = retry_llm(plans(("CUSHION", ["WORD0"])))
    agent = SpymasterAgent(Team.BLUE, llm)
    await agent.choose_clue(spy_state())
    trace = agent.last_trace
    assert [c.purpose for c in trace.calls][3:] == [
        "generate-retry",
        "screen-retry",
        "verify-retry",
    ]
    assert any(a.clue == "CUSHION" and a.verified for a in trace.assessments)
    assert trace.selected.verified and not trace.selected.acceptable
    assert trace.selection_mode == "fallback" and "least bad" in trace.note
    assert [c.purpose for c in trace.calls].count("generate-retry") == 1  # never twice


async def test_illegal_retry_clues_are_rejected_before_any_check() -> None:
    llm = retry_llm(plans(("WHATYOUHUG", ["WORD0"]), ("AIRMOVEMENT", ["WORD0"])))
    agent = SpymasterAgent(Team.BLUE, llm)
    clue = await agent.choose_clue(spy_state())
    trace = agent.last_trace
    assert clue.word == "HUGGABLE"  # the least bad legal clue, flagged as a fallback
    assert [c.purpose for c in trace.calls][3:] == ["generate-retry"]  # nothing to screen
    reasons = dict(trace.rejected)
    assert reasons["WHATYOUHUG"].startswith("retry: unnatural")
    assert reasons["AIRMOVEMENT"].startswith("retry: unnatural")
    assert trace.regenerated == [] and trace.selection_mode == "fallback"
