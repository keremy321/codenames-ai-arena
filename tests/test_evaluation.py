"""The offline harness must keep running; its semantic results need a real model."""

from typing import Any

import pytest

from codenames_ai.evaluation.cases import OPERATIVE_CASES, SPYMASTER_CASES, Board
from codenames_ai.evaluation.harness import run_operative_case, run_spymaster_case


class SchemaFakeLLM:
    """Answers any agent schema with the first allowed values (no semantics)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        props = kwargs["schema"]["properties"]
        if "words" in props or "groups" in props:
            friendly = list(props["words"]["properties"]) if "words" in props else []
            return {
                "words": {
                    w: {"themes": [], "specific": [f"idea{chr(97 + i)}"]}
                    for i, w in enumerate(friendly)
                },
                "groups": [],
            }
        items = props["rankings"]["items"]["properties"]
        words = items["strong"]["items"]["enum"]
        reply: dict[str, Any] = {
            "rankings": [
                {"clue": clue, "strong": [words[i % len(words)]], "possible": []}
                for i, clue in enumerate(items["clue"]["enum"])
            ]
        }
        if "best_guess" in props:
            reply["best_guess"] = words[0]
        return reply


def test_fixture_boards_are_valid_and_distinct() -> None:
    names = [c.name for c in (*SPYMASTER_CASES, *OPERATIVE_CASES)]
    assert len(names) == len(set(names)) >= 12
    for case in (*SPYMASTER_CASES, *OPERATIVE_CASES):
        cards = case.board.cards(case.name)
        assert len(cards) == 25 and sorted(c.index for c in cards) == list(range(25))
    with pytest.raises(ValueError):
        Board(blue=("A",), red=(), neutral=(), assassin="X").cards("bad")


async def test_harness_runs_spymaster_case_offline() -> None:
    llm = SchemaFakeLLM()
    result = await run_spymaster_case(SPYMASTER_CASES[0], llm)  # type: ignore[arg-type]
    assert not result.error
    assert [c.purpose for c in result.calls][:2] == ["generate", "screen"]
    assert "clue" in result.data and result.checks
    assert result.as_dict()["calls"]


async def test_harness_runs_operative_cases_offline() -> None:
    for case in OPERATIVE_CASES:
        result = await run_operative_case(case, SchemaFakeLLM())  # type: ignore[arg-type]
        assert not result.error and result.checks
        assert len(result.calls) == 1
