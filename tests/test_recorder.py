import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from codenames_ai.agents.association import Fit, RankedWord
from codenames_ai.agents.base import LLMCall
from codenames_ai.domain.enums import CardColor, Role, Team
from codenames_ai.domain.models import (
    Card,
    Clue,
    ClueDecision,
    GuessDecision,
    PublicGameState,
    SpymasterGameState,
)
from codenames_ai.llm.base import LLMConfig
from codenames_ai.recording import MatchRecorder, RecordedOperative, RecordedSpymaster

SECRET = "sk-ant-SECRET-0123456789"
ROOM = "https://codenames.game/room/radun-kapuf"
QWEN = LLMConfig(provider="ollama", model="qwen3:14b")
CLAUDE = LLMConfig(provider="anthropic", model="claude-x")


def recorder(root: Path) -> MatchRecorder:
    return MatchRecorder(
        root,
        room_url=ROOM,
        players={"blue_spymaster": CLAUDE, "blue_operative": QWEN},
        secret_values=[SECRET],
    )


def events(rec: MatchRecorder) -> list[dict]:
    return [json.loads(line) for line in rec.events_path.read_text("utf-8").splitlines()]


def metadata(rec: MatchRecorder) -> dict:
    return json.loads(rec.metadata_path.read_text("utf-8"))


def test_creates_logs_directory_and_initial_metadata(tmp_path: Path) -> None:
    rec = recorder(tmp_path / "logs")
    assert rec.directory.parent == tmp_path / "logs"
    assert rec.match_id.endswith("_radun-kapuf")
    meta = metadata(rec)
    assert meta["match_id"] == rec.match_id == rec.directory.name
    assert (meta["room_url"], meta["room_code"]) == (ROOM, "radun-kapuf")
    assert meta["players"] == {
        "blue_spymaster": {"provider": "anthropic", "model": "claude-x"},
        "blue_operative": {"provider": "ollama", "model": "qwen3:14b"},
    }
    assert meta["ended_at"] is None and meta["termination_reason"] is None


def test_same_second_matches_get_distinct_directories(tmp_path: Path) -> None:
    first, second = recorder(tmp_path), recorder(tmp_path)
    assert first.directory != second.directory
    assert metadata(second)["match_id"] == second.directory.name


def test_events_are_valid_jsonl(tmp_path: Path) -> None:
    rec = recorder(tmp_path)
    rec.start()
    rec.event("clue", team=Team.BLUE, word="SPACE", number=3)
    rec.event("guess", team=Team.BLUE, word="MOON", result="friendly")
    lines = events(rec)
    assert [e["type"] for e in lines] == ["match_started", "clue", "guess"]
    assert lines[1] == {
        "type": "clue",
        "timestamp": lines[1]["timestamp"],
        "team": "blue",
        "word": "SPACE",
        "number": 3,
    }
    assert all(e["timestamp"] for e in lines)


def test_finish_finalizes_metadata_once(tmp_path: Path) -> None:
    rec = recorder(tmp_path)
    rec.start()
    rec.finish(reason="game_over", winner=Team.RED)
    rec.finish(reason="error", error="late cleanup")  # ignored: already final
    meta = metadata(rec)
    assert meta["termination_reason"] == "game_over" and meta["winner"] == "red"
    assert meta["started_at"] and meta["ended_at"] and meta["duration_seconds"] >= 0
    assert meta["error"] is None
    assert [e["type"] for e in events(rec)] == ["match_started", "match_ended"]


def test_crash_leaves_partial_log_and_error_reason(tmp_path: Path) -> None:
    rec = recorder(tmp_path)
    rec.event("room_ready")  # the match never started
    rec.finish(reason="error", error="BrowserIntegrationError: lobby changed")
    meta = metadata(rec)
    assert meta["termination_reason"] == "error"
    assert meta["error"] == "BrowserIntegrationError: lobby changed"
    assert meta["started_at"] is None and meta["duration_seconds"] is not None
    assert [e["type"] for e in events(rec)] == ["room_ready", "match_ended"]


def test_secrets_are_never_written(tmp_path: Path) -> None:
    rec = recorder(tmp_path)
    rec.event("agent_error", error=f"401 invalid x-api-key {SECRET}")
    rec.finish(reason="error", error=f"AuthenticationError {SECRET}")
    for path in rec.directory.iterdir():
        text = path.read_text("utf-8")
        assert SECRET not in text
    assert "[REDACTED]" in rec.events_path.read_text("utf-8")


def test_write_failure_is_reported_but_never_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocked = tmp_path / "logs"
    blocked.write_text("a file where the logs directory should be")
    with caplog.at_level(logging.ERROR):
        rec = recorder(blocked)
        rec.start()
        rec.event("clue", word="SPACE")
        rec.finish(reason="game_over")
    assert caplog.text.count("the game continues unrecorded") == 1


@dataclass
class Trace:
    calls: list[LLMCall] = field(default_factory=list)
    selected: object = None
    mode: str = "current"
    note: str = ""


@dataclass
class Selected:
    # Generator proposed three words; the operative is expected to find two of them.
    intended: tuple[str, ...] = ("MOON", "STAR", "SUN")
    hits: tuple[str, ...] = ("MOON", "STAR")
    ranking: tuple[RankedWord, ...] = (
        RankedWord("MOON", Fit.STRONG),
        RankedWord("STAR", Fit.POSSIBLE),
    )
    verified: bool = True
    acceptable: bool = True
    vetoed: bool = False
    expected_value: float = 1.23456
    expected_hits: float = 1.5
    p_first_friendly: float = 0.9
    finish_chance: float = 0.0
    win_probability: float = 0.6
    reason: str = "best"


class Spymaster:
    team, role = Team.BLUE, Role.SPYMASTER

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.last_trace: Trace | None = None

    async def choose_clue(self, state: SpymasterGameState) -> ClueDecision:
        self.last_trace = Trace([LLMCall("generate", 1.25, True)], Selected())
        if self.fail:
            self.last_trace.calls.append(LLMCall("verify", 0.5, False, "timed out"))
            raise ValueError("No safe clue")
        return ClueDecision(word="SPACE", number=2)


class Operative:
    team, role = Team.BLUE, Role.OPERATIVE

    def __init__(self) -> None:
        self.seen: list[PublicGameState] = []
        self.last_trace = Trace()

    async def choose_guesses(
        self, state: PublicGameState, *, guesses_made: int, max_guesses: int
    ) -> GuessDecision:
        self.seen.append(state)
        return GuessDecision(indices=(1,), clue_words=("STAR",))


def cards() -> tuple[Card, ...]:
    return (
        Card(index=0, word="MOON", color=CardColor.BLUE),
        Card(index=1, word="STAR", color=CardColor.BLUE),
        Card(index=2, word="KING", color=CardColor.ASSASSIN),
    )


async def test_recorded_spymaster_logs_decision_and_metrics(tmp_path: Path) -> None:
    rec = recorder(tmp_path)
    decision = await RecordedSpymaster(Spymaster(), rec, CLAUDE).choose_clue(
        SpymasterGameState(team=Team.BLUE, cards=cards())
    )
    assert decision == ClueDecision(word="SPACE", number=2)  # unchanged
    (event,) = events(rec)
    assert event["type"] == "spymaster_decision"
    assert (event["team"], event["role"], event["provider"], event["model"]) == (
        "blue",
        "spymaster",
        "anthropic",
        "claude-x",
    )
    assert (event["clue"], event["number"]) == ("SPACE", 2)
    assert event["llm_calls"] == [{"purpose": "generate", "seconds": 1.25, "ok": True}]
    selection = event["selection"]
    assert selection["expected_value"] == 1.2346
    assert "intended" not in selection and "verified" not in selection  # ambiguous names
    assert selection["generator_targets"] == ["MOON", "STAR", "SUN"]
    assert selection["expected_guesses"] == ["MOON", "STAR"]
    assert selection["predicted_ranking"] == [
        {"word": "MOON", "fit": "strong", "side": "friendly"},
        {"word": "STAR", "fit": "possible", "side": "friendly"},
    ]
    assert (selection["verification_ran"], selection["verification_passed"]) == (True, True)
    assert event["selection_note"] is None


async def test_verified_but_unsafe_selection_is_not_reported_as_passed(tmp_path: Path) -> None:
    """A CAPITAL 1 style pick: verification ran, the operative's top pick is neutral."""

    class Unsafe(Spymaster):
        async def choose_clue(self, state: SpymasterGameState) -> ClueDecision:
            self.last_trace = Trace(
                selected=Selected(
                    intended=("SUN",),
                    hits=(),
                    ranking=(RankedWord("BEIJING", Fit.STRONG), RankedWord("SUN", Fit.POSSIBLE)),
                    acceptable=False,
                    expected_value=-0.4,
                    p_first_friendly=0.09,
                    reason="operative's top pick BEIJING is neutral",
                ),
                note="no acceptable action; using the least bad non-vetoed candidate",
            )
            return ClueDecision(word="CAPITAL", number=1)

    rec = recorder(tmp_path)
    await RecordedSpymaster(Unsafe(), rec, CLAUDE).choose_clue(
        SpymasterGameState(
            team=Team.BLUE,
            cards=(
                *cards(),
                Card(index=3, word="SUN", color=CardColor.BLUE),
                Card(index=4, word="BEIJING", color=CardColor.NEUTRAL),
            ),
        )
    )
    (event,) = events(rec)
    selection = event["selection"]
    assert (selection["verification_ran"], selection["verification_passed"]) == (True, False)
    assert selection["acceptable"] is False
    assert selection["predicted_ranking"][0] == {
        "word": "BEIJING",
        "fit": "strong",
        "side": "neutral",
    }
    assert event["selection_note"].startswith("no acceptable action")


async def test_unverified_selection_has_no_pass_verdict(tmp_path: Path) -> None:
    class Unverified(Spymaster):
        async def choose_clue(self, state: SpymasterGameState) -> ClueDecision:
            self.last_trace = Trace(selected=Selected(verified=False))
            return ClueDecision(word="SPACE", number=2)

    rec = recorder(tmp_path)
    await RecordedSpymaster(Unverified(), rec, CLAUDE).choose_clue(
        SpymasterGameState(team=Team.BLUE, cards=cards())
    )
    selection = events(rec)[0]["selection"]
    assert (selection["verification_ran"], selection["verification_passed"]) == (False, None)


async def test_recorded_agent_error_is_logged_and_reraised(tmp_path: Path) -> None:
    rec = recorder(tmp_path)
    with pytest.raises(ValueError, match="No safe clue"):
        await RecordedSpymaster(Spymaster(fail=True), rec, CLAUDE).choose_clue(
            SpymasterGameState(team=Team.BLUE, cards=cards())
        )
    (event,) = events(rec)
    assert event["type"] == "agent_error" and event["error"] == "ValueError: No safe clue"
    assert event["llm_calls"][-1] == {
        "purpose": "verify",
        "seconds": 0.5,
        "ok": False,
        "error": "timed out",
    }


async def test_recorded_operative_sees_only_its_public_state(tmp_path: Path) -> None:
    rec = recorder(tmp_path)
    rec.event("spymaster_decision", selection={"intended": ["MOON", "STAR"]})
    agent = Operative()
    state = PublicGameState(team=Team.BLUE, cards=cards(), clue=Clue(word="SPACE", number=2))
    decision = await RecordedOperative(agent, rec, QWEN).choose_guesses(
        state, guesses_made=0, max_guesses=3
    )
    assert decision.indices == (1,)
    assert agent.seen == [state]  # the exact object: nothing recorded was added
    assert "assassin" not in agent.seen[0].model_dump_json()
    event = events(rec)[-1]
    assert event["type"] == "operative_decision"
    assert (event["clue"], event["guesses"], event["end_turn"]) == ("SPACE", ["STAR"], False)
    assert (event["provider"], event["model"]) == ("ollama", "qwen3:14b")


def test_attach_saves_binary_next_to_logs(tmp_path: Path) -> None:
    rec = recorder(tmp_path)
    assert rec.attach("startup_failure.png", b"\x89PNG")
    assert (rec.directory / "startup_failure.png").read_bytes() == b"\x89PNG"
