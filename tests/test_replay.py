import json
import socket
from pathlib import Path

import pytest
from boards import GAME, board_json, clue, end, guess, write_match

from codenames_ai.domain.enums import CardColor, GamePhase, Team
from codenames_ai.replay import (
    PREDATES,
    ReplayError,
    ReplayUnavailableError,
    load_replay,
)
from codenames_ai.replay.__main__ import main as replay_cli

START = [{"type": "match_started"}]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay is local: any socket use fails the test."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("replay opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def replay(tmp_path: Path, events: list[dict], **kwargs: object):  # type: ignore[no-untyped-def]
    return load_replay(write_match(tmp_path / "match", events, **kwargs))  # type: ignore[arg-type]


def test_normal_match_is_reconstructed_in_order(tmp_path: Path) -> None:
    state = replay(tmp_path, GAME).state
    assert [(t.number, t.team, t.clue, t.clue_number) for t in state.turns] == [
        (1, Team.BLUE, "SOUND", 4),
        (2, Team.RED, "TINY", 2),
        (3, Team.BLUE, "HEARING", 2),
        (4, Team.RED, "SLICE", 2),
        (5, Team.BLUE, "CAPITAL", 2),
    ]
    # Multi-guess turn: four friendly reveals under one clue.
    assert [(g.word, g.result) for g in state.turns[0].guesses] == [
        ("NOTE", "friendly"),
        ("WHISTLE", "friendly"),
        ("BRAZIL", "friendly"),
        ("DANCE", "friendly"),
    ]
    assert (state.winner, state.decided_by) == (Team.BLUE, "all cards")
    assert state.recorded_winner == Team.BLUE and state.termination_reason == "game_over"
    assert state.phase == GamePhase.GAME_OVER and state.current_team is None
    assert len(state.revealed) == 12 and state.remaining(Team.BLUE) == 0
    assert state.remaining(Team.RED) == 6 and not state.warnings


def test_neutral_and_opponent_results_end_the_turn(tmp_path: Path) -> None:
    red = replay(tmp_path, GAME).state.turns
    assert [(g.word, g.side, g.result) for g in red[1].guesses] == [
        ("DWARF", CardColor.RED, "friendly"),
        ("ASH", CardColor.NEUTRAL, "neutral"),
    ]
    assert [(g.word, g.side, g.result) for g in red[3].guesses][-1] == (
        "KITCHEN",
        CardColor.BLUE,
        "opponent",
    )


def test_current_phase_follows_the_events(tmp_path: Path) -> None:
    state = replay(tmp_path, [*START, *clue("blue", "SOUND", 4), guess("blue", "NOTE")]).state
    assert state.phase == GamePhase.BLUE_OPERATIVE and state.current_team == Team.BLUE
    state = replay(tmp_path, [*clue("blue", "SOUND", 4), guess("blue", "ASH")]).state
    assert state.phase == GamePhase.RED_SPYMASTER  # the neutral card ended blue's turn
    state = replay(tmp_path, [*clue("blue", "SOUND", 4), guess("blue", "NOTE"), end("blue")]).state
    assert state.phase == GamePhase.RED_SPYMASTER  # passed after one card
    assert state.winner is None and state.termination_reason is None


def test_assassin_loses_the_game_for_the_guessing_team(tmp_path: Path) -> None:
    events = [
        *clue("blue", "SOUND", 1),
        guess("blue", "NOTE"),
        end("blue"),
        *clue("red", "HEAD", 1),
        guess("red", "SKULL"),
        {"type": "game_over", "winner": "blue"},
        {"type": "match_ended", "reason": "game_over", "winner": "blue", "error": None},
    ]
    result = replay(
        tmp_path, events, metadata={"winner": "blue", "termination_reason": "game_over"}
    )
    assert (result.state.winner, result.state.decided_by) == (Team.BLUE, "assassin")
    assert result.state.turns[-1].guesses[0].result == "assassin"
    assert result.summary.assassin_hits == 1


def test_summary_is_derived_from_the_replayed_events(tmp_path: Path) -> None:
    summary = replay(tmp_path, GAME).summary
    assert summary.to_dict() == {
        "turns": 5,
        "guesses": 12,
        "blue_guesses": 8,
        "red_guesses": 4,
        "neutral_hits": 1,
        "enemy_hits": 1,
        "assassin_hits": 0,
        "winner": "blue",
    }


# --- integrity checks ------------------------------------------------------------


def swap(events: list[dict], position: int, event: dict) -> list[dict]:
    return [*events[:position], event, *events[position + 1 :]]


NOTE_AT = GAME.index(guess("blue", "NOTE"))


@pytest.mark.parametrize(
    "events,problem",
    [
        # A card cannot be newly revealed twice.
        ([*clue("blue", "SOUND", 4), guess("blue", "NOTE"), guess("blue", "NOTE")],
         "NOTE was already revealed"),
        # The recorded result must match the board key.
        ([*clue("blue", "SOUND", 4), guess("blue", "NOTE", result="neutral")],
         "NOTE recorded as 'neutral', but it is blue"),
        ([*clue("blue", "SOUND", 4), guess("blue", "DWARF", result="friendly")],
         "DWARF recorded as 'friendly'.*opponent for blue"),
        # Guesses must name board words, and the index must match the word.
        ([*clue("blue", "SOUND", 4), {**guess("blue", "NOTE"), "word": "TRUMPET"}],
         "'TRUMPET' is not on the board"),
        ([*clue("blue", "SOUND", 4), {**guess("blue", "NOTE"), "index": 3}],
         "NOTE is card 0, but the guess names 3"),
        # Turn order.
        (clue("red", "TINY", 1), "red gave a clue during blue's turn"),
        ([guess("blue", "NOTE")], "guess before any clue"),
        ([*clue("blue", "SOUND", 4), guess("red", "DWARF")], "red guessed outside"),
        ([*clue("blue", "SOUND", 4), guess("blue", "ASH"), guess("blue", "NOTE")],
         "blue guessed outside its operative turn"),
        # Nothing happens after the game was decided.
        ([*clue("blue", "SOUND", 4), guess("blue", "SKULL"), guess("blue", "NOTE")],
         "guess after the game was decided"),
        # The recorded winner must match the reconstructed one.
        (swap(GAME, -2, {"type": "game_over", "winner": "red"}),
         "recorded winner red but the board shows blue won by all cards"),
        ([*clue("blue", "SOUND", 4), {"type": "game_over", "winner": "blue"}],
         "recorded winner blue but the board shows no winner"),
    ],
)  # fmt: skip
def test_inconsistent_recordings_are_rejected(
    tmp_path: Path, events: list[dict], problem: str
) -> None:
    with pytest.raises(ReplayError, match=problem):
        replay(tmp_path, events)


def test_errors_name_the_events_line(tmp_path: Path) -> None:
    events = swap(GAME, NOTE_AT, guess("blue", "NOTE", result="opponent"))
    with pytest.raises(ReplayError, match=f"events.jsonl line {NOTE_AT + 1}: NOTE recorded"):
        replay(tmp_path, events)


def test_match_json_winner_is_checked_too(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="match.json: recorded winner red"):
        replay(tmp_path, GAME, metadata={"winner": "red", "termination_reason": "game_over"})


def test_interrupted_match_replays_without_a_winner_check(tmp_path: Path) -> None:
    events = [*clue("blue", "SOUND", 4), guess("blue", "NOTE"), {
        "type": "match_ended", "reason": "error", "winner": None, "error": "Timeout"}]  # fmt: skip
    state = replay(tmp_path, events).state
    assert state.termination_reason == "error" and state.winner is None
    assert state.phase == GamePhase.BLUE_OPERATIVE


def test_unreadable_live_result_is_settled_by_the_key_with_a_warning(tmp_path: Path) -> None:
    events = [*clue("blue", "SOUND", 4), guess("blue", "ASH", result="unknown")]
    state = replay(tmp_path, events).state
    assert state.turns[0].guesses[0].result == "neutral"
    assert state.warnings == ["events.jsonl line 3: result of ASH unreadable live; key: neutral"]


def test_events_keep_file_order_even_with_a_clock_step(tmp_path: Path) -> None:
    directory = write_match(tmp_path / "match", [*clue("blue", "SOUND", 4), guess("blue", "NOTE")])
    lines = directory.joinpath("events.jsonl").read_text("utf-8").splitlines()
    early = json.loads(lines[2]) | {"timestamp": "2026-09-29T18:40:00.000+03:00"}
    directory.joinpath("events.jsonl").write_text(
        "\n".join([*lines[:2], json.dumps(early)]) + "\n", "utf-8"
    )
    state = load_replay(directory).state
    assert [g.word for g in state.turns[0].guesses] == ["NOTE"]  # applied, not re-sorted
    assert state.warnings == ["events.jsonl line 3: timestamp earlier than the previous event"]


def test_malformed_event_line_is_an_error(tmp_path: Path) -> None:
    directory = write_match(tmp_path / "match", clue("blue", "SOUND", 4))
    with directory.joinpath("events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"type": "guess", "team": \n')
    with pytest.raises(ReplayError, match="events.jsonl line 3: not valid JSON"):
        load_replay(directory)


@pytest.mark.parametrize(
    "board,problem",
    [
        (board_json(cards=board_json()["cards"][:24]), "board.json: board has 24 cards"),
        ({**board_json(), "cards": [{**c, "side": "assassin"} if c["index"] == 0 else c
                                     for c in board_json()["cards"]]}, "2 assassins"),
        ({**board_json(), "cards": [{**c, "index": 1} if c["index"] == 0 else c
                                     for c in board_json()["cards"]]}, "indexes must be 0-24"),
        ({**board_json(), "cards": [{**c, "word": "NOTE"} if c["index"] == 1 else c
                                     for c in board_json()["cards"]]}, "duplicate board words"),
    ],
)  # fmt: skip
def test_malformed_board_is_rejected(tmp_path: Path, board: dict, problem: str) -> None:
    with pytest.raises(ReplayError, match=problem):
        replay(tmp_path, GAME, board=board)


# --- compatibility ---------------------------------------------------------------


def test_old_logs_without_a_board_snapshot_fail_clearly(tmp_path: Path) -> None:
    directory = write_match(tmp_path / "old", GAME, with_board=False)
    directory.joinpath("match.json").write_text(json.dumps({"match_id": "old"}), "utf-8")
    with pytest.raises(ReplayUnavailableError, match=PREDATES):
        load_replay(directory)
    assert PREDATES == "This match predates replayable board snapshots."


def test_versioned_match_whose_capture_failed_says_so(tmp_path: Path) -> None:
    with pytest.raises(ReplayUnavailableError, match="has no board snapshot"):
        replay(tmp_path, GAME, with_board=False)


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="schema_version 99 is newer"):
        replay(tmp_path, GAME, metadata={"schema_version": 99})


# --- CLI -------------------------------------------------------------------------


def test_cli_prints_turns_and_summary(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    directory = write_match(tmp_path / "2026-09-29T184559_zuloh-bovuv", GAME)
    assert replay_cli([str(directory)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Match: 2026-09-29T184559_zuloh-bovuv\nBoard: 25 cards")
    assert "Starting team: Blue" in out
    assert "Turn 1 - Blue\nClue: SOUND 4\n  NOTE -> blue\n  WHISTLE -> blue" in out
    assert "Turn 2 - Red\nClue: TINY 2\n  DWARF -> red\n  ASH -> neutral" in out
    assert "Winner: Blue (all cards)" in out and "  guesses: 12" in out
    assert replay_cli([str(directory), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["winner"] == "blue"


def test_cli_reports_unreplayable_logs(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    directory = write_match(tmp_path / "old", GAME, with_board=False)
    directory.joinpath("match.json").write_text("{}", "utf-8")
    assert replay_cli([str(directory)]) == 1
    assert PREDATES in capsys.readouterr().err
