import ast
import json
from pathlib import Path

import pytest
from boards import WORDS, board_json, spymaster_state, standard_cards

from codenames_ai.domain.enums import CardColor, Team
from codenames_ai.domain.models import Card, PublicGameState, SpymasterGameState
from codenames_ai.recording import (
    SCHEMA_VERSION,
    BoardSnapshot,
    BoardSnapshotError,
    MatchRecorder,
    capture_board,
)


def recorder(tmp_path: Path) -> MatchRecorder:
    return MatchRecorder(tmp_path, room_url="https://codenames.game/r/board-test", players={})


def events(rec: MatchRecorder) -> list[dict]:
    return [json.loads(line) for line in rec.events_path.read_text("utf-8").splitlines()]


def test_snapshot_has_all_25_cards_with_their_sides() -> None:
    snapshot = BoardSnapshot.from_state(spymaster_state(), "blue_spymaster")
    data = snapshot.to_json(SCHEMA_VERSION)
    assert data["schema_version"] == SCHEMA_VERSION and data["source"] == "blue_spymaster"
    assert data["starting_team"] == Team.BLUE  # nine blue cards
    assert [c["index"] for c in data["cards"]] == list(range(25))
    assert data["cards"][0] == {"index": 0, "word": "NOTE", "side": CardColor.BLUE}
    assert sum(c["side"] == "assassin" for c in data["cards"]) == 1
    assert {c["side"] for c in data["cards"]} == {"blue", "red", "neutral", "assassin"}
    assert BoardSnapshot.from_json(json.loads(json.dumps(data))) == snapshot


def red_starts() -> tuple[Card, ...]:
    swap = {CardColor.BLUE: CardColor.RED, CardColor.RED: CardColor.BLUE}
    return tuple(
        c.model_copy(update={"color": swap.get(c.color, c.color)}) for c in standard_cards()
    )


def test_starting_team_comes_from_the_key() -> None:
    state = SpymasterGameState(team=Team.BLUE, cards=red_starts())
    assert BoardSnapshot.from_state(state, "blue_spymaster").starting_team == Team.RED


def cards_with(**changes: object) -> tuple[Card, ...]:
    cards = list(standard_cards())
    for index, update in changes.items():
        cards[int(index.removeprefix("i"))] = cards[int(index.removeprefix("i"))].model_copy(
            update=update  # type: ignore[arg-type]
        )
    return tuple(cards)


@pytest.mark.parametrize(
    "cards,problem",
    [
        (standard_cards()[:24], "24 cards"),
        (cards_with(i3={"color": CardColor.ASSASSIN}), "2 assassins"),
        (cards_with(i24={"color": CardColor.NEUTRAL}), "0 assassins"),
        (cards_with(i1={"word": "NOTE"}), "duplicate board words: note"),
        (cards_with(i1={"index": 0}), "indexes must be 0-24"),
        (cards_with(i5={"color": CardColor.UNKNOWN}), r"card 5 \(FIELD\) has no readable side"),
        (cards_with(i5={"color": None}), "no readable side"),
        (cards_with(i5={"revealed": True}), "already has revealed cards"),
    ],
)
def test_incomplete_or_invalid_boards_are_refused(cards: tuple, problem: str) -> None:
    with pytest.raises(BoardSnapshotError, match=problem):
        BoardSnapshot.from_state(SpymasterGameState(team=Team.BLUE, cards=cards), "x")


@pytest.mark.parametrize(
    "board,problem",
    [
        (board_json(cards="nope"), "no card list"),
        (board_json(cards=board_json()["cards"][:20]), "20 cards"),
        (board_json(starting_team="red"), "starting_team 'red' does not match"),
        ({**board_json(), "cards": [*board_json()["cards"][:24], {"index": 24, "word": "X"}]},
         "invalid side None"),
    ],
)  # fmt: skip
def test_malformed_board_json_is_refused(board: dict, problem: str) -> None:
    with pytest.raises(BoardSnapshotError, match=problem):
        BoardSnapshot.from_json(board)


def test_match_json_is_versioned_and_board_is_written_once(tmp_path: Path) -> None:
    rec = recorder(tmp_path)
    assert json.loads(rec.metadata_path.read_text("utf-8"))["schema_version"] == SCHEMA_VERSION
    first = BoardSnapshot.from_state(spymaster_state(), "blue_spymaster")
    other = BoardSnapshot.from_state(
        SpymasterGameState(team=Team.BLUE, cards=red_starts()), "red_spymaster"
    )
    assert rec.record_board(first)
    assert not rec.record_board(other)  # ground truth is never overwritten
    saved = json.loads(rec.board_path.read_text("utf-8"))
    assert saved["source"] == "blue_spymaster" and saved["starting_team"] == "blue"
    (event,) = events(rec)  # one pointer event, not the key itself
    assert event["type"] == "board_snapshot" and event["file"] == "board.json"
    assert "cards" not in event and "NOTE" not in json.dumps(event)


async def test_capture_waits_for_the_key_to_render(tmp_path: Path) -> None:
    rec = recorder(tmp_path)
    reads = [
        SpymasterGameState(team=Team.BLUE, cards=standard_cards()[:10]),  # still rendering
        SpymasterGameState(team=Team.BLUE, cards=cards_with(i7={"color": CardColor.UNKNOWN})),
        spymaster_state(),
    ]

    async def read() -> SpymasterGameState:
        return reads.pop(0)

    assert await capture_board(read, rec, source="blue_spymaster", delay=0) is True
    assert not reads
    assert BoardSnapshot.from_json(json.loads(rec.board_path.read_text("utf-8"))).cards[7].side
    assert [e["type"] for e in events(rec)] == ["board_snapshot"]


async def test_failed_capture_is_logged_and_never_raises(tmp_path: Path) -> None:
    rec = recorder(tmp_path)

    async def broken() -> SpymasterGameState:
        raise RuntimeError("page closed")

    assert await capture_board(broken, rec, source="blue_spymaster", attempts=3, delay=0) is False
    assert not rec.board_path.exists()
    (event,) = events(rec)
    assert event["type"] == "board_snapshot_failed"
    assert event["reason"] == "RuntimeError: page closed"


async def test_started_board_is_not_retried(tmp_path: Path) -> None:
    rec = recorder(tmp_path)
    calls = []

    async def started() -> SpymasterGameState:
        calls.append(1)
        return SpymasterGameState(team=Team.BLUE, cards=cards_with(i0={"revealed": True}))

    assert not await capture_board(started, rec, source="blue_spymaster", delay=0)
    assert len(calls) == 1
    assert "revealed" in events(rec)[0]["reason"]


def test_operative_state_still_hides_every_unrevealed_side() -> None:
    public = PublicGameState(team=Team.RED, cards=standard_cards())
    assert all(card.color is None for card in public.cards)
    dumped = public.model_dump_json()
    assert '"assassin"' not in dumped and '"neutral"' not in dumped
    assert all(card["color"] is None for card in json.loads(dumped)["cards"])
    assert [c.word for c in public.cards] == WORDS


SRC = Path(__file__).resolve().parents[1] / "src" / "codenames_ai"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text("utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            names.add(prefix)
            names.update(f"{prefix}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.parametrize(
    "path", sorted([*(SRC / "agents").glob("*.py"), SRC / "controller.py"]), ids=lambda p: p.name
)
def test_agents_and_controller_cannot_reach_the_recorded_key(path: Path) -> None:
    names = imported_modules(path)
    leaks = {n for n in names if "replay" in n or "board" in n.split(".")[-1] or "capture" in n}
    assert not leaks, f"{path.name} imports {leaks}"
