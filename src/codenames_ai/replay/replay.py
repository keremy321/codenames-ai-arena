"""Rebuild a recorded match from its files, applying events strictly in file order.

Pure and local: reads match.json, board.json and events.jsonl, never a browser, a
model provider or the network. Every event that changes the game is checked against
the board key; an inconsistency raises ReplayError naming the events.jsonl line.
This is an integrity check of the recording, not a full rules engine.
"""

import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from codenames_ai.domain.enums import CardColor, GamePhase, Role, Team
from codenames_ai.recording.board import BOARD_FILE, BoardSnapshot, BoardSnapshotError
from codenames_ai.recording.recorder import SCHEMA_VERSION

from .models import (
    Replay,
    ReplayError,
    ReplayGuess,
    ReplayState,
    ReplayTurn,
    ReplayUnavailableError,
)

PREDATES = "This match predates replayable board snapshots."


def load_replay(directory: str | Path) -> Replay:
    root = Path(directory)
    metadata = _read_json(root / "match.json")
    version = metadata.get("schema_version")
    if version is None:
        raise ReplayUnavailableError(PREDATES)
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        raise ReplayError(
            f"match.json schema_version {version!r} is newer than this replay "
            f"(supports up to {SCHEMA_VERSION})"
        )
    board_path = root / BOARD_FILE
    if not board_path.exists():
        raise ReplayUnavailableError(
            "This match has no board snapshot (board_snapshot_failed in events.jsonl "
            "says why); it cannot be replayed."
        )
    try:
        board = BoardSnapshot.from_json(_read_json(board_path))
    except BoardSnapshotError as exc:
        raise ReplayError(f"{BOARD_FILE}: {exc}") from exc
    state = replay_events(board, read_events(root / "events.jsonl"))
    _check_metadata(state, metadata)
    return Replay(str(root), metadata, state)


def read_events(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """(line number, event) in file order; malformed lines are errors, not skipped."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReplayError(f"Cannot read {path.name}: {exc}") from exc
    events = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError as exc:
            raise ReplayError(f"{path.name} line {line_no}: not valid JSON") from exc
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ReplayError(f"{path.name} line {line_no}: not an event object")
        events.append((line_no, event))
    return events


def replay_events(
    board: BoardSnapshot, events: Iterable[tuple[int, Mapping[str, Any]]]
) -> ReplayState:
    start = board.starting_team
    state = ReplayState(
        board=board,
        phase=GamePhase(f"{start}_{Role.SPYMASTER}") if start else GamePhase.UNKNOWN,
    )
    words = {card.word.casefold(): card for card in board.cards}
    previous: datetime | None = None
    for line_no, event in events:
        where = f"events.jsonl line {line_no}"
        previous = _check_time(state, where, event, previous)
        kind = event["type"]
        if kind == "clue":
            _clue(state, where, event)
        elif kind == "guess":
            _guess(state, where, event, words)
        elif kind == "turn_ended":
            _turn_ended(state, where, event)
        elif kind == "game_over":
            state.recorded_winner = _winner(where, event.get("winner"))
            _check_winner(state, where, state.recorded_winner)
        elif kind == "match_ended":
            state.termination_reason = event.get("reason")
            if event.get("reason") == "game_over":
                _check_winner(state, where, _winner(where, event.get("winner")))
        # Other events (decisions, timings, setup) do not change the game.
    if state.winner is not None or state.termination_reason == "game_over":
        state.phase = GamePhase.GAME_OVER
    return state


def _clue(state: ReplayState, where: str, event: Mapping[str, Any]) -> None:
    team = _team(where, event.get("team"))
    _require_live(state, where, "clue")
    expected = state.current_team
    if expected is not None and team is not expected:
        raise ReplayError(f"{where}: {team} gave a clue during {expected}'s turn")
    word, number = event.get("word"), event.get("number")
    if not isinstance(word, str) or not word.strip():
        raise ReplayError(f"{where}: clue has no word")
    if not isinstance(number, int) or isinstance(number, bool) or number < 0:
        raise ReplayError(f"{where}: clue {word} has invalid number {number!r}")
    state.turns.append(ReplayTurn(len(state.turns) + 1, team, word, number))
    state.phase = GamePhase(f"{team}_{Role.OPERATIVE}")


def _guess(
    state: ReplayState, where: str, event: Mapping[str, Any], words: Mapping[str, Any]
) -> None:
    team = _team(where, event.get("team"))
    _require_live(state, where, "guess")
    if not state.turns:
        raise ReplayError(f"{where}: guess before any clue")
    turn = state.turns[-1]
    if team is not turn.team or state.phase != GamePhase(f"{team}_{Role.OPERATIVE}"):
        raise ReplayError(f"{where}: {team} guessed outside its operative turn")
    word = event.get("word")
    card = words.get(word.casefold()) if isinstance(word, str) else None
    if card is None:
        raise ReplayError(f"{where}: guessed word {word!r} is not on the board")
    if "index" in event and event["index"] != card.index:
        raise ReplayError(
            f"{where}: {card.word} is card {card.index}, but the guess names {event['index']!r}"
        )
    if card.index in state.revealed:
        raise ReplayError(f"{where}: {card.word} was already revealed")
    truth = result_for(team, card.side)
    recorded = event.get("result")
    if recorded == "unknown":
        # The live reader could not see the colour; the board key settles it.
        state.warnings.append(f"{where}: result of {card.word} unreadable live; key: {truth}")
    elif recorded != truth:
        raise ReplayError(
            f"{where}: {card.word} recorded as {recorded!r}, but it is {card.side} "
            f"({truth} for {team})"
        )
    state.revealed[card.index] = team
    turn.guesses.append(
        ReplayGuess(card.word, card.index, card.side, truth, event.get("bonus") is True)
    )
    if card.side is CardColor.ASSASSIN:
        state.winner, state.decided_by = team.other, "assassin"
    elif card.side in (CardColor.BLUE, CardColor.RED):
        owner = Team(card.side.value)
        if state.remaining(owner) == 0:
            state.winner, state.decided_by = owner, "all cards"
    if state.winner is not None:
        state.phase = GamePhase.GAME_OVER
    elif truth != "friendly":
        state.phase = GamePhase(f"{team.other}_{Role.SPYMASTER}")  # a miss ends the turn


def _turn_ended(state: ReplayState, where: str, event: Mapping[str, Any]) -> None:
    team = _team(where, event.get("team"))
    role = event.get("role")
    if not state.turns or team is not state.turns[-1].team:
        raise ReplayError(f"{where}: {team} {role} turn ended, but it was not {team}'s turn")
    if role == Role.OPERATIVE and state.winner is None:
        state.phase = GamePhase(f"{team.other}_{Role.SPYMASTER}")


def _require_live(state: ReplayState, where: str, action: str) -> None:
    if state.winner is not None:
        raise ReplayError(f"{where}: {action} after the game was decided ({state.winner} won)")


def _check_winner(state: ReplayState, where: str, recorded: Team | None) -> None:
    if recorded is not state.winner:
        board = f"{state.winner} won by {state.decided_by}" if state.winner else "no winner"
        raise ReplayError(f"{where}: recorded winner {recorded} but the board shows {board}")


def _check_metadata(state: ReplayState, metadata: Mapping[str, Any]) -> None:
    if metadata.get("termination_reason") == "game_over":
        _check_winner(state, "match.json", _winner("match.json", metadata.get("winner")))
    if state.termination_reason is None:
        state.termination_reason = metadata.get("termination_reason")


def _check_time(
    state: ReplayState, where: str, event: Mapping[str, Any], previous: datetime | None
) -> datetime | None:
    raw = event.get("timestamp")
    if not isinstance(raw, str):
        return previous
    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ReplayError(f"{where}: invalid timestamp {raw!r}") from exc
    if previous is not None and stamp < previous:
        # File order is the recorded order; it is kept, never re-sorted.
        state.warnings.append(f"{where}: timestamp earlier than the previous event")
    return stamp


def _team(where: str, value: Any) -> Team:
    try:
        return Team(value)
    except ValueError as exc:
        raise ReplayError(f"{where}: invalid team {value!r}") from exc


def _winner(where: str, value: Any) -> Team | None:
    return None if value is None else _team(where, value)


def result_for(team: Team, side: CardColor) -> str:
    """A reveal's result for the guessing team, as the controller records it."""
    if side is CardColor(team.value):
        return "friendly"
    if side in (CardColor.BLUE, CardColor.RED):
        return "opponent"
    return side.value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReplayError(f"No {path.name} in {path.parent}") from exc
    except (OSError, ValueError) as exc:
        raise ReplayError(f"Cannot read {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReplayError(f"{path.name} is not a JSON object")
    return data
