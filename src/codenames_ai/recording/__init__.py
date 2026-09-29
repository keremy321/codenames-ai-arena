"""Persistent match logs: one directory per match with match.json, events.jsonl and
board.json (the initial key, for replay and evaluation only)."""

from .board import BOARD_FILE, BoardSnapshot, BoardSnapshotError, capture_board
from .recorder import (
    LOGS_DIR,
    SCHEMA_VERSION,
    MatchRecorder,
    RecordedOperative,
    RecordedSpymaster,
)

__all__ = [
    "BOARD_FILE",
    "LOGS_DIR",
    "SCHEMA_VERSION",
    "BoardSnapshot",
    "BoardSnapshotError",
    "MatchRecorder",
    "RecordedOperative",
    "RecordedSpymaster",
    "capture_board",
]
