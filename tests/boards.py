"""Synthetic standard boards and match logs for recording and replay tests."""

import json
from pathlib import Path
from typing import Any

from codenames_ai.domain.enums import CardColor, Team
from codenames_ai.domain.models import Card, SpymasterGameState

# 9 blue (blue starts), 8 red, 7 neutral, 1 assassin, in index order.
SIDES = [CardColor.BLUE] * 9 + [CardColor.RED] * 8 + [CardColor.NEUTRAL] * 7 + [CardColor.ASSASSIN]
WORDS = [
    "NOTE", "WHISTLE", "BRAZIL", "DANCE", "EAR", "FIELD", "KITCHEN", "WASHINGTON", "POLISH",
    "DWARF", "KNIFE", "JAM", "SEED", "BROOM", "PIANO", "MOON", "STAR",
    "ASH", "POP", "TABLE", "CHAIR", "RIVER", "HORSE", "CLOCK",
    "SKULL",
]  # fmt: skip
SIDE_OF = dict(zip(WORDS, SIDES, strict=True))


def standard_cards() -> tuple[Card, ...]:
    return tuple(
        Card(index=i, word=word, color=side)
        for i, (word, side) in enumerate(zip(WORDS, SIDES, strict=True))
    )


def spymaster_state(team: Team = Team.BLUE) -> SpymasterGameState:
    return SpymasterGameState(team=team, cards=standard_cards())


def board_json(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": 1,
        "source": "blue_spymaster",
        "starting_team": "blue",
        "cards": [
            {"index": i, "word": w, "side": s.value}
            for i, (w, s) in enumerate(zip(WORDS, SIDES, strict=True))
        ],
    }
    return data | overrides


def clue(team: str, word: str, number: int) -> list[dict[str, Any]]:
    """A clue and the spymaster turn ending, as the controller records them."""
    return [
        {"type": "clue", "team": team, "word": word, "number": number},
        {"type": "turn_ended", "team": team, "role": "spymaster"},
    ]


def guess(team: str, word: str, result: str | None = None) -> dict[str, Any]:
    side = SIDE_OF[word]
    if result is None:
        result = (
            "friendly"
            if side.value == team
            else "opponent"
            if side in (CardColor.BLUE, CardColor.RED)
            else side.value
        )
    index = WORDS.index(word)
    return {"type": "guess", "team": team, "word": word, "index": index, "result": result}


def end(team: str) -> dict[str, Any]:
    return {"type": "turn_ended", "team": team, "role": "operative"}


def write_match(
    directory: Path,
    events: list[dict[str, Any]],
    *,
    board: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    with_board: bool = True,
) -> Path:
    """A match directory like the recorder writes; timestamps follow file order."""
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"schema_version": 1, "match_id": directory.name, "termination_reason": None}
    (directory / "match.json").write_text(json.dumps(meta | (metadata or {})), "utf-8")
    if with_board:
        (directory / "board.json").write_text(json.dumps(board or board_json()), "utf-8")
    lines = [
        json.dumps({"timestamp": f"2026-09-29T18:46:{i:02d}.000+03:00", **event})
        for i, event in enumerate(events)
    ]
    (directory / "events.jsonl").write_text("\n".join(lines) + "\n", "utf-8")
    return directory


# A complete game: blue clears all nine cards over four turns; red misses twice.
GAME = [
    {"type": "match_started"},
    {"type": "board_snapshot", "file": "board.json", "starting_team": "blue"},
    *clue("blue", "SOUND", 4),
    guess("blue", "NOTE"),
    guess("blue", "WHISTLE"),
    guess("blue", "BRAZIL"),
    guess("blue", "DANCE"),
    end("blue"),
    *clue("red", "TINY", 2),
    guess("red", "DWARF"),
    guess("red", "ASH"),  # neutral: ends the turn
    end("red"),
    *clue("blue", "HEARING", 2),
    guess("blue", "EAR"),
    guess("blue", "FIELD"),
    end("blue"),
    *clue("red", "SLICE", 2),
    guess("red", "KNIFE"),
    guess("red", "KITCHEN"),  # blue's card: opponent hit
    end("red"),
    *clue("blue", "CAPITAL", 2),
    guess("blue", "WASHINGTON"),
    guess("blue", "POLISH"),  # blue's ninth card: blue wins
    {"type": "game_over", "winner": "blue"},
    {"type": "match_ended", "reason": "game_over", "winner": "blue", "error": None},
]
