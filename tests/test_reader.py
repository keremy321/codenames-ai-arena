import logging

import pytest

from codenames_ai.browser import selectors as s
from codenames_ai.browser.reader import (
    DOMParseError,
    detect_turn,
    normalize_word,
    parse_card,
    parse_card_color,
    parse_clue,
)
from codenames_ai.domain.enums import CardColor, Role, Team


@pytest.mark.parametrize(
    "text,expected",
    [
        ("MILE\nMILE", "MILE"),
        ("  MILE \n\n MILE  ", "MILE"),
        ("NEW YORK\nNEW YORK", "NEW YORK"),
        ("MOON", "MOON"),
    ],
)
def test_duplicate_words(text: str, expected: str) -> None:
    assert normalize_word(text) == expected


@pytest.mark.parametrize("text", ["", " \n", "MOON\nKING"])
def test_ambiguous_words_fail(text: str) -> None:
    with pytest.raises(DOMParseError):
        normalize_word(text)


@pytest.mark.parametrize(
    "token,color",
    [
        ("blue", CardColor.BLUE),
        ("red", CardColor.RED),
        ("neutral", CardColor.NEUTRAL),
    ],
)
def test_color(token: str, color: CardColor) -> None:
    assert parse_card_color(f"opacity: 1; --CardColor: var( --{token}-cardBg );") == color


@pytest.mark.parametrize(
    "style",
    ["", "--CardColor: black", "--other:var(--blue-cardBg)", "--CardColor:var(--assassin-cardBg)"],
)
def test_unknown_color_warns(style: str, caplog: pytest.LogCaptureFixture) -> None:
    assert parse_card_color(style, word="MOON") == CardColor.UNKNOWN
    assert "Unable to determine card color" in caplog.text


def test_operative_parser_does_not_log_or_return_styles(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    card = parse_card(0, "MOON", "SECRET")
    assert card.color is None
    assert "SECRET" not in caplog.text
    assert (
        parse_card(0, "MOON", "--CardColor:var(--red-cardBg)", revealed=True).color == CardColor.RED
    )


def test_clue(caplog: pytest.LogCaptureFixture) -> None:
    clue = parse_clue([" SPACE ", "SPACE"], " 2 ")
    assert clue is not None and clue.word == "SPACE" and clue.number == 2
    assert parse_clue([], None) is None
    assert parse_clue(["SPACE"], None) is None
    assert "Clue number" in caplog.text
    assert parse_clue(["SPACE"], "infinity") is None
    assert parse_clue(["SPACE", "OTHER"], "2") is None
    assert parse_clue(["SPACE"], "0").number == 0


def test_turn_detection_is_not_identity() -> None:
    turn = detect_turn("Tap on cards you think match the clue", [s.RED_OPERATIVE])
    assert (turn.team, turn.role) == (Team.RED, Role.OPERATIVE)
    assert detect_turn("Give your operatives a clue", []).role == Role.SPYMASTER
    assert detect_turn("Give your operatives a clue", [s.RED_OPERATIVE]).role is None
    assert detect_turn("", [s.RED_OPERATIVE, s.BLUE_OPERATIVE]).team is None
