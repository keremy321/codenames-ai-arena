import pytest

from codenames_ai.browser import selectors as s
from codenames_ai.browser.reader import detect_phase
from codenames_ai.domain.enums import GamePhase


@pytest.mark.parametrize(
    "instruction,panel,clue,expected",
    [
        ("GIVE YOUR OPERATIVES A CLUE", s.BLUE_SPYMASTER, False, GamePhase.BLUE_SPYMASTER),
        ("Wait for RedSpy to give you a clue", s.RED_SPYMASTER, False, GamePhase.RED_SPYMASTER),
        (
            "Wait for your turn, RedSpy is giving a clue",
            s.RED_SPYMASTER,
            False,
            GamePhase.RED_SPYMASTER,
        ),
        ("Tap on cards you think match the clue", s.BLUE_OPERATIVE, True, GamePhase.BLUE_OPERATIVE),
        ("Tap on cards you think match the clue", s.RED_OPERATIVE, True, GamePhase.RED_OPERATIVE),
        ("Give your operatives a clue", s.RED_OPERATIVE, True, GamePhase.UNKNOWN),
        ("Tap on cards you think match the clue", s.BLUE_OPERATIVE, False, GamePhase.UNKNOWN),
        ("", s.BLUE_SPYMASTER, False, GamePhase.UNKNOWN),
    ],
)
def test_multiple_phase_signals(
    instruction: str, panel: str, clue: bool, expected: GamePhase
) -> None:
    assert (
        detect_phase(
            {
                "instruction": instruction,
                "activePanels": [panel],
                "cards": [{}],
                "clueWords": ["ORBIT"] if clue else [],
                "clueNumber": "2" if clue else "",
            }
        )
        == expected
    )


def test_waiting_vs_unknown() -> None:
    assert detect_phase({"lobby": True, "cards": []}) == GamePhase.WAITING
    assert detect_phase({"lobby": False, "cards": []}) == GamePhase.UNKNOWN
