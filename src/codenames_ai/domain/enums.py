from enum import StrEnum


class Team(StrEnum):
    BLUE = "blue"
    RED = "red"

    @property
    def other(self) -> "Team":
        return Team.RED if self is Team.BLUE else Team.BLUE


class Role(StrEnum):
    SPYMASTER = "spymaster"
    OPERATIVE = "operative"


class CardColor(StrEnum):
    BLUE = "blue"
    RED = "red"
    NEUTRAL = "neutral"
    ASSASSIN = "assassin"
    UNKNOWN = "unknown"


class GamePhase(StrEnum):
    WAITING = "waiting"
    BLUE_SPYMASTER = "blue_spymaster"
    BLUE_OPERATIVE = "blue_operative"
    RED_SPYMASTER = "red_spymaster"
    RED_OPERATIVE = "red_operative"
    GAME_OVER = "game_over"
    UNKNOWN = "unknown"
