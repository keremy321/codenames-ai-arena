"""Fixed boards for offline decision evaluation (no browser, local Ollama only).

Expectations are soft checks that describe good play; they are measurements for
tuning, not proof of semantic quality. BLUE is always the evaluated team and has 9
cards; RED has 8, neutral 7, assassin 1. Words in ``revealed`` are already public.
"""

import random
from dataclasses import dataclass, field

from codenames_ai.agents.spymaster import PrivateClue
from codenames_ai.domain.enums import CardColor, Team
from codenames_ai.domain.models import Card, PublicClueMemory


@dataclass(frozen=True)
class Board:
    blue: tuple[str, ...]
    red: tuple[str, ...]
    neutral: tuple[str, ...]
    assassin: str
    revealed: frozenset[str] = frozenset()

    def cards(self, seed: str) -> tuple[Card, ...]:
        colored = (
            [(w, CardColor.BLUE) for w in self.blue]
            + [(w, CardColor.RED) for w in self.red]
            + [(w, CardColor.NEUTRAL) for w in self.neutral]
            + [(self.assassin, CardColor.ASSASSIN)]
        )
        if [len(self.blue), len(self.red), len(self.neutral)] != [9, 8, 7]:
            raise ValueError("Evaluation boards need 9 blue, 8 red, 7 neutral, 1 assassin")
        if len({w for w, _ in colored}) != 25 or not self.revealed <= {w for w, _ in colored}:
            raise ValueError("Board words must be 25 distinct words; revealed words on board")
        random.Random(seed).shuffle(colored)
        return tuple(
            Card(index=i, word=w, color=c, revealed=w in self.revealed)
            for i, (w, c) in enumerate(colored)
        )


@dataclass(frozen=True)
class SpymasterCase:
    name: str
    purpose: str
    board: Board
    min_number: int = 1
    max_number: int = 4
    target_pool: frozenset[str] | None = None  # good plans only use these words
    avoid: frozenset[str] = frozenset()  # never guessed, never ranked above our words
    memory: tuple[PrivateClue, ...] = ()
    team: Team = Team.BLUE
    # Fixed candidate set (clue, intended targets) instead of generation, to measure
    # the selection strategy apart from generator variance. The agent never sees which
    # one is expected.
    candidates: tuple[tuple[str, tuple[str, ...]], ...] | None = None
    expect_clue: str | None = None
    expect_win: bool = False
    reject_clue: str | None = None
    min_multi_generated: int = 0  # generation diagnostic (generated cases only)
    min_largest_generated: int = 0  # the largest generated group must reach this size


@dataclass(frozen=True)
class OperativeCase:
    name: str
    purpose: str
    board: Board
    clue: str
    number: int
    guesses_made: int
    history: tuple[PublicClueMemory, ...] = ()
    bonus_only: bool = False
    expect_guess: frozenset[str] | None = None  # None: ending the turn is correct
    expect_source: str | None = None
    team: Team = Team.BLUE
    notes: tuple[str, ...] = field(default=())


def _memory(clue: str, number: int, found: tuple[str, ...] = (), wrong: tuple[str, ...] = ()):
    return PublicClueMemory(
        clue=clue,
        number=number,
        guesses_made=len(found) + len(wrong),
        friendly_hits=len(found),
        unresolved_count=number - len(found),
        words_guessed=found + wrong,
    )


_FOUND_BLUE = ("APPLE", "TRAIN", "CLOCK", "RIVER", "CANDLE", "MIRROR", "TOWER", "BOTTLE", "COIN")
_FOUND_RED = ("FOREST", "HOTEL", "CHAIR", "NURSE", "ROCKET", "CHEESE", "SNOW", "FLAG")


def _race_board(
    ours: tuple[str, ...], theirs: tuple[str, ...], neutral: tuple[str, ...], assassin: str
) -> Board:
    """BLUE (9 cards) has ``ours`` left, RED (8) has ``theirs`` left; the rest are found."""
    blue_found = tuple(w for w in _FOUND_BLUE if w not in ours)[: 9 - len(ours)]
    red_found = tuple(w for w in _FOUND_RED if w not in theirs)[: 8 - len(theirs)]
    return Board(
        blue=ours + blue_found,
        red=theirs + red_found,
        neutral=neutral,
        assassin=assassin,
        revealed=frozenset(blue_found + red_found),
    )


SPYMASTER_CASES: tuple[SpymasterCase, ...] = (
    SpymasterCase(
        name="obvious_two",
        purpose="One clean 2-word link (COFFEE, TEA); a 2 should beat safe 1s.",
        board=Board(
            blue=(
                "COFFEE",
                "TEA",
                "PIANO",
                "KNIGHT",
                "GLACIER",
                "TOOTH",
                "COMPASS",
                "LADDER",
                "SPIDER",
            ),
            red=("BANK", "ROBOT", "FOREST", "HOTEL", "CHAIR", "NURSE", "ROCKET", "CHEESE"),
            neutral=("WALL", "SHOE", "CARPET", "TRUCK", "POLICE", "SNOW", "PEN"),
            assassin="FIRE",
        ),
        min_number=2,
        target_pool=frozenset({"COFFEE", "TEA"}),
    ),
    SpymasterCase(
        name="obvious_three",
        purpose="LION, TIGER, LEOPARD share one direct link; a verified 3 is possible.",
        board=Board(
            blue=(
                "LION",
                "TIGER",
                "LEOPARD",
                "VIOLIN",
                "BRIDGE",
                "CANDLE",
                "PASSPORT",
                "HAMMER",
                "CLOUD",
            ),
            red=("TRAIN", "APPLE", "MIRROR", "SOLDIER", "DIAMOND", "TENT", "BOOK", "OCEAN"),
            neutral=("CHURCH", "SALT", "PENCIL", "ENGINE", "CASTLE", "GLOVE", "RADIO"),
            assassin="ANCHOR",
        ),
        min_number=3,
        target_pool=frozenset({"LION", "TIGER", "LEOPARD"}),
    ),
    SpymasterCase(
        name="hard_board_one",
        purpose="Every friendly word has an opponent/neutral/assassin twin; no enemy or KING.",
        board=Board(
            blue=(
                "GUITAR",
                "PENGUIN",
                "VOLCANO",
                "SCISSORS",
                "PYRAMID",
                "HONEY",
                "SUBMARINE",
                "WALLET",
                "CROWN",
            ),
            red=("VIOLIN", "ICE", "EARTHQUAKE", "KNIFE", "SPHINX", "BEE", "QUEEN", "PURSE"),
            neutral=("DRUM", "SHIP", "MOUNTAIN", "PAPER", "JUNGLE", "TOWER", "CLOCK"),
            assassin="KING",
        ),
        avoid=frozenset({"KING"}),
    ),
    SpymasterCase(
        name="assassin_trap",
        purpose="BEACH, WAVE, SHELL invite a sea clue, but the assassin is SHARK.",
        board=Board(
            blue=(
                "BEACH",
                "WAVE",
                "SHELL",
                "ORCHESTRA",
                "CARROT",
                "LAPTOP",
                "BADGE",
                "MUSEUM",
                "SADDLE",
            ),
            red=("TRUMPET", "POTATO", "KEYBOARD", "SHERIFF", "GALLERY", "HORSE", "CANDY", "WHEEL"),
            neutral=("TOWEL", "CASTLE", "GARLIC", "MOUSE", "PAINT", "BOOT", "DESERT"),
            assassin="SHARK",
        ),
        avoid=frozenset({"SHARK"}),
    ),
    SpymasterCase(
        name="enemy_trap",
        purpose="APPLE, BANANA invite FRUIT, but ORANGE, CHERRY, LEMON are the opponent's.",
        board=Board(
            blue=(
                "APPLE",
                "BANANA",
                "TELESCOPE",
                "PILLOW",
                "MARBLE",
                "CACTUS",
                "HELMET",
                "JACKET",
                "TICKET",
            ),
            red=(
                "ORANGE",
                "CHERRY",
                "LEMON",
                "MICROSCOPE",
                "BLANKET",
                "DESERT",
                "COAT",
                "PASSPORT",
            ),
            neutral=("SUGAR", "LENS", "SHEET", "STATUE", "SAND", "GLOVE", "TRAIN"),
            assassin="MONKEY",
        ),
        avoid=frozenset({"ORANGE", "CHERRY", "LEMON", "MONKEY"}),
    ),
    SpymasterCase(
        name="private_memory_reconnect",
        purpose="Earlier HIGHWAY 3 found only BUS; the spymaster privately knows CAR, TRAIN.",
        board=Board(
            blue=("CAR", "TRAIN", "BUS", "NEEDLE", "PARROT", "OLIVE", "LANTERN", "PLANET", "CARD"),
            red=("TEACHER", "SOAP", "FLAG", "GARDEN", "BUTTON", "ROOF", "WHISTLE", "STAMP"),
            neutral=("CLOCK", "SPOON", "MAP", "SALT", "PAN", "GHOST", "BELL"),
            assassin="THIEF",
            revealed=frozenset({"BUS", "PLANET", "SOAP", "SALT"}),
        ),
        min_number=2,
        target_pool=frozenset({"CAR", "TRAIN"}),
        memory=(PrivateClue("HIGHWAY", 3, ("CAR", "TRAIN", "BUS"), ("CAR", "TRAIN", "BUS")),),
    ),
    # Race awareness. Revealed ("found") cards set the score; BLUE is to move.
    SpymasterCase(
        name="race_match_point",
        purpose="ours=2 theirs=1: a safe 1 hands RED the game; a reasonable 2 wins now.",
        board=_race_board(
            ours=("PIANO", "GUITAR"),
            theirs=("BANK",),
            neutral=("RADIO", "WALL", "SHOE", "CARPET", "TRUCK", "POLICE", "PEN"),
            assassin="SKULL",
        ),
        candidates=(
            ("KEYS", ("PIANO",)),
            ("STRUM", ("GUITAR",)),
            ("INSTRUMENTS", ("PIANO", "GUITAR")),
            ("BAND", ("PIANO", "GUITAR")),
        ),
        min_number=2,
        expect_win=True,
    ),
    SpymasterCase(
        name="race_ahead_safety",
        purpose="ours=5 theirs=8: a safe 1 should beat a 3 that also fits BEAR and WOLF.",
        board=_race_board(
            ours=("LION", "TIGER", "EAGLE", "CANDLE", "MIRROR"),
            theirs=("BEAR", "WOLF", "BANK", "FOREST", "HOTEL", "CHAIR", "NURSE", "ROCKET"),
            neutral=("HORSE", "WALL", "SHOE", "CARPET", "TRUCK", "POLICE", "PEN"),
            assassin="SKULL",
        ),
        candidates=(
            ("ANIMALS", ("LION", "TIGER", "EAGLE")),
            ("PREDATORS", ("LION", "TIGER", "EAGLE")),
            ("STRIPES", ("TIGER",)),
            ("TALONS", ("EAGLE",)),
        ),
        max_number=2,
        avoid=frozenset({"BEAR", "WOLF"}),
    ),
    SpymasterCase(
        name="race_finish_three",
        purpose="ours=3 theirs=1: LION, TIGER, LEOPARD can all go now (generated clues).",
        board=_race_board(
            ours=("LION", "TIGER", "LEOPARD"),
            theirs=("OCEAN",),
            neutral=("CHURCH", "SALT", "PENCIL", "ENGINE", "CASTLE", "GLOVE", "RADIO"),
            assassin="ANCHOR",
        ),
        min_number=3,
        expect_win=True,
        min_largest_generated=3,
    ),
    SpymasterCase(
        name="race_last_card",
        purpose="ours=1 theirs=5: one precise clue for the last card (generated clues).",
        board=_race_board(
            ours=("VOLCANO",),
            theirs=("ICE", "EARTHQUAKE", "KNIFE", "SPHINX", "BEE"),
            neutral=("DRUM", "SHIP", "MOUNTAIN", "PAPER", "JUNGLE", "LADDER", "SCARF"),
            assassin="KING",
        ),
        max_number=1,
        expect_win=True,
    ),
    SpymasterCase(
        name="race_behind_pressure",
        purpose="ours=6 theirs=2: a credible 2 (COFFEE, TEA) should beat a perfect 1.",
        board=_race_board(
            ours=("COFFEE", "TEA", "PIANO", "KNIGHT", "GLACIER", "LADDER"),
            theirs=("BANK", "ROBOT"),
            neutral=("MILK", "WALL", "SHOE", "CARPET", "TRUCK", "POLICE", "PEN"),
            assassin="SKULL",
        ),
        candidates=(
            ("ESPRESSO", ("COFFEE",)),
            ("CHESS", ("KNIGHT",)),
            ("DRINKS", ("COFFEE", "TEA")),
        ),
        min_number=2,
        expect_clue="DRINKS",
    ),
    SpymasterCase(
        name="race_turns_to_finish",
        purpose="ours=4 theirs=4: two safe clues; the 2-card one needs fewer turns.",
        board=_race_board(
            ours=("LION", "TIGER", "CANDLE", "MIRROR"),
            theirs=("BANK", "FOREST", "HOTEL", "CHAIR"),
            neutral=("WALL", "SHOE", "CARPET", "TRUCK", "POLICE", "PEN", "CLOUD"),
            assassin="SKULL",
        ),
        candidates=(("STRIPES", ("TIGER",)), ("FELINES", ("LION", "TIGER"))),
        min_number=2,
        expect_clue="FELINES",
    ),
    # Regressions from the first race-aware live match.
    SpymasterCase(
        name="live_motion_keeps_number",
        purpose="ours=8 theirs=6: MOTION 3 must stay MOTION 3 through verification.",
        board=_race_board(
            ours=("DANCE", "SWING", "WAVE", "LETTUCE", "PIANO", "KNIGHT", "GLACIER", "LADDER"),
            theirs=("BANK", "ROBOT", "HOTEL", "CHAIR", "NURSE", "CHEESE"),
            neutral=("WALL", "SHOE", "CARPET", "TRUCK", "POLICE", "PEN", "CLOUD"),
            assassin="SKULL",
        ),
        candidates=(
            ("MOTION", ("DANCE", "SWING", "WAVE")),
            ("RHYTHM", ("DANCE", "SWING")),
            ("SALAD", ("LETTUCE",)),
        ),
    ),
    SpymasterCase(
        name="live_ride_keeps_number",
        purpose="ours=6 theirs=6: RIDE 2 is its own action and is never mutated to RIDE 1.",
        board=_race_board(
            ours=("HORSE", "BICYCLE", "TOOTH", "COMPASS", "SPIDER", "LADDER"),
            theirs=("BANK", "ROBOT", "HOTEL", "CHAIR", "NURSE", "CHEESE"),
            neutral=("WALL", "SHOE", "CARPET", "POLICE", "PEN", "CLOUD", "GLASS"),
            assassin="SKULL",
        ),
        candidates=(("RIDE", ("HORSE", "BICYCLE")), ("PEDAL", ("BICYCLE",))),
    ),
    SpymasterCase(
        name="critical_four_generated",
        purpose="ours=4 theirs=1: four instruments; the generator must try 3-4 word plans.",
        board=_race_board(
            ours=("GUITAR", "DRUM", "PIANO", "VIOLIN"),
            theirs=("OCEAN",),
            neutral=("WALL", "SHOE", "CARPET", "TRUCK", "POLICE", "PEN", "CLOUD"),
            assassin="SKULL",
        ),
        min_number=3,
        min_multi_generated=2,
        min_largest_generated=3,
        expect_win=True,
    ),
    SpymasterCase(
        name="critical_four_given",
        purpose="ours=4 theirs=1: safe 1, good 2, riskier 4 (JAGUAR is a neutral cat).",
        board=_race_board(
            ours=("LION", "TIGER", "LEOPARD", "CHEETAH"),
            theirs=("OCEAN",),
            neutral=("JAGUAR", "WALL", "SHOE", "CARPET", "TRUCK", "POLICE", "PEN"),
            assassin="SKULL",
        ),
        candidates=(
            ("STRIPES", ("TIGER",)),
            ("SAVANNA", ("LION", "CHEETAH")),
            ("FELINES", ("LION", "TIGER", "LEOPARD", "CHEETAH")),
        ),
        min_number=2,
    ),
    SpymasterCase(
        name="early_unsafe_four",
        purpose="ours=8 theirs=8: FRUIT 4 also fits ORANGE, LEMON, CHERRY; coverage is not enough.",
        board=_race_board(
            ours=("APPLE", "BANANA", "GRAPE", "PEAR", "TELESCOPE", "PILLOW", "HELMET", "JACKET"),
            theirs=(
                "ORANGE",
                "LEMON",
                "CHERRY",
                "MICROSCOPE",
                "BLANKET",
                "DESERT",
                "COAT",
                "PASSPORT",
            ),
            neutral=("SUGAR", "LENS", "SHEET", "STATUE", "SAND", "GLOVE", "TOAST"),
            assassin="MONKEY",
        ),
        candidates=(
            ("FRUIT", ("APPLE", "BANANA", "GRAPE", "PEAR")),
            ("ASTRONOMY", ("TELESCOPE",)),
            ("VINEYARD", ("GRAPE",)),
        ),
        reject_clue="FRUIT",
        avoid=frozenset({"ORANGE", "LEMON", "CHERRY"}),
    ),
    SpymasterCase(
        name="behind_generated_multi",
        purpose="ours=7 theirs=3: several natural pairs; generation must not collapse to 1s.",
        board=_race_board(
            ours=("COFFEE", "TEA", "PIANO", "GUITAR", "KNIGHT", "SWORD", "LADDER"),
            theirs=("BANK", "ROBOT", "HOTEL"),
            neutral=("WALL", "SHOE", "CARPET", "TRUCK", "POLICE", "PEN", "CLOUD"),
            assassin="SKULL",
        ),
        min_number=2,
        min_multi_generated=3,
    ),
)

_SPACE_BOARD = Board(
    blue=("TOAST", "STAR", "MOON", "BUTTON", "RIVER", "PAINT", "LOCK", "HAT", "DRUM"),
    red=("PLANET", "ROCKET", "SHIP", "NOSE", "FIELD", "GLOVE", "KNIFE", "BOX"),
    neutral=("COMET", "TABLE", "SNAKE", "CARD", "BELT", "POOL", "FAN"),
    assassin="ORBIT",
    revealed=frozenset({"TOAST", "STAR"}),
)

OPERATIVE_CASES: tuple[OperativeCase, ...] = (
    OperativeCase(
        name="current_with_old_memory",
        purpose="Old ANIMAL 3 has 2 unresolved; the current KITCHEN clue comes first.",
        board=Board(
            blue=("OVEN", "SPOON", "HORSE", "RABBIT", "DOG", "GUITAR", "CLOUD", "PAPER", "TRUCK"),
            red=("CASTLE", "ROBOT", "OCEAN", "PENCIL", "MIRROR", "TIGER", "BANK", "CROWN"),
            neutral=("GARDEN", "LAMP", "TRAIN", "SHOE", "CHURCH", "MAP", "STAR"),
            assassin="WITCH",
            revealed=frozenset({"DOG"}),
        ),
        clue="KITCHEN",
        number=2,
        guesses_made=0,
        history=(_memory("ANIMAL", 3, found=("DOG",)), _memory("KITCHEN", 2)),
        expect_guess=frozenset({"OVEN", "SPOON"}),
    ),
    OperativeCase(
        name="bonus_should_use",
        purpose="MUSIC 1 is done; ANIMAL has 1 unresolved and GIRAFFE is the only animal.",
        board=Board(
            blue=("GIRAFFE", "PIANO", "DOG", "CAT", "LEMON", "CLOCK", "ROPE", "SWORD", "MOON"),
            red=("PAPER", "TABLE", "BRIDGE", "WINDOW", "COIN", "SOAP", "CHAIR", "TRUCK"),
            neutral=("GLASS", "BOTTLE", "PENCIL", "TOWER", "WALL", "SHOE", "KEY"),
            assassin="CASTLE",
            revealed=frozenset({"PIANO", "DOG", "CAT"}),
        ),
        clue="MUSIC",
        number=1,
        guesses_made=1,
        history=(_memory("ANIMAL", 3, found=("DOG", "CAT")), _memory("MUSIC", 1, found=("PIANO",))),
        bonus_only=True,
        expect_guess=frozenset({"GIRAFFE"}),
        expect_source="ANIMAL",
    ),
    OperativeCase(
        name="bonus_should_not_use",
        purpose="SPACE has 1 unresolved but MOON, PLANET, ROCKET, COMET all fit: stay put.",
        board=_SPACE_BOARD,
        clue="BREAKFAST",
        number=1,
        guesses_made=1,
        history=(_memory("SPACE", 2, found=("STAR",)), _memory("BREAKFAST", 1, found=("TOAST",))),
        bonus_only=True,
        expect_guess=None,
    ),
    OperativeCase(
        name="ambiguous_mid_turn",
        purpose="SPACE 2 found STAR; one word left but four strong candidates: stop.",
        board=_SPACE_BOARD,
        clue="SPACE",
        number=2,
        guesses_made=1,
        history=(_memory("BREAKFAST", 1, found=("TOAST",)), _memory("SPACE", 2, found=("STAR",))),
        expect_guess=None,
    ),
    OperativeCase(
        name="weak_evidence_stop",
        purpose="OCEAN 2 found WAVE; nothing else on the board relates to the ocean.",
        board=Board(
            blue=(
                "WAVE",
                "PENCIL",
                "CASTLE",
                "MIRROR",
                "SOLDIER",
                "CANDLE",
                "HAMMER",
                "BOOK",
                "CLOCK",
            ),
            red=("TRAIN", "APPLE", "DIAMOND", "TENT", "CHURCH", "GARDEN", "ROBOT", "CARPET"),
            neutral=("ENGINE", "GLOVE", "RADIO", "PIANO", "TRUCK", "POLICE", "JACKET"),
            assassin="DRAGON",
            revealed=frozenset({"WAVE"}),
        ),
        clue="OCEAN",
        number=2,
        guesses_made=1,
        history=(_memory("OCEAN", 2, found=("WAVE",)),),
        expect_guess=None,
    ),
    OperativeCase(
        name="strong_continue",
        purpose="ORCHESTRA 3 found VIOLIN; CELLO and FLUTE are the only musical words left.",
        board=Board(
            blue=(
                "VIOLIN",
                "CELLO",
                "FLUTE",
                "BRIDGE",
                "SPOON",
                "TIGER",
                "CLOUD",
                "WALLET",
                "ROPE",
            ),
            red=("TRAIN", "APPLE", "MIRROR", "TENT", "BOOK", "OCEAN", "DIAMOND", "SOLDIER"),
            neutral=("CHURCH", "SALT", "PENCIL", "ENGINE", "CASTLE", "GLOVE", "LAMP"),
            assassin="ANCHOR",
            revealed=frozenset({"VIOLIN"}),
        ),
        clue="ORCHESTRA",
        number=3,
        guesses_made=1,
        history=(_memory("ORCHESTRA", 3, found=("VIOLIN",)),),
        expect_guess=frozenset({"CELLO", "FLUTE"}),
    ),
    OperativeCase(
        name="bonus_per_clue_evidence",
        purpose="SOUND has 1 unresolved but RADIO, WHISTLE, BELL all fit; GEM -> DIAMOND alone.",
        board=Board(
            blue=("RIVER", "DRUM", "DIAMOND", "RADIO", "BALL", "MAPLE", "LOCK", "HAT", "PAINT"),
            red=("WHISTLE", "BELL", "FLAG", "NOSE", "FIELD", "GLOVE", "KNIFE", "BOX"),
            neutral=("TABLE", "SNAKE", "CARD", "BELT", "POOL", "FAN", "COMET"),
            assassin="ORBIT",
            revealed=frozenset({"RIVER", "DRUM"}),
        ),
        clue="WATER",
        number=1,
        guesses_made=1,
        history=(
            _memory("SOUND", 2, found=("DRUM",)),
            _memory("GEM", 1),
            _memory("WATER", 1, found=("RIVER",)),
        ),
        bonus_only=True,
        expect_guess=frozenset({"DIAMOND"}),
        expect_source="GEM",
    ),
)
