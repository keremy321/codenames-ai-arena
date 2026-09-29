import pytest

from codenames_ai.agents.clue_naturalness import phrase_problem
from codenames_ai.agents.spymaster import clue_problem

# Seen live, or the same constructions.
SMUGGLED = [
    "WHATYOUHUG",
    "THINGYOUWEAR",
    "AIRMOVEMENT",
    "PLACEYOUWORK",
    "PLACETOSLEEP",
    "WHEREYOUSLEEP",
    "HOWTOCOOK",
    "KINDOFBLUE",
    "FOODYOUEAT",
    "BLACKANDWHITE",
    "HEARTOFGOLD",
    "WATERPOLLUTION",
    "AIRQUALITY",
    "FIRESTATION",
    "EruptingMountain",
    "thing-you-wear",
]

# Real words, including compounds and words that contain the same pieces.
REAL = [
    "snowfall", "football", "sunflower", "heartbreaking", "earthquake", "waterfall",
    "fireplace", "workplace", "moonlight", "firepower", "timetable", "bodyguard",
    "rainbow", "headquarters", "homesickness", "seasickness", "airworthiness",
    "eyewitness", "mindfulness", "homelessness", "childishness", "carnation", "starvation",
    "plantation", "coloration", "government", "apartment", "tournament", "management",
    "statement", "signature", "redemption", "layout", "payouts", "tryouts", "bayou",
    "joyous", "youth", "yourself", "commandment", "islanders", "wonderland", "husbandry",
    "demanding", "understand", "handoffs", "playoffs", "waterproof", "whereabouts",
    "wherewithal", "wherefore", "whereupon", "whatever", "whatsoever", "somewhere",
    "someone", "typewriter", "placeholder", "kindergarten", "stuffing", "objection",
    "personality", "howitzer", "wayfarer", "together", "inasmuch",
    "cartoonist", "handedness", "international", "ice-cold", "orbit",
]  # fmt: skip


@pytest.mark.parametrize("clue", SMUGGLED)
def test_phrase_smuggling_is_rejected(clue: str) -> None:
    assert phrase_problem(clue) is not None
    assert clue_problem(clue, {"moon"}) is not None


@pytest.mark.parametrize("clue", REAL)
def test_real_words_and_compounds_stay_legal(clue: str) -> None:
    assert phrase_problem(clue) is None
    assert clue_problem(clue, {"piano"}) is None


def test_rejection_names_the_phrase_parts() -> None:
    assert clue_problem("WHATYOUHUG", set()) == ("unnatural: sentence fragment (what + you + hug)")
    assert clue_problem("AIRMOVEMENT", set()) == "unnatural: glued noun phrase (air + movement)"
