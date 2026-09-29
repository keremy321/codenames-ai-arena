"""Reject clues that smuggle a phrase into one token: WHATYOUHUG, AIRMOVEMENT.

Structure plus a few small closed word lists; no dictionary. Every rule needs a
phrase signal inside the clue (a pronoun, a definition opener followed by a function
word, "and" between content words, or a common noun glued to an abstract noun), so
real closed compounds such as SNOWFALL, FOOTBALL, WORKPLACE or HEARTBREAKING never
match. False negatives only fall back to the earlier behaviour; false positives would
lose real clues, so every list stays conservative.
"""

import re

# Pronouns that almost never sit inside a real word (LAYOUT/PAYOUTS are exempted below).
PRONOUNS = ("you", "they")
# Openers of a definition written as a clue: "thing you wear", "place to sleep".
OPENERS = (
    "what", "where", "who", "when", "how", "why", "which", "thing", "things", "stuff",
    "place", "something", "someone", "somewhere", "person", "people", "kind", "type",
    "sort", "way", "item", "object",
)  # fmt: skip
# Words that join a phrase. Kept out on purpose: "a", "as", "so", "up", "it", "at"
# (WHEREAS, WHATSOEVER, WHEREUPON, HOWITZER).
GLUE = (
    "you", "your", "we", "they", "people", "one", "to", "for", "of", "in", "on", "with",
    "that", "the", "and", "is", "are", "can", "used",
)  # fmt: skip
# Common concrete nouns and adjectives that start glued phrases. Short words that
# begin many unrelated words (sea: SEASON, car: CARBONATION, red: REDEMPTION) are
# deliberately absent.
CONTENT = frozenset(
    """
    air water fire earth sun sky land sound light heat body mind heart food blood hand
    head eye face tree plant animal road house home city money time space star moon rain
    snow ice wind storm rock stone metal wood paper music word color colour power energy
    game sport ball team work school book life death night day river forest farm garden
    flower fruit bread milk glass ship boat train family crowd voice song dance dream
    sleep love child friend black white green blue yellow big small hot cold high low
    good bad fast slow dark soft hard sweet salt pepper butter roll iron gold steel
    """.split()  # noqa: SIM905 (a word list reads better as text)
)
JOINERS = ("and", "of")  # BLACKANDWHITE, HEARTOFGOLD
# Abstract noun endings; "movement" in AIRMOVEMENT. -ing is left out (HEARTBREAKING).
ABSTRACT = ("ment", "tion", "sion", "ness", "ity", "ance", "ence")
# Heads that end like an abstract noun but form real compounds (EYEWITNESS,
# SEASICKNESS, AIRWORTHINESS), or stems that are derived adjectives (HOMELESSNESS).
COMPOUND_HEADS = ("witness", "business", "harness", "wilderness", "sickness", "worthiness")
DERIVED_STEMS = ("less", "ful", "ish", "some", "like", "ous", "able", "ible", "ive", "ed")


def _pronoun_inside(word: str) -> str | None:
    for pronoun in PRONOUNS:
        for match in re.finditer(pronoun, word):
            before, after = word[: match.start()], word[match.end() :]
            # LAYOUT(S), PAYOUT, TRYOUTS: "you" followed by t is the -yout ending.
            if len(before) >= 3 and len(after) >= 2 and not after.startswith("t"):
                return f"{before} + {pronoun} + {after}"
    return None


def _opener_phrase(word: str) -> str | None:
    for opener in OPENERS:
        if not word.startswith(opener):
            continue
        rest = word[len(opener) :]
        for glue in GLUE:
            # At least three letters after the glue: WHEREWITHAL, WHEREFORE stay legal.
            if rest.startswith(glue) and len(rest) - len(glue) >= 3:
                return f"{opener} + {glue} + {rest[len(glue) :]}"
    return None


def _conjoined(word: str) -> str | None:
    # A listed noun on one side: COMMANDMENT, ISLANDERS, HANDOFFS stay legal.
    for joiner in JOINERS:
        for match in re.finditer(joiner, word):
            left, right = word[: match.start()], word[match.end() :]
            if len(left) >= 3 and len(right) >= 3 and (left in CONTENT or right in CONTENT):
                return f"{left} + {joiner} + {right}"
    return None


def _noun_plus_abstract(word: str) -> str | None:
    for size in range(3, len(word) - 5):
        prefix, head = word[:size], word[size:]
        if prefix not in CONTENT or head in COMPOUND_HEADS:
            continue
        suffix = next((s for s in ABSTRACT if head.endswith(s)), None)
        if suffix is None:
            continue
        stem = head[: -len(suffix)]
        # CARNATION (na-tion), STARVATION (va-tion): the rest is no word of its own.
        if len(stem) >= 3 and not stem.endswith(DERIVED_STEMS):
            return f"{prefix} + {head}"
    return None


def phrase_problem(clue: str) -> str | None:
    """Why ``clue`` reads as a compressed phrase rather than one word, or None."""
    if re.search(r"[a-z][A-Z]", clue):
        return "camel-case phrase"
    word = re.sub(r"[-']", "", clue).casefold()
    for rule, label in (
        (_pronoun_inside, "sentence fragment"),
        (_opener_phrase, "definition phrase"),
        (_conjoined, "joined phrase"),
        (_noun_plus_abstract, "glued noun phrase"),
    ):
        parts = rule(word)
        if parts is not None:
            return f"{label} ({parts})"
    return None
