"""Shared colour-blind ranking format for clues.

The operative ranks board words for its clue, and the spymaster asks the SAME
question (same system prompt, same schema) to predict what its operative will do.
Keeping one format for both is what makes the spymaster's simulation meaningful.

Words are sorted into two coarse buckets instead of 0-100 scores (qwen3 produced
near-constant high numbers for those). Unlisted words are treated as unconnected.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Fit(StrEnum):
    STRONG = "strong"
    POSSIBLE = "possible"
    WEAK = "weak"  # only for a forced guess that the model did not list


# Small caps: qwen3 pads lists up to the cap, and padding blurs ambiguity. Strong keeps
# one slot beyond a 3-word clue so a competing fourth word stays visible.
MAX_STRONG = 4
MAX_POSSIBLE = 3

RANKING_SYSTEM = (
    "You are a Codenames operative. You cannot see card colours. Treat every clue as a "
    "separate question: re-read the board words for it and never copy another clue's lists. "
    "strong: most players would connect the word to the clue immediately. possible: a real "
    "but less direct connection. Leave out words that are a stretch or only share letters "
    "or sound with the clue. Strongest first; lists may be empty; use strong honestly."
)


@dataclass(frozen=True)
class RankedWord:
    word: str
    fit: Fit
    clue: str | None = None


def ranking_schema(clues: Sequence[str], words: Sequence[str]) -> dict[str, Any]:
    def bucket(cap: int) -> dict[str, Any]:
        return {
            "type": "array",
            "items": {"type": "string", "enum": list(words)},
            "maxItems": min(cap, len(words)),
        }

    return {
        "type": "object",
        "properties": {
            "rankings": {
                "type": "array",
                "minItems": len(clues),
                "maxItems": len(clues),
                "items": {
                    "type": "object",
                    "properties": {
                        "clue": {"type": "string", "enum": list(clues)},
                        "strong": bucket(MAX_STRONG),
                        "possible": bucket(MAX_POSSIBLE),
                    },
                    "required": ["clue", "strong", "possible"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rankings"],
        "additionalProperties": False,
    }


def ranking_user(words: Sequence[str], clues: Sequence[str]) -> str:
    quoted = ", ".join(f'"{clue}"' for clue in clues)
    return f"Board words: {', '.join(words)}\nClues: [{quoted}]"


def parse_rankings(
    reply: Mapping[str, Any], clues: Sequence[str], words: Sequence[str]
) -> dict[str, tuple[RankedWord, ...]]:
    """Map canonical clue -> strong words then possible words, distinct and on board.

    Unknown words, unknown or repeated clues, and malformed entries are dropped rather
    than retried, so a partly malformed reply still yields the valid part.
    """
    allowed = {word.casefold(): word for word in words}
    known = {clue.casefold(): clue for clue in clues}
    raw = reply.get("rankings")
    result: dict[str, tuple[RankedWord, ...]] = {}
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, Mapping):
            continue
        clue = known.get(str(entry.get("clue", "")).strip().casefold())
        if clue is None or clue in result:
            continue
        seen: set[str] = set()
        ranked: list[RankedWord] = []
        for fit in (Fit.STRONG, Fit.POSSIBLE):
            bucket = entry.get(fit.value)
            for item in bucket if isinstance(bucket, list) else []:
                word = allowed.get(str(item).strip().casefold())
                if word is not None and word not in seen:
                    seen.add(word)
                    ranked.append(RankedWord(word, fit, clue))
        result[clue] = tuple(ranked)
    return result


def format_ranking(
    ranking: Iterable[RankedWord],
    labels: Mapping[str, str] | None = None,
    *,
    show_clue: bool = False,
) -> str:
    parts = []
    for item in ranking:
        label = f"[{labels[item.word]}]" if labels and item.word in labels else ""
        clue = f"@{item.clue}" if show_clue and item.clue else ""
        parts.append(f"{item.word}{label}:{item.fit.value}{clue}")
    return ", ".join(parts) or "-"


RANKER_TEMPERATURE = 0.0
SINGLE_NUM_PREDICT = 240


def canonical_clue(clue: str) -> str:
    """One spelling everywhere: the site may display the clue in another case than
    typed, and qwen3 ranks "wave" and "WAVE" differently (measured)."""
    return " ".join(clue.split()).upper()


def single_clue_request(clue: str, words: Sequence[str]) -> dict[str, Any]:
    """The exact request the operative sends for its clue, reused verbatim by the
    spymaster to verify a finalist clue. Identical requests at temperature 0 gave
    identical rankings in every probe, so the verified ranking is the operative's."""
    clue = canonical_clue(clue)
    schema = ranking_schema([clue], words)
    # Generated after the lists, so it cannot change them; used only for a forced guess.
    schema["properties"]["best_guess"] = {"type": "string", "enum": list(words)}
    schema["required"].append("best_guess")
    return {
        "system": RANKING_SYSTEM,
        "user": ranking_user(words, [clue]),
        "schema": schema,
        "num_predict": SINGLE_NUM_PREDICT,
        "temperature": RANKER_TEMPERATURE,
    }


def parse_single(
    reply: Mapping[str, Any], clue: str, words: Sequence[str]
) -> tuple[RankedWord, ...]:
    """Ranking for one clue; a forced best guess stands in when nothing is listed."""
    clue = canonical_clue(clue)
    ranking = parse_rankings(reply, [clue], words).get(clue, ())
    best = reply.get("best_guess")
    if not ranking and isinstance(best, str) and best in words:
        ranking = (RankedWord(best, Fit.WEAK, clue),)
    return ranking
