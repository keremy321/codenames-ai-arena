import logging
import re
from dataclasses import dataclass
from typing import Any

from playwright.async_api import Page

from codenames_ai.domain.enums import CardColor, GamePhase, Role, Team
from codenames_ai.domain.models import Card, Clue, PublicGameState, SpymasterGameState

from . import selectors as s
from .dom import DOM_ARGS, SNAPSHOT_JS

logger = logging.getLogger(__name__)


class DOMParseError(ValueError):
    """The board cannot be interpreted without guessing."""


def normalize_word(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if not lines:
        raise DOMParseError("Card word is empty")
    unique = list(dict.fromkeys(lines))
    if len(unique) == 1:
        return unique[0]
    raise DOMParseError(f"Ambiguous card word: {text[:120]!r}")


def parse_card_color(style: str, *, word: str = "") -> CardColor:
    # Inspect the assignment, not other unrelated variables anywhere in style.
    match = re.search(r"(?:^|;)\s*--CardColor\s*:\s*([^;]+)", style)
    value = match.group(1).strip() if match else ""
    known = {
        "var(--blue-cardBg)": CardColor.BLUE,
        "var(--red-cardBg)": CardColor.RED,
        "var(--neutral-cardBg)": CardColor.NEUTRAL,
        "var(--black-cardBg)": CardColor.ASSASSIN,
    }
    compact = re.sub(r"\s+", "", value)
    if compact in known:
        return known[compact]
    logger.warning("Unable to determine card color: word=%r style=%r", word, style[:240])
    return CardColor.UNKNOWN


def parse_cover_color(style: str) -> CardColor:
    """Only public, geometrically matched cover styles may enter this parser."""
    match = re.fullmatch(r"var\(--cover-card-bg-(blue|red|neutral|black)-url\)", style.strip())
    if not match:
        logger.warning("Unknown public cover color: %r", style[:120])
        return CardColor.UNKNOWN
    return parse_card_color(f"--CardColor:var(--{match.group(1)}-cardBg)")


def parse_dom_card(raw: dict[str, Any], *, include_hidden: bool) -> Card:
    if raw["ambiguousCover"]:
        logger.warning("Ambiguous cover association at card index=%d", raw["index"])
    card = parse_card(raw["index"], raw["text"], raw["style"], include_hidden=include_hidden)
    return Card(
        index=card.index,
        word=card.word,
        revealed=raw["revealed"],
        selected=raw["selected"],
        color=parse_cover_color(raw["coverStyle"]) if raw["revealed"] else card.color,
    )


def parse_card(
    index: int,
    text: str,
    style: str = "",
    *,
    revealed: bool = False,
    include_hidden: bool = False,
) -> Card:
    word = normalize_word(text)
    color = parse_card_color(style, word=word) if include_hidden or revealed else None
    return Card(index=index, word=word, color=color, revealed=revealed)


def parse_clue(words: list[str], number: str | None) -> Clue | None:
    candidates = list(dict.fromkeys(w.strip() for w in words if w.strip()))
    if not candidates and not number:
        return None  # Empty clue area before the first clue is normal.
    if not number:
        logger.warning("Clue number element not found or empty")
        return None
    number = number.strip()
    if len(candidates) != 1 or not re.fullmatch(r"[0-9]+", number):
        logger.warning("Unable to parse clue: words=%r number=%r", candidates[:3], number[:40])
        return None
    return Clue(word=candidates[0], number=int(number))


@dataclass(frozen=True)
class TurnInfo:
    team: Team | None
    role: Role | None


def detect_turn(instruction: str, active_panels: list[str]) -> TurnInfo:
    """Active panels indicate the turn, never the local player's identity."""
    instruction_role = None
    instruction = instruction.casefold()
    if (
        "give your operatives a clue" in instruction
        or "giving a clue" in instruction
        or "to give you a clue" in instruction
    ):
        instruction_role = Role.SPYMASTER
    elif "tap on cards you think match the clue" in instruction or "guessing" in instruction:
        instruction_role = Role.OPERATIVE
    matched = [
        identity for identity, selector in s.ROLE_PANELS.items() if selector in active_panels
    ]
    if len(matched) == 1:
        team, role = matched[0]
        if instruction_role is not None and instruction_role != role:
            logger.warning("Instruction and active role panel disagree")
            return TurnInfo(None, None)
        logger.debug("Detected active turn: %s_%s", team, role)
        return TurnInfo(team, role)
    logger.warning("Cannot determine active team: %d active panels", len(matched))
    return TurnInfo(None, instruction_role)


def detect_phase(raw: dict[str, Any]) -> GamePhase:
    if raw.get("gameOver"):
        return GamePhase.GAME_OVER
    if raw.get("lobby") and not raw["cards"]:
        return GamePhase.WAITING
    if not raw["cards"]:
        return GamePhase.UNKNOWN
    turn = detect_turn(raw["instruction"], raw["activePanels"])
    if turn.team is None or turn.role is None:
        return GamePhase.UNKNOWN
    instruction = raw["instruction"].casefold()
    if turn.role == Role.SPYMASTER:
        corroborated = raw.get("clueInput") or "clue" in instruction
    else:
        corroborated = bool(raw["clueNumber"] and raw["clueWords"])
    return GamePhase(f"{turn.team}_{turn.role}") if corroborated else GamePhase.UNKNOWN


@dataclass(frozen=True)
class BoardReading:
    state: PublicGameState | SpymasterGameState
    instruction: str
    turn: TurnInfo
    phase: GamePhase = GamePhase.UNKNOWN


class GameReader:
    def __init__(self, page: Page) -> None:
        self.page = page

    async def wait_for_board(self, *, timeout_ms: float = 300_000) -> None:
        await self.page.locator(f"{s.LIVE_CARDS}, {s.CARDS}").first.wait_for(
            state="visible", timeout=timeout_ms
        )

    async def _snapshot(self, *, include_hidden: bool) -> dict[str, Any]:
        return await self.page.evaluate(SNAPSHOT_JS, {**DOM_ARGS, "includeHidden": include_hidden})

    async def phase(self) -> GamePhase:
        return detect_phase(await self._snapshot(include_hidden=False))

    async def read(self, team: Team, role: Role) -> BoardReading:
        raw = await self._snapshot(include_hidden=role == Role.SPYMASTER)
        if not raw["cards"]:
            raise DOMParseError("No visible cards found in the game grid")
        cards = tuple(
            parse_dom_card(c, include_hidden=role == Role.SPYMASTER) for c in raw["cards"]
        )
        logger.info("Found %d cards", len(cards))
        if len(cards) != 25:
            logger.warning("Expected a standard 25-card board; found %d cards", len(cards))
        clue = parse_clue(raw["clueWords"], raw["clueNumber"])
        state = (
            SpymasterGameState(team=team, cards=cards, clue=clue)
            if role == Role.SPYMASTER
            else PublicGameState(team=team, cards=cards, clue=clue)
        )
        return BoardReading(
            state,
            raw["instruction"],
            detect_turn(raw["instruction"], raw["activePanels"]),
            detect_phase(raw),
        )

    async def read_spymaster_state(self, team: Team) -> SpymasterGameState:
        reading = await self.read(team, Role.SPYMASTER)
        assert isinstance(reading.state, SpymasterGameState)
        return reading.state

    async def read_operative_state(self, team: Team) -> PublicGameState:
        reading = await self.read(team, Role.OPERATIVE)
        assert isinstance(reading.state, PublicGameState)
        return reading.state
