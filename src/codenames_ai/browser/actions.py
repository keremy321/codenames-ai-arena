import logging
from dataclasses import dataclass

from playwright.async_api import Locator, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from codenames_ai.domain.enums import CardColor, GamePhase, Role
from codenames_ai.domain.models import Card, PlayerAssignment

from . import selectors as s
from .errors import (
    AlreadyRevealedError,
    AmbiguousCardError,
    BrowserIntegrationError,
    CardNotFoundError,
    GameStateTimeoutError,
    GuessConfirmationNotFoundError,
    WrongPhaseError,
)
from .identity import detect_assignment, dismiss_guide
from .reader import GameReader, normalize_word
from .waits import wait_snapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuessResult:
    card: Card
    phase: GamePhase
    turn_continues: bool


class GameActions:
    def __init__(
        self, page: Page, assignment: PlayerAssignment, *, timeout_ms: float = 10_000
    ) -> None:
        self.page = page
        self.assignment = assignment
        self.timeout_ms = timeout_ms
        self.reader = GameReader(page)
        self.trusted_join = False

    async def _guard(self, role: Role) -> GamePhase:
        if self.assignment.role != role:
            raise WrongPhaseError(f"{role} action requested by {self.assignment.role}")
        if not self.trusted_join:
            await detect_assignment(self.page, self.assignment)
        phase = await self.reader.phase()
        expected = GamePhase(f"{self.assignment.team}_{role}")
        if phase != expected:
            raise WrongPhaseError(f"Expected {expected}, found {phase}")
        return phase

    async def find_card(self, word: str) -> tuple[Locator, Card]:
        target = normalize_word(word).casefold()
        state = await self.reader.read_operative_state(self.assignment.team)
        matches = [card for card in state.cards if card.word.casefold() == target]
        if not matches:
            raise CardNotFoundError(f"No board card matches {word!r}")
        if len(matches) != 1:
            raise AmbiguousCardError(f"{len(matches)} board cards match {word!r}")
        card = matches[0]
        roots = self.page.locator(s.LIVE_CARDS).filter(visible=True)
        if not await roots.count():
            roots = self.page.locator(s.CARDS).filter(
                has_not=self.page.locator("article"), visible=True
            )
        return roots.nth(card.index), card

    async def select_card(self, word: str) -> None:
        await self._guard(Role.OPERATIVE)
        locator, card = await self.find_card(word)
        if card.revealed:
            raise AlreadyRevealedError(f"{card.word} is already revealed")
        if not card.selected:
            logger.info("[%s] considering %s", self.assignment.nickname, card.word)
            try:
                await locator.click(timeout=self.timeout_ms)
            except PlaywrightTimeoutError as exc:
                raise GameStateTimeoutError(f"Could not select {card.word}") from exc
        await wait_snapshot(
            self.page,
            "state.cards[v.index]?.selected === true",
            values={"index": card.index},
            timeout_ms=self.timeout_ms,
            description=f"selection of {card.word}",
        )

    async def guess_card(self, word: str) -> GuessResult:
        await self.select_card(word)
        _, card = await self.find_card(word)
        before = GamePhase(f"{self.assignment.team}_{Role.OPERATIVE}")
        confirm = self.page.locator(s.CONFIRM_GUESS)
        if await confirm.count() > 1:
            raise GuessConfirmationNotFoundError("Ambiguous guess confirmation controls")
        try:
            await confirm.wait_for(state="visible", timeout=self.timeout_ms)
        except PlaywrightTimeoutError as exc:
            raise GuessConfirmationNotFoundError(
                f"No confirmation control for {card.word}"
            ) from exc
        if await confirm.count() != 1:
            raise GuessConfirmationNotFoundError("Ambiguous guess confirmation controls")
        current = await self.reader.read_operative_state(self.assignment.team)
        selected = [c for c in current.cards if c.selected]
        if (
            len(selected) != 1
            or selected[0].word != card.word
            or await self.reader.phase() != before
        ):
            raise BrowserIntegrationError("Selection or turn changed before confirmation")
        try:
            await confirm.click(timeout=self.timeout_ms)
        except PlaywrightTimeoutError as exc:
            raise GameStateTimeoutError(
                "Confirmation could not be clicked; do not retry blindly"
            ) from exc
        # Never retry confirmation after timeout: the server may have accepted it.
        await wait_snapshot(
            self.page,
            "state.cards[v.index]?.revealed === true",
            values={"index": card.index},
            timeout_ms=self.timeout_ms,
            description=f"public reveal of {card.word}",
        )
        reading = await self.reader.read(self.assignment.team, Role.OPERATIVE)
        revealed = next(c for c in reading.state.cards if c.index == card.index)
        if revealed.color in (None, CardColor.UNKNOWN):
            raise BrowserIntegrationError(
                "Revealed card color is unknown; refusing further guesses"
            )
        if revealed.color != CardColor(self.assignment.team.value):
            # Wrong-team/neutral/assassin reveals end this operative turn, but
            # the site updates the cover before it updates the active panel.
            await self._wait_next_turn(Role.OPERATIVE)
            reading = await self.reader.read(self.assignment.team, Role.OPERATIVE)
        logger.info(
            "[%s] confirmed %s; revealed=%s",
            self.assignment.nickname,
            revealed.word,
            revealed.color,
        )
        return GuessResult(revealed, reading.phase, reading.phase == before)

    async def end_guessing(self) -> None:
        expected = GamePhase(f"{self.assignment.team}_{Role.OPERATIVE}")
        if await self.reader.phase() != expected:
            return
        try:
            await self._guard(Role.OPERATIVE)
        except WrongPhaseError:
            if await self.reader.phase() != expected:
                return
            raise
        if not await self.page.locator(s.CLUE_NUMBER).is_visible():
            if await self.reader.phase() != expected:
                return
            raise BrowserIntegrationError("No operative clue number; refusing to end guessing")
        try:
            await self._click_clue_action()
        except GameStateTimeoutError:
            if await self.reader.phase() != expected:
                return
            raise
        if await self.reader.phase() != expected:
            return
        trigger = self.page.locator(s.CLUE_ACTION).filter(visible=True)
        if await trigger.get_attribute("aria-expanded") == "true":
            dialog_id = await trigger.get_attribute("aria-controls")
            dialog = self.page.get_by_role("dialog").filter(
                has=self.page.get_by_role("button", name="End Guessing", exact=True)
            )
            if (
                not dialog_id
                or await dialog.count() != 1
                or await dialog.get_attribute("id") != dialog_id
            ):
                raise BrowserIntegrationError(
                    "End-guessing confirmation does not match its trigger"
                )
            await dialog.get_by_role("button", name="End Guessing", exact=True).click(
                timeout=self.timeout_ms
            )
        await self._wait_next_turn(Role.OPERATIVE)

    async def can_end_guessing(self) -> bool:
        expected = GamePhase(f"{self.assignment.team}_{Role.OPERATIVE}")
        if await self.reader.phase() != expected:
            return False
        action = self.page.locator(s.CLUE_ACTION).filter(visible=True)
        return (
            await self.page.locator(s.CLUE_NUMBER).is_visible()
            and await action.count() == 1
            and await action.is_visible()
            and await action.is_enabled()
        )

    async def submit_clue(self, clue: str, number: int) -> None:
        clue = clue.strip()
        if not clue or len(clue) > 32 or len(clue.split()) != 1:
            raise ValueError("Clue must be one nonempty word, at most 32 characters")
        if type(number) is not int or not 0 <= number <= 9:
            raise ValueError("Clue count must be an integer from 0 to 9")
        await self._guard(Role.SPYMASTER)
        field = self.page.locator(s.CLUE_INPUT).filter(visible=True)
        try:
            await field.first.wait_for(state="visible", timeout=self.timeout_ms)
        except PlaywrightTimeoutError:
            pass
        visible_inputs = await field.count()
        if visible_inputs != 1:
            instructions = await self.page.locator(s.INSTRUCTION).all_text_contents()
            raise BrowserIntegrationError(
                "No unique visible spymaster clue input; "
                f"total clue inputs={await self.page.locator(s.CLUE_INPUT).count()}, "
                f"visible clue inputs={visible_inputs}, "
                f"current instruction={instructions[0].strip() if instructions else ''!r}"
            )
        await field.fill(clue)
        count = self.page.locator(s.CLUE_COUNT_BUTTON).filter(visible=True)
        await count.first.wait_for(state="visible", timeout=self.timeout_ms)
        if await count.count() != 1:
            raise BrowserIntegrationError("No unique visible clue count selector")
        await count.click(timeout=self.timeout_ms)
        dialog_id = await count.get_attribute("aria-controls")
        if not dialog_id:
            raise BrowserIntegrationError("Clue count button has no aria-controls relationship")
        dialog = self.page.get_by_role("dialog").filter(
            has=self.page.get_by_role("button", name=str(number), exact=True)
        )
        await dialog.wait_for(timeout=self.timeout_ms)
        if await dialog.count() != 1 or await dialog.get_attribute("id") != dialog_id:
            raise BrowserIntegrationError("Clue number dialog does not match its trigger")
        await dialog.get_by_role("button", name=str(number), exact=True).click()
        logger.info("[%s] submitting clue: %s %d", self.assignment.nickname, clue, number)
        await self._click_clue_action()
        await self._wait_next_turn(Role.SPYMASTER)
        reading = await self.reader.read(self.assignment.team, Role.SPYMASTER)
        if (
            reading.state.clue is None
            or reading.state.clue.word.casefold() != clue.casefold()
            or reading.state.clue.number != number
        ):
            raise BrowserIntegrationError("Submitted clue does not match the resulting public clue")

    async def _click_clue_action(self) -> None:
        await dismiss_guide(self.page)
        action = self.page.locator(s.CLUE_ACTION).filter(visible=True)
        try:
            await action.first.wait_for(state="visible", timeout=self.timeout_ms)
        except PlaywrightTimeoutError as exc:
            raise GameStateTimeoutError("Clue action could not be clicked") from exc
        if await action.count() != 1:
            raise BrowserIntegrationError("Clue area does not have one visible action button")
        try:
            await action.click(timeout=self.timeout_ms)
        except PlaywrightTimeoutError as exc:
            raise GameStateTimeoutError("Clue action could not be clicked") from exc

    async def _wait_next_turn(self, role: Role) -> None:
        panel = s.ROLE_PANELS[(self.assignment.team, role)]
        await wait_snapshot(
            self.page,
            "state.gameOver || (state.activePanels.length === 1 && !state.activePanels.includes(v.panel))",
            values={"panel": panel},
            timeout_ms=self.timeout_ms,
            description="next game phase",
        )


async def select_card(page: Page, word: str) -> None:
    await GameActions(page, await detect_assignment(page)).select_card(word)


async def guess_card(page: Page, word: str) -> GuessResult:
    return await GameActions(page, await detect_assignment(page)).guess_card(word)


async def end_guessing(page: Page) -> None:
    await GameActions(page, await detect_assignment(page)).end_guessing()


async def submit_clue(page: Page, clue: str, number: int) -> None:
    await GameActions(page, await detect_assignment(page)).submit_clue(clue, number)
