from pathlib import Path

import pytest
from playwright.async_api import Page
from pydantic import ValidationError

from codenames_ai.agents.operative import OperativeAgent
from codenames_ai.browser import selectors as s
from codenames_ai.browser.actions import GameActions
from codenames_ai.browser.errors import (
    AlreadyRevealedError,
    AmbiguousCardError,
    BrowserIntegrationError,
    CardNotFoundError,
    GameStateTimeoutError,
    GuessConfirmationNotFoundError,
    PlayerAssignmentError,
    WrongPhaseError,
)
from codenames_ai.browser.identity import detect_assignment
from codenames_ai.browser.probe import sanitize_snippet, save_snapshots
from codenames_ai.browser.reader import GameReader, parse_cover_color
from codenames_ai.browser.waits import wait_snapshot
from codenames_ai.domain.enums import CardColor, GamePhase, Role, Team
from codenames_ai.domain.models import PlayerAssignment, PublicGameState
from codenames_ai.llm.ollama import OllamaClient

FIXTURES = Path(__file__).parent / "fixtures" / "live"
CSS = """<style>
.absolute{position:absolute}.relative{position:relative}.top-0{top:0}.left-0{left:0}
.inset-0{inset:0}.block{display:block}[aria-hidden=true]{visibility:hidden}
button{min-width:30px;min-height:25px} #appMount{position:relative;height:700px}
[data-match-slot] > main{min-height:1px}
</style>"""


def snippet(name: str) -> str:
    return (FIXTURES / f"{name}.html").read_text(encoding="utf-8")


async def board(
    page: Page, *, role: Role = Role.OPERATIVE, card: str = "card-selected"
) -> GameActions:
    card_html = snippet(card).replace("scale-[1.02]", "scale-100")
    slot = "blueOp" if role == Role.OPERATIVE else "blueSpy"
    instruction = (
        "Tap on cards you think match the clue"
        if role == Role.OPERATIVE
        else "Give your operatives a clue"
    )
    clue_html = snippet("clue-operative" if role == Role.OPERATIVE else "clue-spymaster")
    # Control behavior below is a simulator; element structure comes from live captures.
    await page.set_content(
        CSS
        + f'''<main id="appMount">
      <section data-match-slot="instruction">{instruction}</section>
      <section data-match-slot="grid"></section>
      <div data-match-slot="{slot}"><main class="activeRoleShadow"><button aria-haspopup="dialog">FixturePlayer</button></main></div>
      <div data-match-slot="redSpy"><main></main></div>
      <button data-testid="settings-button" onclick="document.querySelector('#local-settings').hidden=false">Settings</button>
      <div id="local-settings" role="dialog" hidden><input id="settings-nickname-input" value="FixturePlayer"></div>
      {clue_html}{card_html}
      <div id="distractor">DRONE</div>
    </main><script>
      document.addEventListener('keydown',e=>{{if(e.key==='Escape') document.querySelector('#local-settings').hidden=true;}});
      window.cardClicks=0; window.confirmClicks=0;
    </script>'''
    )
    return GameActions(
        page, PlayerAssignment(team=Team.BLUE, role=role, nickname="FixturePlayer"), timeout_ms=500
    )


@pytest.mark.parametrize(
    "kind,color",
    [
        ("blue", CardColor.BLUE),
        ("red", CardColor.RED),
        ("neutral", CardColor.NEUTRAL),
        ("assassin", CardColor.ASSASSIN),
    ],
)
async def test_captured_reveals(live_page: Page, kind: str, color: CardColor) -> None:
    await board(live_page, card=f"card-revealed-{kind}")
    state = await GameReader(live_page).read_operative_state(Team.BLUE)
    assert len(state.cards) == 1  # nested word article and cover are not additional cards
    assert state.cards[0].revealed
    assert state.cards[0].color == color


async def test_assassin_hidden_from_operative(
    live_page: Page, caplog: pytest.LogCaptureFixture
) -> None:
    await board(live_page, card="card-assassin")
    reader = GameReader(live_page)
    public = await reader.read_operative_state(Team.BLUE)
    assert public.cards[0].color is None and not public.cards[0].revealed
    async with OllamaClient() as llm:
        prompt = OperativeAgent(Team.BLUE, llm).prompt_state(public)
    for secret in [
        "assassin",
        "--black",
        "style",
        "<article",
        "cardBg",
        "target_cards",
        "reasoning",
    ]:
        assert secret not in prompt
        assert secret not in caplog.text
    assert (await reader.read_spymaster_state(Team.BLUE)).cards[0].color == CardColor.ASSASSIN
    with pytest.raises(ValidationError):
        PublicGameState(team=Team.BLUE, cards=[], reasoning="secret")


async def test_captured_selection_ignores_avatar_name(live_page: Page) -> None:
    await live_page.set_content(CSS + '<main id="appMount">' + snippet("card-selected") + "</main>")
    state = await GameReader(live_page).read_operative_state(Team.BLUE)
    assert state.cards[0].word == "DRONE"
    assert state.cards[0].selected
    assert state.cards[0].color is None


async def test_deck_cover_is_not_a_revealed_card(live_page: Page) -> None:
    await board(live_page, card="card-revealed-blue")
    await live_page.locator(s.COVERS).evaluate("el=>el.style.translate='30px 200px'")
    state = await GameReader(live_page).read_operative_state(Team.BLUE)
    assert not state.cards[0].revealed and state.cards[0].color is None


async def test_ambiguous_covers_fail_closed(live_page: Page) -> None:
    await board(live_page, card="card-revealed-blue")
    await live_page.locator(s.COVERS).evaluate("el=>el.after(el.cloneNode(true))")
    state = await GameReader(live_page).read_operative_state(Team.BLUE)
    assert not state.cards[0].revealed and state.cards[0].color is None


async def test_exact_card_lookup_and_ambiguity(live_page: Page) -> None:
    actions = await board(live_page)
    _, card = await actions.find_card("  drone ")
    assert card.word == "DRONE"
    with pytest.raises(CardNotFoundError):
        await actions.find_card("DRON")
    await live_page.locator(s.LIVE_CARDS).evaluate("el=>el.after(el.cloneNode(true))")
    with pytest.raises(AmbiguousCardError):
        await actions.find_card("DRONE")


async def wire_guess(page: Page, *, confirm: bool = True, reveal: bool = True) -> None:
    await page.evaluate(
        """({confirmation, revealed, withConfirm, withReveal})=>{
        const card=document.querySelector('article[style*="--CardColor:"]');
        card.onclick=()=>{
            window.cardClicks++;
            card.querySelector('section').className='absolute inset-0 scale-[1.02]';
            if(withConfirm){
                document.querySelector('#appMount').insertAdjacentHTML('beforeend',confirmation);
                const button=document.querySelector('button:has(img)');
                button.onclick=()=>{
                    window.confirmClicks++;
                    if(withReveal){card.insertAdjacentHTML('afterend',revealed);card.remove();button.remove();}
                };
            }
        };
    }""",
        {
            "confirmation": snippet("guess-confirmation"),
            "revealed": snippet("card-revealed-blue"),
            "withConfirm": confirm,
            "withReveal": reveal,
        },
    )


async def test_select_confirm_and_duplicate_prevention(live_page: Page) -> None:
    actions = await board(live_page)
    await wire_guess(live_page)
    result = await actions.guess_card("DRONE")
    assert result.card.color == CardColor.BLUE and result.turn_continues
    assert await live_page.evaluate("[cardClicks,confirmClicks]") == [1, 1]
    with pytest.raises(AlreadyRevealedError):
        await actions.guess_card("DRONE")
    assert await live_page.evaluate("confirmClicks") == 1


async def test_missing_confirmation(live_page: Page) -> None:
    actions = await board(live_page)
    await wire_guess(live_page, confirm=False)
    with pytest.raises(GuessConfirmationNotFoundError):
        await actions.guess_card("DRONE")
    assert await live_page.evaluate("confirmClicks") == 0


async def test_reveal_timeout_does_not_retry(live_page: Page) -> None:
    actions = await board(live_page)
    await wire_guess(live_page, reveal=False)
    with pytest.raises(GameStateTimeoutError):
        await actions.guess_card("DRONE")
    assert await live_page.evaluate("confirmClicks") == 1


async def test_wrong_phase_never_clicks_card(live_page: Page) -> None:
    actions = await board(live_page)
    await wire_guess(live_page)
    await live_page.locator('[data-match-slot="instruction"]').evaluate(
        "el=>el.textContent='Give your operatives a clue'"
    )
    with pytest.raises(WrongPhaseError):
        await actions.select_card("DRONE")
    assert await live_page.evaluate("cardClicks") == 0


async def test_identity_mismatch(live_page: Page) -> None:
    actions = await board(live_page)
    assert await detect_assignment(live_page) == actions.assignment
    with pytest.raises(PlayerAssignmentError):
        await detect_assignment(
            live_page,
            PlayerAssignment(team=Team.RED, role=Role.OPERATIVE, nickname="FixturePlayer"),
        )


async def test_arena_joined_action_does_not_need_settings(live_page: Page) -> None:
    actions = await board(live_page)
    await live_page.locator(s.SETTINGS_BUTTON).evaluate("el=>el.remove()")
    await live_page.locator("#local-settings").evaluate("el=>el.remove()")
    actions.trusted_join = True
    assert await actions._guard(Role.OPERATIVE) == GamePhase.BLUE_OPERATIVE


async def test_end_guessing(live_page: Page) -> None:
    actions = await board(live_page)
    await live_page.locator(s.CLUE_ACTION).evaluate("""el=>el.onclick=()=>{
        document.querySelector('.activeRoleShadow').classList.remove('activeRoleShadow');
        document.querySelector('[data-match-slot=redSpy] > main').classList.add('activeRoleShadow');
        document.querySelector('[data-match-slot=instruction]').textContent='Wait for RedSpy to give you a clue';
    }""")
    await actions.end_guessing()
    assert await actions.reader.phase() == GamePhase.RED_SPYMASTER


async def test_end_guessing_is_noop_after_phase_change(live_page: Page) -> None:
    actions = await board(live_page)
    await live_page.locator(s.CLUE_ACTION).evaluate("el=>el.onclick=()=>{window.endClicked=true}")
    await live_page.locator(".activeRoleShadow").evaluate(
        "el=>{el.classList.remove('activeRoleShadow');document.querySelector('[data-match-slot=redSpy] > main').classList.add('activeRoleShadow')}"
    )
    await live_page.locator(s.INSTRUCTION).evaluate(
        "el=>el.textContent='Wait for RedSpy to give you a clue'"
    )
    assert not await actions.can_end_guessing()
    await actions.end_guessing()
    assert await live_page.evaluate("window.endClicked || false") is False


async def test_end_guessing_ignores_click_timeout_after_phase_change(
    live_page: Page, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions = await board(live_page)

    async def phase_changes_during_click() -> None:
        await live_page.locator(".activeRoleShadow").evaluate(
            "el=>{el.classList.remove('activeRoleShadow');document.querySelector('[data-match-slot=redSpy] > main').classList.add('activeRoleShadow')}"
        )
        await live_page.locator(s.INSTRUCTION).evaluate(
            "el=>el.textContent='Wait for RedSpy to give you a clue'"
        )
        raise GameStateTimeoutError("Clue action could not be clicked")

    monkeypatch.setattr(actions, "_click_clue_action", phase_changes_during_click)
    await actions.end_guessing()
    assert await actions.reader.phase() == GamePhase.RED_SPYMASTER


async def test_end_guessing_confirmation_dialog(live_page: Page) -> None:
    actions = await board(live_page)
    confirmation = snippet("end-guessing-confirmation")
    await live_page.locator(s.CLUE_ACTION).evaluate(
        """(el, html)=>el.onclick=()=>{
            el.setAttribute('aria-controls','end-confirm');
            el.setAttribute('aria-expanded','true');
            document.body.insertAdjacentHTML('beforeend',html);
                const dialog=document.querySelector('[role=dialog]:not(#local-settings)');
                dialog.id='end-confirm';
                dialog.querySelector('button.absolute').style.display='none';
                dialog.querySelector('button:nth-last-of-type(2)').onclick=()=>{
                window.endConfirmed=true;
                document.querySelector('.activeRoleShadow').classList.remove('activeRoleShadow');
                document.querySelector('[data-match-slot=redSpy] > main').classList.add('activeRoleShadow');
                document.querySelector('[data-match-slot=instruction]').textContent='Wait for RedSpy to give you a clue';
                dialog.remove();
            };
        }""",
        confirmation,
    )
    await actions.end_guessing()
    assert await live_page.evaluate("endConfirmed") is True


@pytest.mark.parametrize("hidden_duplicate", [False, True])
async def test_submit_clue_with_dynamic_dialog_id(live_page: Page, hidden_duplicate: bool) -> None:
    actions = await board(live_page, role=Role.SPYMASTER)
    if hidden_duplicate:
        await live_page.locator(s.CLUE_AREA).evaluate(
            "el=>el.insertAdjacentHTML('afterbegin', "
            '\'<input hidden placeholder="Your clue" value="WRONG">\')'
        )
    await live_page.evaluate(
        """({numbers,operative})=>{
        const count=document.querySelector('[data-match-slot=clue] button:not([data-slot])');
        count.setAttribute('aria-controls','dynamic-42');
        count.onclick=()=>{
            document.body.insertAdjacentHTML('beforeend',numbers);
            const dialog=document.querySelector('[role=dialog]:not(#local-settings)');
            dialog.id='dynamic-42';
            dialog.querySelectorAll('button').forEach(b=>b.onclick=()=>{count.textContent=b.textContent;dialog.remove();});
        };
        document.querySelector('[data-match-slot=clue] button[data-slot]').onclick=()=>{
            const input=document.querySelector('[data-match-slot=clue] input[placeholder="Your clue"]:not([hidden])');
            window.submitted=[input.value,count.textContent];
            document.querySelector('[data-match-slot=clue]').outerHTML=operative;
            const panel=document.querySelector('[data-match-slot=blueSpy]');panel.dataset.matchSlot='blueOp';
            document.querySelector('[data-match-slot=instruction]').textContent='Tap on cards you think match the clue';
        };
    }""",
        {"numbers": snippet("clue-numbers"), "operative": snippet("clue-operative")},
    )
    await actions.submit_clue("ORBIT", 2)
    assert await live_page.evaluate("submitted") == ["ORBIT", "2"]


async def test_multiple_visible_clue_inputs_report_diagnostics(live_page: Page) -> None:
    actions = await board(live_page, role=Role.SPYMASTER)
    await live_page.locator(s.CLUE_AREA).evaluate(
        "el=>el.insertAdjacentHTML('afterbegin', '<input placeholder=\"Your clue\">')"
    )
    with pytest.raises(
        BrowserIntegrationError,
        match="total clue inputs=2, visible clue inputs=2, current instruction='Give your operatives a clue'",
    ):
        await actions.submit_clue("ORBIT", 2)


@pytest.mark.parametrize(
    "word,count", [("", 2), ("two words", 2), ("X" * 33, 2), ("OK", -1), ("OK", 10), ("OK", True)]
)
async def test_invalid_clue_before_interaction(live_page: Page, word: str, count: int) -> None:
    actions = await board(live_page, role=Role.SPYMASTER)
    with pytest.raises(ValueError):
        await actions.submit_clue(word, count)


async def test_state_wait_timeout(live_page: Page) -> None:
    await board(live_page)
    with pytest.raises(GameStateTimeoutError, match="test condition"):
        await wait_snapshot(
            live_page, "state.gameOver", timeout_ms=50, description="test condition"
        )


async def test_debug_snapshot_redaction(live_page: Page, tmp_path: Path) -> None:
    await board(live_page)
    html = '<article id="secret-id" data-session="TOKEN" onclick="steal()"><script>secret</script><a href="https://codenames.game/room/PRIVATE">https://codenames.game/room/PRIVATE</a><input value="secret"><p>Alice</p></article>'
    clean = await sanitize_snippet(live_page, html, ["Alice"])
    for secret in ["TOKEN", "secret", "PRIVATE", "Alice", "onclick", "https://"]:
        assert secret not in clean
    paths = await save_snapshots(live_page, tmp_path)
    assert paths
    assert all("radix-" not in p.read_text() for p in paths)


def test_unknown_cover() -> None:
    assert parse_cover_color("var(--cover-card-bg-unverified-url)") == CardColor.UNKNOWN
