import logging

import pytest
from playwright.async_api import Page

from codenames_ai.browser.errors import PlayerAssignmentError
from codenames_ai.browser.identity import detect_assignment, join_room
from codenames_ai.browser.selectors import ROLE_PANELS
from codenames_ai.domain.enums import Role, Team
from codenames_ai.domain.models import PlayerAssignment


async def lobby(page: Page, *, modal: bool = True, joined: str | None = None) -> None:
    panels = "".join(
        f'<section data-match-slot="{selector.split(chr(34))[1]}">'
        "<span>ROLE</span>"
        '<button class="joinTeam">JOIN TEAM</button>'
        + (
            f'<button aria-haspopup="dialog">{joined}</button>'
            if joined and selector == ROLE_PANELS[(Team.BLUE, Role.SPYMASTER)]
            else ""
        )
        + "</section>"
        for selector in ROLE_PANELS.values()
    )
    welcome = (
        '<form id="welcome"><input id="nickname" name="nickname" placeholder="Nickname">'
        '<button type="submit">Enter Game</button></form>'
        if modal
        else ""
    )
    await page.set_content(
        '<section data-match-slot="participants"></section>'
        + panels
        + welcome
        + '<button data-testid="settings-button" onclick="document.querySelector(\'#settings\').hidden=false">Settings</button>'
        + '<div id="settings" role="dialog" hidden><input id="settings-nickname-input"></div>'
        + """<script>
          window.joinedPanels=[];
          document.querySelector('#settings-nickname-input').value='BlueSpymasterAI';
          const form=document.querySelector('#welcome');
          const entry=document.querySelector('#nickname');
          if(entry) entry.addEventListener('keyup',e=>window.typedNickname=e.target.value);
          if(form) form.onsubmit=e=>{
            e.preventDefault();
            const name=window.nicknameOverride || window.typedNickname || 'Player1';
            const settings=document.querySelector('#settings-nickname-input');
            if(settings) settings.value=name;
            const spectator=document.createElement('span');
            spectator.textContent=name;
            document.querySelector('[data-match-slot=participants]').append(spectator);
            form.remove();
          };
          document.querySelectorAll('button.joinTeam').forEach(button=>button.onclick=()=>{
            const panel=button.parentElement;
            window.joinedPanels.push(panel.getAttribute('data-match-slot'));
            const member=document.createElement('button');
            member.setAttribute('aria-haspopup','dialog');
            member.textContent=document.querySelector('#settings-nickname-input')?.value || window.typedNickname || 'Player1';
            panel.append(member);
          });
          document.addEventListener('keydown',e=>{
            if(e.key==='Escape') document.querySelector('#settings').hidden=true;
          });
        </script>"""
    )


@pytest.mark.parametrize(
    "team,role,nickname,slot",
    [
        (Team.BLUE, Role.SPYMASTER, "BlueSpymasterAI", "blueSpy"),
        (Team.BLUE, Role.OPERATIVE, "BlueOperativeAI", "blueOp"),
        (Team.RED, Role.SPYMASTER, "RedSpymasterAI", "redSpy"),
        (Team.RED, Role.OPERATIVE, "RedOperativeAI", "redOp"),
    ],
)
async def test_joins_only_assigned_panel(
    live_page: Page, team: Team, role: Role, nickname: str, slot: str
) -> None:
    await lobby(live_page)
    assignment = PlayerAssignment(team=team, role=role, nickname=nickname)
    await join_room(live_page, assignment)
    assert await live_page.evaluate("joinedPanels") == [slot]
    actual = (await detect_assignment(live_page, assignment)).nickname
    assert (
        await live_page.locator(ROLE_PANELS[(team, role)])
        .get_by_role("button", name=actual, exact=True)
        .count()
        == 1
    )


async def test_already_entered_page_joins_role(live_page: Page) -> None:
    await lobby(live_page, modal=False)
    assignment = PlayerAssignment(team=Team.BLUE, role=Role.SPYMASTER, nickname="BlueSpymasterAI")
    await join_room(live_page, assignment)
    assert await live_page.evaluate("joinedPanels") == ["blueSpy"]


async def test_waits_for_join_button_to_render(live_page: Page) -> None:
    await lobby(live_page)
    await (
        live_page.locator(ROLE_PANELS[(Team.BLUE, Role.SPYMASTER)])
        .locator("button.joinTeam")
        .evaluate("el=>{el.hidden=true;setTimeout(()=>el.hidden=false,500)}")
    )
    assignment = PlayerAssignment(team=Team.BLUE, role=Role.SPYMASTER, nickname="BlueSpymasterAI")
    await join_room(live_page, assignment)
    assert await live_page.evaluate("joinedPanels") == ["blueSpy"]


async def test_duplicate_join_text_does_not_block_click(live_page: Page) -> None:
    await lobby(live_page)
    await (
        live_page.locator(ROLE_PANELS[(Team.BLUE, Role.SPYMASTER)])
        .locator("button.joinTeam")
        .evaluate("el=>el.innerHTML='<span>JOIN TEAM</span><span>JOIN TEAM</span>'")
    )
    assignment = PlayerAssignment(team=Team.BLUE, role=Role.SPYMASTER, nickname="BlueSpymasterAI")
    await join_room(live_page, assignment)
    assert await live_page.evaluate("joinedPanels") == ["blueSpy"]


async def test_site_assigned_player1_joins_expected_role(
    live_page: Page, caplog: pytest.LogCaptureFixture
) -> None:
    await lobby(live_page)
    await live_page.evaluate("window.nicknameOverride='Player1'")
    assignment = PlayerAssignment(team=Team.BLUE, role=Role.SPYMASTER, nickname="BlueSpymasterAI")
    with caplog.at_level(logging.INFO):
        await join_room(live_page, assignment)
    assert await live_page.evaluate("joinedPanels") == ["blueSpy"]
    assert (await detect_assignment(live_page, assignment)).nickname == "Player1"
    assert "[BlueSpymasterAI] entered room" in caplog.text
    assert "[BlueSpymasterAI] joined blue spymaster" in caplog.text


async def test_join_succeeds_without_settings_nickname(live_page: Page) -> None:
    await lobby(live_page)
    await live_page.locator('[data-testid="settings-button"]').evaluate("el=>el.remove()")
    await live_page.locator("#settings").evaluate("el=>el.remove()")
    assignment = PlayerAssignment(team=Team.BLUE, role=Role.SPYMASTER, nickname="BlueSpymasterAI")
    await join_room(live_page, assignment)
    assert await live_page.evaluate("joinedPanels") == ["blueSpy"]


async def test_missing_join_button_fails_clearly(live_page: Page) -> None:
    await lobby(live_page)
    await (
        live_page.locator(ROLE_PANELS[(Team.BLUE, Role.SPYMASTER)])
        .locator("button.joinTeam")
        .evaluate("el=>el.remove()")
    )
    assignment = PlayerAssignment(team=Team.BLUE, role=Role.SPYMASTER, nickname="BlueSpymasterAI")
    with pytest.raises(PlayerAssignmentError, match="JOIN TEAM button count=0; panel text='ROLE'"):
        await join_room(live_page, assignment)
