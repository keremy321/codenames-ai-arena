from codenames_ai.domain.enums import Role, Team

INSTRUCTION = '[data-match-slot="instruction"]'
BLUE_OPERATIVE = '[data-match-slot="blueOp"]'
BLUE_SPYMASTER = '[data-match-slot="blueSpy"]'
RED_OPERATIVE = '[data-match-slot="redOp"]'
RED_SPYMASTER = '[data-match-slot="redSpy"]'
BLUE_DECK = '[data-match-slot="blueDeck"]'
RED_DECK = '[data-match-slot="redDeck"]'
CLUE_AREA = '[data-match-slot="clue"]'
CLUE_NUMBER = '[data-quick-guide-anchor="clue-number"]'
GRID = '[data-match-slot="grid"]'
CARDS = f"{GRID} article"
# Live observations: 2026-09-27, English classic game. Board word cards and
# revealed covers are portaled under #appMount, outside the empty grid slot.
LIVE_CARDS = '#appMount article[style*="--CardColor:"]'
COVERS = '#appMount article[style*="--BgScale:"]'
CONFIRM_GUESS = '#appMount button:has(img[src$="/icon-touch-card.svg"])'
CLUE_INPUT = f'{CLUE_AREA} input[placeholder="Your clue"]'
CLUE_COUNT_BUTTON = f'{CLUE_AREA} button[aria-haspopup="dialog"]:not([data-slot])'
CLUE_ACTION = f'{CLUE_AREA} button[data-slot="tooltip-trigger"]'
SETTINGS_BUTTON = '[data-testid="settings-button"]'
NICKNAME_INPUT = "#settings-nickname-input"
ENTRY_NICKNAME = "#nickname"
JOIN_TEAM = "button.joinTeam"
LOBBY_SETTINGS = '[data-match-slot="settings"]'
ROLE_PANELS = {
    (Team.BLUE, Role.OPERATIVE): BLUE_OPERATIVE,
    (Team.BLUE, Role.SPYMASTER): BLUE_SPYMASTER,
    (Team.RED, Role.OPERATIVE): RED_OPERATIVE,
    (Team.RED, Role.SPYMASTER): RED_SPYMASTER,
}
