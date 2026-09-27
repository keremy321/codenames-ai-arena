# Live DOM evidence

Captured on 2026-09-27 from a disposable English classic game on codenames.game.
These are actual scoped DOM fragments, reduced with the project's allowlist
sanitizer. They contain no room URL, cookies, session tokens, generated runtime
IDs, or original player nicknames. `FixturePlayer` replaces player names.

`card-revealed-*.html` contains **two sibling articles**: the word card and its
matching cover. Their identical DOM layout positions are intentional. The live
site keeps cards and covers outside the empty grid slot; they are descendants
of `#appMount`. Offline tests add that wrapper and minimal layout CSS.

- Word-card root: `article[style*="--CardColor:"]`; nested articles are not cards.
- Assassin word-card color: `var(--black-cardBg)`.
- Public reveal: a `--BgScale` cover occupies the same DOM rectangle as a word
  card. The cover has `var(--cover-card-bg-{blue,red,neutral,black}-url)`.
- Local selection: direct child section has class `scale-[1.02]`; an avatar
  inside the selected card is not part of its word.
- Guess confirmation: separate button contains `img` ending `/icon-touch-card.svg`.
- Clue entry: `Your clue` input; count button uses `aria-haspopup=dialog` without
  `data-slot`; popup count buttons include 0–9 and infinity. Dynamic IDs have
  been replaced consistently within each individual snippet.
- Clue action: unique `button[data-slot=tooltip-trigger]` within clue slot;
  submits when the input is present. When ending guessing without a confirmed
  guess, it opens a dialog with an `End Guessing` button that must be clicked.
- Game over: `Opposing team wins!` / `Your team wins!`, corroborated by both
  winning role panels being active.

`role-panel.html` proves the nickname/role relationship, not local identity by
itself. The local user's nickname must first be read from Settings > Player >
`#settings-nickname-input`. Hosts start on the Admin tab. The active-role class
tracks the current turn, not the current user.

The JavaScript interaction simulator in `test_live_adapter.py` is synthetic;
it exercises actions with these observed elements without contacting the site.
