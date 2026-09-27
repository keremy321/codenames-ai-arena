"""One DOM extraction component shared by readers, actions, and debug tooling.

No screenshots, network state, or framework internals. Cover/word associations
use DOM layout rectangles because the live site portals them as siblings.
"""

from . import selectors as s

DOM_ARGS = {
    "liveCards": s.LIVE_CARDS,
    "legacyCards": s.CARDS,
    "covers": s.COVERS,
    "instruction": s.INSTRUCTION,
    "clue": s.CLUE_AREA,
    "number": s.CLUE_NUMBER,
    "panels": list(s.ROLE_PANELS.values()),
    "lobby": s.LOBBY_SETTINGS,
    "clueInput": s.CLUE_INPUT,
    "confirm": s.CONFIRM_GUESS,
}

# Kept in one function so wait_for_function observes exactly the same semantics
# as a reader snapshot. Styles never cross the JS boundary for unrevealed cards
# in an operative read.
SNAPSHOT_JS = """(args) => {
    const visible = el => !!el && el.getClientRects().length > 0 &&
        getComputedStyle(el).visibility !== 'hidden' && getComputedStyle(el).display !== 'none';
    const text = el => el ? el.textContent.trim() : '';
    const all = sel => Array.from(document.querySelectorAll(sel));
    const live = all(args.liveCards).filter(visible);
    const roots = live.length ? live : all(args.legacyCards).filter(el =>
        visible(el) && !el.parentElement.closest(args.legacyCards));
    const covers = all(args.covers).filter(visible);
    const cards = roots.map((el, index) => {
        const rect = el.getBoundingClientRect();
        const matches = covers.filter(cover => {
            const r = cover.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0 &&
                ['x','y','width','height'].every(k => Math.abs(rect[k]-r[k]) < 0.75);
        });
        const revealed = matches.length === 1;
        const coverStyle = revealed ? Array.from(matches[0].querySelectorAll('[style]'))
            .map(n => n.style.backgroundImage).find(v => v.includes('--cover-card-bg-')) : '';
        const wordNode = el.querySelector('section article') || el;
        const copy = wordNode.cloneNode(true);
        copy.querySelectorAll('aside,button,[aria-hidden="true"]').forEach(n => n.remove());
        const selected = !revealed && Array.from(el.children).some(n =>
            n.tagName === 'SECTION' && n.classList.contains('scale-[1.02]'));
        return {index, text: wordNode === el ? el.innerText : copy.textContent, revealed, selected,
            style: args.includeHidden ? (el.getAttribute('style') || '') : '',
            coverStyle: coverStyle || '', ambiguousCover: matches.length > 1};
    });
    const area = document.querySelector(args.clue);
    const instruction = text(document.querySelector(args.instruction));
    const activePanels = args.panels.filter(sel => {
        const main = document.querySelector(sel + ' > main');
        return visible(main) && main.classList.contains('activeRoleShadow');
    });
    return {cards, instruction, activePanels,
        clueWords: area ? Array.from(area.querySelectorAll('p')).filter(visible).map(text).filter(Boolean) : [],
        clueNumber: area ? text(area.querySelector(args.number)) : '',
        clueInput: visible(document.querySelector(args.clueInput)),
        lobby: visible(document.querySelector(args.lobby)),
        gameOver: ['Opposing team wins!', 'Your team wins!'].includes(instruction) &&
            activePanels.length === 2 && activePanels.every(p =>
                p.includes('blue') === activePanels[0].includes('blue')),
        confirmationCount: all(args.confirm).filter(visible).length,
        live: live.length > 0};
}"""
