"""Developer diagnostics. Snippets are secret-bearing debug data, never agent state."""

import asyncio
import logging
from pathlib import Path

from playwright.async_api import Page

from . import selectors as s
from .errors import PlayerAssignmentError
from .identity import detect_assignment
from .reader import GameReader, detect_phase, detect_turn, parse_clue

logger = logging.getLogger(__name__)

SANITIZE_JS = r"""({html, names}) => {
    const doc = new DOMParser().parseFromString(html, 'text/html');
    doc.querySelectorAll('script,style,noscript,iframe,link,meta,path,figure').forEach(n=>n.remove());
    const keep = new Set(['class','style','role','type','placeholder','maxlength','disabled',
        'aria-label','aria-haspopup','aria-expanded','aria-hidden','aria-checked',
        'data-match-slot','data-quick-guide-anchor','data-slot','data-testid']);
    const classes = new Set(['absolute','relative','top-0','left-0','inset-0','block',
        'hidden','activeRoleShadow','scale-[1.02]','scale-100','joinTeam','rotate-90']);
    const properties = new Set(['--CardColor','--BgScale','width','height','translate',
        'scale','rotate','z-index','position','display','visibility','background-image']);
    const ids = new Map();
    const id = value => {
        if (value === 'appMount' || value === 'settings-nickname-input') return value;
        if (!ids.has(value)) ids.set(value, 'fixture-id-' + (ids.size+1));
        return ids.get(value);
    };
    for (const el of doc.body.querySelectorAll('*')) {
        if (el.tagName === 'IMG') {
            if ((el.getAttribute('src') || '').split('?')[0].endsWith('/icon-touch-card.svg')) {
                el.setAttribute('src', '/icon-touch-card.svg');
            } else {el.remove(); continue;}
        }
        for (const a of Array.from(el.attributes)) {
            if (a.name === 'id' || a.name === 'aria-controls') {
                el.setAttribute(a.name, id(a.value));
            } else if (a.name === 'src' && el.tagName === 'IMG') {
                continue;
            } else if (!keep.has(a.name)) el.removeAttribute(a.name);
        }
        if (el.hasAttribute('class')) {
            el.setAttribute('class', el.getAttribute('class').split(/\s+/).filter(c=>classes.has(c)).join(' '));
            if (!el.getAttribute('class')) el.removeAttribute('class');
        }
        if (el.hasAttribute('style')) {
            const parts = Array.from(el.style).filter(p=>properties.has(p)).map(p=>[p,el.style.getPropertyValue(p)]);
            el.removeAttribute('style');
            for (const [p,v] of parts) {
                if (/url\s*\(|https?:|[<>]/i.test(v)) continue;
                if (p === 'background-image' && !/^var\(--cover-card-bg-[a-z]+-url\)$/.test(v.trim())) continue;
                el.style.setProperty(p,v);
            }
        }
    }
    const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
        let text=walker.currentNode.textContent;
        for (const name of names) if (name) text=text.split(name).join('FixturePlayer');
        walker.currentNode.textContent=text.replace(/https?:\/\/\S+/g,'[URL removed]');
    }
    return doc.body.innerHTML;
}"""


async def sanitize_snippet(page: Page, html: str, nicknames: list[str]) -> str:
    return await page.evaluate(SANITIZE_JS, {"html": html, "names": nicknames})


async def save_snapshots(page: Page, directory: Path) -> list[Path]:
    """Save scoped, allowlisted DOM fragments, without room URLs or session data."""
    raw = await GameReader(page)._snapshot(include_hidden=True)
    names = await page.locator('[style*="--avatar-"][style*="nameBg"]').all_text_contents()
    names = list(dict.fromkeys(n.strip() for n in names if n.strip()))
    pieces: dict[str, str] = {}
    roots = page.locator(s.LIVE_CARDS)
    if not await roots.count():
        roots = page.locator(s.CARDS)
    for card in raw["cards"]:
        kind = "card-selected" if card["selected"] else "card-unrevealed"
        if card["revealed"]:
            from .reader import parse_cover_color

            kind = "card-revealed-" + parse_cover_color(card["coverStyle"]).value
        elif "--black-cardBg" in card["style"]:
            kind = "card-assassin"
        if kind not in pieces:
            pieces[kind] = await roots.nth(card["index"]).evaluate(
                """(el, covers) => {
                    const r=el.getBoundingClientRect();
                    const matched=Array.from(document.querySelectorAll(covers)).filter(c=>{
                        const q=c.getBoundingClientRect();
                        return ['x','y','width','height'].every(k=>Math.abs(r[k]-q[k])<0.75);
                    });
                    return el.outerHTML + matched.map(c=>c.outerHTML).join('');
                }""",
                s.COVERS,
            )
    for name, selector in {
        "clue-area": s.CLUE_AREA,
        "guess-confirmation": s.CONFIRM_GUESS,
        "instruction": s.INSTRUCTION,
        **{f"role-{team}-{role}": sel for (team, role), sel in s.ROLE_PANELS.items()},
    }.items():
        locator = page.locator(selector)
        if await locator.count() == 1:
            pieces[name] = await locator.evaluate("el=>el.outerHTML")
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, html in pieces.items():
        path = directory / f"{name}.html"
        path.write_text(await sanitize_snippet(page, html, names), encoding="utf-8")
        paths.append(path)
    return paths


async def probe_summary(page: Page) -> str:
    raw = await GameReader(page)._snapshot(include_hidden=False)
    turn = detect_turn(raw["instruction"], raw["activePanels"])
    clue = parse_clue(raw["clueWords"], raw["clueNumber"])
    buttons = (
        await page.locator(s.CLUE_AREA)
        .locator("button")
        .evaluate_all(
            "els=>els.map(el=>({text:el.textContent.trim(),label:el.getAttribute('aria-label'),"
            "popup:el.getAttribute('aria-haspopup'),slot:el.getAttribute('data-slot'),disabled:el.disabled}))"
        )
    )
    try:
        identity = str(await detect_assignment(page))
    except PlayerAssignmentError as exc:
        identity = f"Unverified: {exc}"
    selected = [c["index"] for c in raw["cards"] if c["selected"]]
    return (
        f"Instruction: {raw['instruction'] or 'unknown'}\n"
        f"Active role: {turn.team or 'unknown'} {turn.role or 'unknown'}\n"
        f"Local player: {identity}\nPhase: {detect_phase(raw)}\n"
        f"Cards: {len(raw['cards'])}; revealed: {sum(c['revealed'] for c in raw['cards'])}\n"
        f"Clue: {clue.word + ' ' + str(clue.number) if clue else 'none / unreadable'}\n"
        f"Clue buttons: {buttons}\nSelected card indices: {selected}\n"
        f"Guess confirmation controls: {raw['confirmationCount']}"
    )


async def inspect_loop(page: Page, directory: Path) -> None:
    print("Join manually in Chromium. Enter refreshes; s saves minimal snippets; q exits.")
    while True:
        command = (await asyncio.to_thread(input, "inspect [Enter/s/q]> ")).strip().lower()
        if command == "q":
            return
        print(await probe_summary(page))
        if command == "s":
            paths = await save_snapshots(page, directory)
            print(
                f"Saved {len(paths)} minimal snippets to {directory}. Debug data is never sent to agents."
            )
