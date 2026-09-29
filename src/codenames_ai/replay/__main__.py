"""Print a recorded match turn by turn: python -m codenames_ai.replay logs/<match>"""

import argparse
import json
import sys
from collections import Counter

from .models import Replay, ReplayError
from .replay import load_replay


def format_replay(replay: Replay) -> str:
    state, meta = replay.state, replay.metadata
    sides = Counter(card.side.value for card in state.board.cards)
    start = state.board.starting_team
    lines = [
        f"Match: {meta.get('match_id') or replay.directory}",
        f"Board: {len(state.board.cards)} cards ("
        + ", ".join(f"{sides[s]} {s}" for s in ("blue", "red", "neutral", "assassin"))
        + ")",
        f"Starting team: {start.value.capitalize() if start else 'unknown'}",
    ]
    for turn in state.turns:
        lines += ["", f"Turn {turn.number} - {turn.team.value.capitalize()}"]
        lines.append(f"Clue: {turn.clue} {turn.clue_number}")
        for guess in turn.guesses:
            bonus = " (bonus)" if guess.bonus else ""
            lines.append(f"  {guess.word} -> {guess.side.value}{bonus}")
        if not turn.guesses:
            lines.append("  (no guesses)")
    winner = state.winner.value.capitalize() if state.winner else "none"
    decided = f" ({state.decided_by})" if state.decided_by else ""
    lines += ["", f"Winner: {winner}{decided}"]
    lines.append(f"Termination: {state.termination_reason or 'not recorded'}")
    lines += ["", "Summary:"]
    lines += [f"  {key}: {value}" for key, value in replay.summary.to_dict().items()]
    lines += [f"Warning: {warning}" for warning in state.warnings]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m codenames_ai.replay",
        description="Replay a recorded match locally (read-only; no browser or model)",
    )
    parser.add_argument("match", help="Match directory, e.g. logs/<match-id>")
    parser.add_argument("--json", action="store_true", help="Print only the summary as JSON")
    args = parser.parse_args(argv)
    try:
        replay = load_replay(args.match)
    except ReplayError as exc:
        print(f"Cannot replay {args.match}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(replay.summary.to_dict(), indent=2))
    else:
        print(format_replay(replay))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
