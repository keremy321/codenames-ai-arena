"""Append-only match logs that never decide anything and never block the game.

Every write is best effort: the first failure is logged clearly and recording stops,
while the match itself carries on. Agent wrappers only observe finished decisions,
so nothing recorded here ever flows back into a prompt.
"""

import json
import logging
import os
import secrets
import time
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from codenames_ai.domain.enums import CardColor
from codenames_ai.domain.models import (
    ClueDecision,
    GuessDecision,
    PublicGameState,
    SpymasterGameState,
)
from codenames_ai.llm.base import LLMConfig

logger = logging.getLogger(__name__)

# Relative to the working directory: the repository root when run as documented.
LOGS_DIR = Path("logs")
REDACTED = "[REDACTED]"


def now() -> datetime:
    return datetime.now().astimezone()


def room_code(room_url: str) -> str:
    segments = [part for part in urlparse(room_url).path.split("/") if part]
    return segments[-1] if segments else ""


class MatchRecorder:
    def __init__(
        self,
        root: Path,
        *,
        room_url: str,
        players: Mapping[str, LLMConfig],
        secret_values: Iterable[str] = (),
    ) -> None:
        self.created = now()
        self._secrets = [value for value in secret_values if value and len(value) >= 8]
        self._failed = False
        self._finished = False
        self._started_clock: float | None = None
        code = room_code(room_url)
        self.match_id = f"{self.created:%Y-%m-%dT%H%M%S}_{code or secrets.token_hex(3)}"
        self.directory = root / self.match_id
        self.metadata: dict[str, Any] = {
            "match_id": self.match_id,
            "room_url": room_url,
            "room_code": code,
            "created_at": self.created.isoformat(timespec="seconds"),
            "started_at": None,
            "ended_at": None,
            "duration_seconds": None,
            "winner": None,
            "termination_reason": None,
            "error": None,
            "players": {
                role: {"provider": config.provider, "model": config.model}
                for role, config in players.items()
            },
        }
        try:
            root.mkdir(parents=True, exist_ok=True)
            suffix = 1
            while True:
                try:
                    self.directory.mkdir()
                    break
                except FileExistsError:
                    suffix += 1
                    self.directory = root / f"{self.match_id}-{suffix}"
            self.match_id = self.metadata["match_id"] = self.directory.name
        except OSError as exc:
            self._fail(exc)
        self._write_metadata()

    @property
    def events_path(self) -> Path:
        return self.directory / "events.jsonl"

    @property
    def metadata_path(self) -> Path:
        return self.directory / "match.json"

    def event(self, type: str, **fields: Any) -> None:
        record = {"type": type, "timestamp": now().isoformat(timespec="milliseconds"), **fields}
        self._write(lambda: self._append(self._dumps(record)))

    def start(self) -> None:
        self._started_clock = time.monotonic()
        self.metadata["started_at"] = now().isoformat(timespec="seconds")
        self._write_metadata()
        self.event("match_started")

    def finish(self, *, reason: str, winner: str | None = None, error: str | None = None) -> None:
        """Finalize match.json once; later calls (e.g. from a cleanup path) are ignored."""
        if self._finished:
            return
        self._finished = True
        ended = now()
        started = self._started_clock
        self.metadata.update(
            ended_at=ended.isoformat(timespec="seconds"),
            duration_seconds=round(
                time.monotonic() - started
                if started is not None
                else (ended - self.created).total_seconds(),
                1,
            ),
            winner=winner,
            termination_reason=reason,
            error=error,
        )
        self.event("match_ended", reason=reason, winner=winner, error=error)
        self._write_metadata()

    def attach(self, name: str, data: bytes) -> bool:
        """Save a binary file (e.g. a screenshot) next to the logs; best effort."""
        written = False

        def write() -> None:
            nonlocal written
            (self.directory / name).write_bytes(data)
            written = True

        self._write(write)
        return written

    def _dumps(self, value: Any, **kwargs: Any) -> str:
        text = json.dumps(value, default=str, ensure_ascii=False, **kwargs)
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        return text

    def _append(self, line: str) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _write_metadata(self) -> None:
        def write() -> None:
            # Replace atomically: a crash never leaves a half-written match.json.
            temporary = self.metadata_path.with_suffix(".json.tmp")
            temporary.write_text(self._dumps(self.metadata, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.metadata_path)

        self._write(write)

    def _write(self, action: Any) -> None:
        if self._failed:
            return
        try:
            action()
        except (OSError, TypeError, ValueError) as exc:
            self._fail(exc)

    def _fail(self, exc: Exception) -> None:
        if not self._failed:
            self._failed = True
            logger.error(
                "Match recording to %s failed (%s); the game continues unrecorded",
                self.directory,
                exc,
            )


def _calls(agent: object) -> list[dict[str, Any]]:
    calls = getattr(getattr(agent, "last_trace", None), "calls", None) or []
    return [
        {"purpose": c.purpose, "seconds": round(c.seconds, 3), "ok": c.ok}
        | ({"error": c.error} if c.error else {})
        for c in calls
    ]


def _rounded(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


class _RecordedAgent:
    """Delegates the decision unchanged, then records it with the agent's own trace."""

    def __init__(self, agent: Any, recorder: MatchRecorder, config: LLMConfig) -> None:
        self.agent = agent
        self.recorder = recorder
        self.config = config

    def _record(self, type: str, started: float, **fields: Any) -> None:
        self.recorder.event(
            type,
            team=self.agent.team,
            role=self.agent.role,
            provider=self.config.provider,
            model=self.config.model,
            seconds=round(time.perf_counter() - started, 3),
            llm_calls=_calls(self.agent),
            **fields,
        )

    def _error(self, started: float, exc: Exception) -> None:
        self._record("agent_error", started, error=f"{type(exc).__name__}: {exc}"[:500])


METRIC_FIELDS = (
    "expected_value",
    "expected_hits",
    "p_first_friendly",
    "finish_chance",
    "win_probability",
)


def _side(state: SpymasterGameState, word: str) -> str:
    color = next((card.color for card in state.cards if card.word == word), None)
    if color == CardColor(state.team.value):
        return "friendly"
    if color == CardColor(state.team.other.value):
        return "enemy"
    return color.value if color in (CardColor.NEUTRAL, CardColor.ASSASSIN) else "unknown"


def selection_record(selected: Any, state: SpymasterGameState) -> dict[str, Any]:
    """The chosen assessment with unambiguous names (spymaster-only; logs only).

    - generator_targets: the words the clue generator proposed the clue for. The
      clue number is chosen separately, so it can be larger or smaller than this.
    - expected_guesses: friendly words the blind operative most likely finds with
      exactly this number (never more than the number).
    - predicted_ranking: the blind operative ranking the scores were computed from.
    - verification_ran: that ranking came from the operative-identical request. The
      event's selection_mode says whether this was a normal verified choice
      (verified / regenerated_verified) or the degraded fallback.
    - verification_passed: it ran AND the action passed the checks (acceptable: not
      vetoed, operative's top pick friendly, target consistency); None if it did not
      run. A selected action can fail them when no candidate passed (see selection_note).
    - target_overlap / target_precision: generator targets among the operative's first
      ``number`` predicted picks, as a count and as a share of those picks (TINY 2 with
      one target and picks DWARF, ASH: 1 and 0.5).
    - target_consistency_passed: that overlap was enough (clue_scoring.required_overlap).
    """
    ran = bool(getattr(selected, "verified", False))
    acceptable = bool(getattr(selected, "acceptable", False))
    return {
        "generator_targets": list(getattr(selected, "intended", ())),
        "expected_guesses": list(getattr(selected, "hits", ())),
        "predicted_ranking": [
            {"word": r.word, "fit": str(r.fit), "side": _side(state, r.word)}
            for r in getattr(selected, "ranking", ())
        ],
        "verification_ran": ran,
        "verification_passed": acceptable if ran else None,
        "acceptable": acceptable,
        "vetoed": bool(getattr(selected, "vetoed", False)),
        "reason": getattr(selected, "reason", None),
        "target_overlap": getattr(selected, "target_overlap", None),
        "target_precision": _rounded(getattr(selected, "target_precision", None)),
        "target_consistency_passed": getattr(selected, "target_consistent", None),
        **{name: _rounded(getattr(selected, name, None)) for name in METRIC_FIELDS},
    }


class RecordedSpymaster(_RecordedAgent):
    async def choose_clue(self, state: SpymasterGameState) -> ClueDecision:
        started = time.perf_counter()
        try:
            decision = await self.agent.choose_clue(state)
        except Exception as exc:
            self._error(started, exc)
            raise
        trace = getattr(self.agent, "last_trace", None)
        selected = getattr(trace, "selected", None)
        self._record(
            "spymaster_decision",
            started,
            clue=decision.word,
            number=decision.number,
            selection=selection_record(selected, state) if selected is not None else None,
            selection_mode=getattr(trace, "selection_mode", "") or None,
            selection_note=getattr(trace, "note", "") or None,
            fallback_round=getattr(trace, "fallback_round", None),
            regeneration=getattr(trace, "focused_regeneration", "") or None,
            # Clues dropped before any ranking: legality, naturalness, reuse.
            rejected_clues=[
                {"clue": clue, "reason": reason}
                for clue, reason in getattr(trace, "rejected", None) or []
            ],
        )
        return decision


class RecordedOperative(_RecordedAgent):
    async def choose_guesses(
        self, state: PublicGameState, *, guesses_made: int, max_guesses: int
    ) -> GuessDecision:
        started = time.perf_counter()
        try:
            decision = await self.agent.choose_guesses(
                state, guesses_made=guesses_made, max_guesses=max_guesses
            )
        except Exception as exc:
            self._error(started, exc)
            raise
        trace = getattr(self.agent, "last_trace", None)
        words = {card.index: card.word for card in state.cards}
        self._record(
            "operative_decision",
            started,
            clue=state.clue.word if state.clue else None,
            clue_number=state.clue.number if state.clue else None,
            mode=getattr(trace, "mode", None),
            guesses_made=guesses_made,
            max_guesses=max_guesses,
            guesses=[words.get(index) for index in decision.indices],
            end_turn=decision.end_turn,
            source_clue=decision.source_clue,
            reason=getattr(getattr(trace, "decision", None), "reason", None),
            reused_ranking=getattr(trace, "reused_ranking", None),
        )
        return decision
