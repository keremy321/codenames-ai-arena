import time
from dataclasses import dataclass
from typing import Any

from codenames_ai.domain.enums import Role, Team
from codenames_ai.llm.ollama import OllamaClient


@dataclass(frozen=True)
class Agent:
    team: Team
    role: Role
    llm: OllamaClient


@dataclass(frozen=True)
class LLMCall:
    """One model request made while taking a decision (for logs and evaluation)."""

    purpose: str
    seconds: float
    ok: bool
    error: str = ""


async def timed_chat(
    llm: OllamaClient, calls: list[LLMCall], purpose: str, **kwargs: Any
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        reply = await llm.chat_json(**kwargs)
    except Exception as exc:
        calls.append(LLMCall(purpose, time.perf_counter() - started, False, str(exc)[:200]))
        raise
    calls.append(LLMCall(purpose, time.perf_counter() - started, True))
    return reply


def describe_calls(calls: list[LLMCall]) -> str:
    total = sum(call.seconds for call in calls)
    detail = ", ".join(
        f"{call.purpose} {call.seconds:.1f}s{'' if call.ok else ' FAILED'}" for call in calls
    )
    return f"{len(calls)} ollama calls, {total:.1f}s" + (f" ({detail})" if detail else "")
