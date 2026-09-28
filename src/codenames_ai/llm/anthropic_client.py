"""Anthropic Messages transport; stateless like the Ollama client."""

from types import TracebackType
from typing import Any, Self

import anthropic

from .base import (
    HOSTED_MAX_OUTPUT_TOKENS,
    LLMResponseError,
    convert_sdk_error,
    parse_json_object,
)

# Constraints Anthropic structured outputs reject. Dropping them loosens only the shape
# the model is held to: the agents' parsers already discard extra or invalid entries.
_UNSUPPORTED = frozenset(
    {
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    }
)


def anthropic_schema(node: Any) -> Any:
    """Copy of a JSON schema without keywords Anthropic cannot enforce."""
    if isinstance(node, list):
        return [anthropic_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    result: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED or (key == "minItems" and value > 1):
            continue
        # Property names are data (board words), never schema keywords.
        result[key] = (
            {name: anthropic_schema(sub) for name, sub in value.items()}
            if key == "properties"
            else anthropic_schema(value)
        )
    return result


class AnthropicClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.model = model
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.close()

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        num_predict: int = 192,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        # ``temperature`` is accepted for interface parity but not sent: current Claude
        # models reject sampling parameters and the SDK no longer exposes them.
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(num_predict, HOSTED_MAX_OUTPUT_TOKENS),
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if schema:
            request["output_config"] = {
                "format": {"type": "json_schema", "schema": anthropic_schema(schema)}
            }
        try:
            message = await self._client.messages.create(**request)
        except anthropic.APIError as exc:
            raise convert_sdk_error(anthropic, exc, "Anthropic", self.model) from exc
        if message.stop_reason == "refusal":
            raise LLMResponseError("Anthropic refused the request")
        if message.stop_reason == "max_tokens":
            raise LLMResponseError("Anthropic reply hit the output token limit")
        text = "".join(block.text for block in message.content if block.type == "text")
        return parse_json_object(text, "Anthropic")
