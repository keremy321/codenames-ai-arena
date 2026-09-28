"""OpenAI Chat Completions transport; stateless like the Ollama client."""

import logging
from types import TracebackType
from typing import Any, Self

import openai

from .base import (
    HOSTED_MAX_OUTPUT_TOKENS,
    LLMResponseError,
    convert_sdk_error,
    parse_json_object,
)

logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self._client = client or openai.AsyncOpenAI(api_key=api_key)
        # Some models accept only their default temperature; learned on first rejection.
        self._send_temperature = True

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
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max(num_predict, HOSTED_MAX_OUTPUT_TOKENS),
            "response_format": (
                {
                    "type": "json_schema",
                    "json_schema": {"name": "reply", "schema": schema, "strict": True},
                }
                if schema
                else {"type": "json_object"}
            ),
        }
        if temperature is not None and self._send_temperature:
            request["temperature"] = temperature
        completion = await self._create(request)
        if not completion.choices:
            raise LLMResponseError("OpenAI returned no choices")
        choice = completion.choices[0]
        if choice.message.refusal:
            raise LLMResponseError(f"OpenAI refused: {choice.message.refusal[:200]}")
        if choice.finish_reason == "length":
            raise LLMResponseError("OpenAI reply hit the output token limit")
        return parse_json_object(choice.message.content, "OpenAI")

    async def _create(self, request: dict[str, Any]) -> Any:
        try:
            return await self._client.chat.completions.create(**request)
        except openai.BadRequestError as exc:
            if "temperature" not in request or "temperature" not in str(exc).lower():
                raise convert_sdk_error(openai, exc, "OpenAI", self.model) from exc
            logger.warning("OpenAI model %s rejects temperature; omitting it", self.model)
            self._send_temperature = False
            request.pop("temperature")
        except openai.APIError as exc:
            raise convert_sdk_error(openai, exc, "OpenAI", self.model) from exc
        try:
            return await self._client.chat.completions.create(**request)
        except openai.APIError as exc:
            raise convert_sdk_error(openai, exc, "OpenAI", self.model) from exc
