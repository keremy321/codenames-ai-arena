import json
from types import TracebackType
from typing import Any, Self

import httpx

from .base import LLMError, LLMHTTPError, LLMModelError, LLMResponseError, LLMTimeoutError


class OllamaError(LLMError):
    pass


class OllamaHTTPError(OllamaError, LLMHTTPError):
    pass


class OllamaTimeoutError(OllamaError, LLMTimeoutError):
    pass


class OllamaResponseError(OllamaError, LLMResponseError):
    pass


class OllamaModelError(OllamaError, LLMModelError):
    pass


def keep_alive_value(text: str) -> str | int:
    """Ollama's keep_alive: a duration ("30m") or whole seconds (-1 keeps it loaded)."""
    text = text.strip()
    return int(text) if text.lstrip("-").isdigit() else text


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:14b",
        *,
        timeout: float = 120,
        keep_alive: str | int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        # Sent with every request so the model stays resident for the whole match.
        self.keep_alive = keep_alive
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/", timeout=timeout, transport=transport
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._http.aclose()

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        num_predict: int = 192,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"num_predict": num_predict}
        if temperature is not None:
            options["temperature"] = temperature
        payload = await self._chat(
            {
                "format": schema or "json",
                "options": options,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        )
        try:
            content = payload["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("content must be text")
            result = json.loads(content)
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaResponseError("Ollama message.content must contain valid JSON") from exc
        if not isinstance(result, dict):
            raise OllamaResponseError("Ollama JSON output must be an object")
        return result

    async def warm_up(self) -> None:
        """Load the model and run one token, so the first game call does not pay for it.

        Same endpoint and option names as chat_json (no num_ctx): a different context
        size would make Ollama reload the model on the first real call.
        """
        payload = await self._chat(
            {"options": {"num_predict": 1}, "messages": [{"role": "user", "content": "Say OK."}]}
        )
        if not isinstance(payload.get("message"), dict):
            raise OllamaResponseError(f"Ollama warm-up of {self.model} returned no message")

    async def _chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST api/chat; the reply envelope, with transport and server errors mapped."""
        body = {"model": self.model, "stream": False, "think": False, **body}
        if self.keep_alive is not None:
            body["keep_alive"] = self.keep_alive
        try:
            response = await self._http.post("api/chat", json=body)
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(f"Ollama timed out using model {self.model}") from exc
        except httpx.RequestError as exc:
            raise OllamaHTTPError(f"Cannot reach Ollama: {exc}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            if response.is_error:
                raise OllamaHTTPError(f"Ollama HTTP {response.status_code}") from exc
            raise OllamaResponseError("Ollama response envelope is not JSON") from exc
        if isinstance(payload, dict) and payload.get("error"):
            raise OllamaModelError(
                f"Ollama HTTP {response.status_code}, model {self.model}: {str(payload['error'])[:300]}"
            )
        if response.is_error:
            raise OllamaHTTPError(f"Ollama HTTP {response.status_code}")
        if not isinstance(payload, dict):
            raise OllamaResponseError("Ollama response envelope must be an object")
        return payload
