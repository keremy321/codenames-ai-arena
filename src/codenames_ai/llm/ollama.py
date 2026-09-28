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


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:14b",
        *,
        timeout: float = 120,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
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
        try:
            response = await self._http.post(
                "api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "think": False,
                    "format": schema or "json",
                    "options": options,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
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
