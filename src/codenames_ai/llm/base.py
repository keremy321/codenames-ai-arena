"""Provider-neutral model interface: one stateless JSON request per call."""

import json
from types import ModuleType
from typing import Any, Literal, Protocol, get_args

from pydantic import BaseModel, ConfigDict, Field

Provider = Literal["ollama", "openai", "anthropic"]
PROVIDERS: tuple[str, ...] = get_args(Provider)


class LLMError(RuntimeError):
    pass


class LLMHTTPError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMResponseError(LLMError):
    """The model answered, but not with the JSON object the agent asked for."""


class LLMModelError(LLMError):
    pass


class LLMClient(Protocol):
    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        num_predict: int = 192,
        temperature: float | None = None,
    ) -> dict[str, Any]: ...


class LLMConfig(BaseModel):
    """Which provider and model one role talks to (hashable: equal configs may share)."""

    model_config = ConfigDict(frozen=True)

    provider: Provider
    model: str = Field(min_length=1)

    def __str__(self) -> str:
        return f"{self.provider} / {self.model}"


# Hosted models may reason before answering and count that against the output cap;
# ``num_predict`` was sized for local non-thinking replies. A ceiling, not a target:
# the JSON replies themselves stay short.
HOSTED_MAX_OUTPUT_TOKENS = 16_000


def convert_sdk_error(sdk: ModuleType, exc: Exception, provider: str, model: str) -> LLMError:
    """Map an OpenAI/Anthropic SDK exception (same class names in both) onto LLM errors.

    Credential failures never echo the provider's message: it may quote part of the key.
    """
    if isinstance(exc, sdk.APITimeoutError):
        return LLMTimeoutError(f"{provider} timed out using model {model}")
    if isinstance(exc, sdk.APIConnectionError):
        return LLMHTTPError(f"Cannot reach {provider}: {type(exc).__name__}")
    if isinstance(exc, sdk.AuthenticationError | sdk.PermissionDeniedError):
        return LLMHTTPError(f"{provider} HTTP {exc.status_code}: API key rejected")
    if isinstance(exc, sdk.NotFoundError | sdk.BadRequestError):
        return LLMModelError(
            f"{provider} HTTP {exc.status_code}, model {model}: {str(exc.message)[:300]}"
        )
    if isinstance(exc, sdk.APIStatusError):
        return LLMHTTPError(f"{provider} HTTP {exc.status_code}")
    return LLMError(f"{provider} request failed: {type(exc).__name__}")


def parse_json_object(content: object, provider: str) -> dict[str, Any]:
    """The shared contract every provider's reply must meet: a JSON object as text."""
    try:
        if not isinstance(content, str):
            raise TypeError("content must be text")
        result = json.loads(content)
    except (ValueError, TypeError) as exc:
        raise LLMResponseError(f"{provider} reply must contain valid JSON") from exc
    if not isinstance(result, dict):
        raise LLMResponseError(f"{provider} JSON output must be an object")
    return result
