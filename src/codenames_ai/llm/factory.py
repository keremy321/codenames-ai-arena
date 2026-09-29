from collections.abc import AsyncIterator, Hashable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager

from codenames_ai.config import Settings

from .anthropic_client import AnthropicClient
from .base import PROVIDERS, LLMClient, LLMConfig, LLMError
from .ollama import OllamaClient, keep_alive_value
from .openai_client import OpenAIClient


def create_llm_client(
    provider: str, model: str, settings: Settings
) -> OllamaClient | OpenAIClient | AnthropicClient:
    """A new model-bound client; the caller closes it (``async with``)."""
    if provider == "ollama":
        return OllamaClient(
            settings.ollama_base_url,
            model,
            timeout=settings.ollama_timeout,
            keep_alive=keep_alive_value(settings.ollama_keep_alive),
        )
    if provider == "openai":
        return OpenAIClient(model, api_key=api_key(provider, settings))
    if provider == "anthropic":
        return AnthropicClient(model, api_key=api_key(provider, settings))
    raise ValueError(f"Unknown LLM provider {provider!r}; choose one of: {', '.join(PROVIDERS)}")


API_KEY_VARIABLES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def api_key(provider: str, settings: Settings) -> str:
    """The hosted provider's key; checked locally, so no request (or token) is spent."""
    key = {"openai": settings.openai_api_key, "anthropic": settings.anthropic_api_key}[provider]
    if key is None or not key.get_secret_value():
        raise LLMError(f"{API_KEY_VARIABLES[provider]} is not set")
    return key.get_secret_value()


@asynccontextmanager
async def open_llm_clients[K: Hashable](
    configs: Mapping[K, LLMConfig], settings: Settings
) -> AsyncIterator[dict[K, LLMClient]]:
    """One client per distinct configuration, all closed on exit.

    Roles with an identical configuration share a client: clients hold no conversation
    history, so sharing a transport never shares information between roles.
    """
    async with AsyncExitStack() as stack:
        clients: dict[LLMConfig, LLMClient] = {}
        for config in configs.values():
            if config not in clients:
                client = create_llm_client(config.provider, config.model, settings)
                clients[config] = await stack.enter_async_context(client)
        yield {key: clients[config] for key, config in configs.items()}
