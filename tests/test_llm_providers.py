import json
import logging
from collections.abc import Callable

import anthropic
import httpx2
import openai
import pytest

from codenames_ai.llm.anthropic_client import AnthropicClient, anthropic_schema
from codenames_ai.llm.base import (
    HOSTED_MAX_OUTPUT_TOKENS,
    LLMError,
    LLMHTTPError,
    LLMModelError,
    LLMResponseError,
    LLMTimeoutError,
)
from codenames_ai.llm.ollama import (
    OllamaHTTPError,
    OllamaModelError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from codenames_ai.llm.openai_client import OpenAIClient

SECRET = "sk-test-SECRET-0123456789"
SCHEMA = {
    "type": "object",
    "properties": {
        "rankings": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "string", "pattern": "^[A-Z]+$", "maxLength": 20},
        },
        # A board word that happens to spell a schema keyword stays a property.
        "maxItems": {"type": "string"},
    },
    "required": ["rankings", "maxItems"],
    "additionalProperties": False,
}

Handler = Callable[[httpx2.Request], httpx2.Response]


def openai_reply(content: str | None, finish: str = "stop", refusal: str | None = None) -> dict:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
    }


def anthropic_reply(*blocks: dict, stop: str = "end_turn") -> dict:
    return {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "model": "m",
        "content": list(blocks),
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def error_body(message: str) -> dict:
    return {"type": "error", "error": {"type": "invalid_request_error", "message": message}}


def openai_client(handler: Handler, model: str = "openai-model") -> OpenAIClient:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    sdk = openai.AsyncOpenAI(api_key=SECRET, http_client=http, max_retries=0)
    return OpenAIClient(model, client=sdk)


def anthropic_client(handler: Handler, model: str = "claude-model") -> AnthropicClient:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    sdk = anthropic.AsyncAnthropic(api_key=SECRET, http_client=http, max_retries=0)
    return AnthropicClient(model, client=sdk)


def test_ollama_errors_are_common_llm_errors() -> None:
    assert issubclass(OllamaResponseError, LLMResponseError)
    assert issubclass(OllamaHTTPError, LLMHTTPError)
    assert issubclass(OllamaTimeoutError, LLMTimeoutError)
    assert issubclass(OllamaModelError, LLMModelError)


async def test_openai_request_and_json_response() -> None:
    requests: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=openai_reply('{"status": "ok"}'))

    async with openai_client(handler) as client:
        reply = await client.chat_json(
            system="sys", user="usr", schema=SCHEMA, num_predict=100, temperature=0.0
        )
    assert reply == {"status": "ok"}
    (body,) = requests
    assert body["model"] == "openai-model"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "reply", "schema": SCHEMA, "strict": True},
    }
    assert body["max_completion_tokens"] == HOSTED_MAX_OUTPUT_TOKENS
    assert body["temperature"] == 0.0


async def test_openai_without_schema_uses_json_mode() -> None:
    requests: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=openai_reply('{"a": 1}'))

    async with openai_client(handler) as client:
        assert await client.chat_json(system="Return JSON", user="u") == {"a": 1}
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "temperature" not in requests[0]


@pytest.mark.parametrize(
    "reply",
    [
        openai_reply("not json"),
        openai_reply("[]"),
        openai_reply('{"a": 1', finish="length"),
        openai_reply(None, refusal="no"),
        {**openai_reply("{}"), "choices": []},
    ],
)
async def test_openai_unusable_reply_is_response_error(reply: dict) -> None:
    async with openai_client(lambda request: httpx2.Response(200, json=reply)) as client:
        with pytest.raises(LLMResponseError):
            await client.chat_json(system="s", user="u", schema=SCHEMA)


async def test_openai_drops_rejected_temperature_once() -> None:
    requests: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        if "temperature" in body:
            return httpx2.Response(
                400, json=error_body("Unsupported value: 'temperature' does not support 0.0")
            )
        return httpx2.Response(200, json=openai_reply('{"ok": true}'))

    async with openai_client(handler) as client:
        assert await client.chat_json(system="s", user="u", temperature=0.0) == {"ok": True}
        assert await client.chat_json(system="s", user="u", temperature=0.4) == {"ok": True}
    assert ["temperature" in body for body in requests] == [True, False, False]


@pytest.mark.parametrize(
    "status,exception",
    [
        (401, LLMHTTPError),
        (403, LLMHTTPError),
        (404, LLMModelError),
        (400, LLMModelError),
        (500, LLMHTTPError),
    ],
)
async def test_openai_status_errors_become_llm_errors(
    status: int, exception: type[Exception], caplog: pytest.LogCaptureFixture
) -> None:
    # Providers echo (part of) a rejected key; it must never reach our errors or logs.
    body = error_body(f"Incorrect API key provided: {SECRET}")
    async with openai_client(lambda request: httpx2.Response(status, json=body)) as client:
        with caplog.at_level(logging.DEBUG), pytest.raises(exception) as info:
            await client.chat_json(system="s", user="u")
    if status in (401, 403):
        assert SECRET not in str(info.value)
    assert SECRET not in caplog.text


async def test_openai_connection_and_timeout_errors() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow")

    async with openai_client(refuse) as client:
        with pytest.raises(LLMHTTPError):
            await client.chat_json(system="s", user="u")
    async with openai_client(slow) as client:
        with pytest.raises(LLMTimeoutError):
            await client.chat_json(system="s", user="u")


async def test_anthropic_request_and_json_response() -> None:
    requests: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json=anthropic_reply(
                {"type": "thinking", "thinking": "", "signature": "x"},
                {"type": "text", "text": '{"status": "ok"}'},
            ),
        )

    async with anthropic_client(handler) as client:
        reply = await client.chat_json(system="sys", user="usr", schema=SCHEMA, temperature=0.4)
    assert reply == {"status": "ok"}
    (body,) = requests
    assert body["model"] == "claude-model"
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "usr"}]
    assert body["max_tokens"] == HOSTED_MAX_OUTPUT_TOKENS
    assert "temperature" not in body  # current Claude models reject sampling parameters
    sent = body["output_config"]["format"]
    assert sent["type"] == "json_schema"
    assert sent["schema"] == anthropic_schema(SCHEMA)


def test_anthropic_schema_drops_only_unsupported_constraints() -> None:
    relaxed = anthropic_schema(SCHEMA)
    rankings = relaxed["properties"]["rankings"]
    assert "minItems" not in rankings and "maxItems" not in rankings
    assert rankings["items"] == {"type": "string", "pattern": "^[A-Z]+$"}
    assert relaxed["properties"]["maxItems"] == {"type": "string"}
    assert relaxed["required"] == ["rankings", "maxItems"]
    assert relaxed["additionalProperties"] is False
    assert SCHEMA["properties"]["rankings"]["maxItems"] == 2  # the agent's schema is untouched


@pytest.mark.parametrize(
    "reply",
    [
        anthropic_reply({"type": "text", "text": "not json"}),
        anthropic_reply({"type": "text", "text": "[1]"}),
        anthropic_reply({"type": "text", "text": '{"a":'}, stop="max_tokens"),
        anthropic_reply(stop="refusal"),
    ],
)
async def test_anthropic_unusable_reply_is_response_error(reply: dict) -> None:
    async with anthropic_client(lambda request: httpx2.Response(200, json=reply)) as client:
        with pytest.raises(LLMResponseError):
            await client.chat_json(system="s", user="u", schema=SCHEMA)


@pytest.mark.parametrize(
    "status,exception",
    [(401, LLMHTTPError), (404, LLMModelError), (400, LLMModelError), (529, LLMHTTPError)],
)
async def test_anthropic_status_errors_become_llm_errors(
    status: int, exception: type[Exception]
) -> None:
    body = error_body(f"invalid x-api-key {SECRET}")
    async with anthropic_client(lambda request: httpx2.Response(status, json=body)) as client:
        with pytest.raises(exception) as info:
            await client.chat_json(system="s", user="u")
    assert isinstance(info.value, LLMError)
    if status == 401:
        assert SECRET not in str(info.value)


@pytest.mark.parametrize("make", [openai_client, anthropic_client])
async def test_hosted_clients_close_their_sdk_client(make: Callable[..., object]) -> None:
    client = make(lambda request: httpx2.Response(500))
    async with client:  # type: ignore[attr-defined]
        assert not client._client.is_closed()  # type: ignore[attr-defined]
    assert client._client.is_closed()  # type: ignore[attr-defined]
