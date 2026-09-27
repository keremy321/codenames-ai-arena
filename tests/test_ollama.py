import json

import httpx
import pytest

from codenames_ai.llm.ollama import (
    OllamaClient,
    OllamaHTTPError,
    OllamaModelError,
    OllamaResponseError,
    OllamaTimeoutError,
)


async def test_request_and_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["model"] == "qwen3:14b"
        assert body["stream"] is False
        assert body["format"] == {"type": "object"}
        assert len(body["messages"]) == 2
        return httpx.Response(200, json={"message": {"content": '{"status":"ok"}'}})

    async with OllamaClient(transport=httpx.MockTransport(handler)) as client:
        assert await client.chat_json(system="test", user="test", schema={"type": "object"}) == {
            "status": "ok"
        }


@pytest.mark.parametrize(
    "payload",
    [
        {"message": {"content": "not json"}},
        {"message": {"content": "[]"}},
        {"message": {"content": "```json\n{}\n```"}},
        {"message": {"content": {}}},
        {},
        [],
    ],
)
async def test_malformed_json(payload: object) -> None:
    async with OllamaClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(OllamaResponseError):
            await client.chat_json(system="test", user="test")


@pytest.mark.parametrize(
    "status,payload,exception",
    [
        (404, {"error": "model not found"}, OllamaModelError),
        (200, {"error": "model failed"}, OllamaModelError),
        (503, {}, OllamaHTTPError),
    ],
)
async def test_server_errors(status: int, payload: object, exception: type[Exception]) -> None:
    async with OllamaClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json=payload))
    ) as client:
        with pytest.raises(exception):
            await client.chat_json(system="test", user="test")


@pytest.mark.parametrize(
    "error,expected",
    [
        (httpx.ReadTimeout, OllamaTimeoutError),
        (httpx.ConnectError, OllamaHTTPError),
    ],
)
async def test_transport_errors(error: type[httpx.RequestError], expected: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("test failure", request=request)

    async with OllamaClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(expected):
            await client.chat_json(system="test", user="test")


async def test_non_json_envelope() -> None:
    async with OllamaClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="not json"))
    ) as client:
        with pytest.raises(OllamaResponseError, match="envelope"):
            await client.chat_json(system="test", user="test")


async def test_temperature_is_sent_only_when_requested() -> None:
    options: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        options.append(json.loads(request.content)["options"])
        return httpx.Response(200, json={"message": {"content": "{}"}})

    async with OllamaClient(transport=httpx.MockTransport(handler)) as client:
        await client.chat_json(system="s", user="u", num_predict=50)
        await client.chat_json(system="s", user="u", temperature=0.0)
    assert options == [{"num_predict": 50}, {"num_predict": 192, "temperature": 0.0}]
