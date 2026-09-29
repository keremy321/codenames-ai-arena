import json

import httpx
import pytest

from codenames_ai.config import Settings
from codenames_ai.llm.factory import create_llm_client
from codenames_ai.llm.ollama import (
    OllamaClient,
    OllamaHTTPError,
    OllamaModelError,
    OllamaResponseError,
    OllamaTimeoutError,
    keep_alive_value,
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


async def test_keep_alive_is_sent_with_every_request_when_configured() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    transport = httpx.MockTransport(handler)
    async with OllamaClient(keep_alive="30m", transport=transport) as client:
        await client.chat_json(system="s", user="u")
        await client.warm_up()
    async with OllamaClient(transport=transport) as client:
        await client.chat_json(system="s", user="u")
    assert [body.get("keep_alive") for body in bodies] == ["30m", "30m", None]


async def test_warm_up_that_cannot_run_the_model_fails() -> None:
    answer = {"error": "model requires more system memory (18 GiB) than is available"}
    transport = httpx.MockTransport(lambda request: httpx.Response(500, json=answer))
    async with OllamaClient(transport=transport) as client:
        with pytest.raises(OllamaModelError, match="system memory"):
            await client.warm_up()


def test_factory_and_settings_pass_keep_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "-1")
    settings = Settings.from_env()
    assert settings.ollama_keep_alive == "-1"
    client = create_llm_client("ollama", "gemma3:12b", settings)
    assert isinstance(client, OllamaClient) and client.keep_alive == -1
    default = create_llm_client("ollama", "gemma3:12b", Settings())
    assert default.keep_alive == "30m"  # type: ignore[union-attr]
    assert keep_alive_value(" 3600 ") == 3600 and keep_alive_value("1h") == "1h"
