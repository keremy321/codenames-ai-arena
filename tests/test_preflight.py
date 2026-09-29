import argparse
import contextlib
import json
import re
from collections.abc import AsyncIterator

import httpx
import pytest

import codenames_ai.main as main_module
from codenames_ai.config import Settings
from codenames_ai.llm import preflight
from codenames_ai.llm.base import LLMConfig, LLMError, LLMHTTPError, LLMModelError
from codenames_ai.llm.ollama import OllamaClient, OllamaModelError
from codenames_ai.llm.preflight import (
    fetch_loaded_ollama_models,
    fetch_ollama_models,
    llm_preflight,
    warm_ollama_model,
)

QWEN = LLMConfig(provider="ollama", model="qwen3:14b")
GEMMA = LLMConfig(provider="ollama", model="gemma3:12b")


class FakeProcess:
    def __init__(self, args: list[str], *, env: dict[str, str], **kwargs: object) -> None:
        self.args, self.env = args, env
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.returncode = -9


class Ollama:
    """Scripted /api/tags answers (None = unreachable) plus any ``ollama serve`` started,
    the models warmed (in order) and the models /api/ps reports as loaded."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *answers: set[str] | None,
        loaded: set[str] | None = None,
        broken: str | None = None,
    ) -> None:
        self.answers = list(answers)
        self.urls: list[str] = []
        self.started: list[FakeProcess] = []
        self.warmed: list[str] = []
        self.events: list[str] = []  # shared order of warm-ups and browser starts

        async def warm(settings: Settings, model: str) -> None:
            self.warmed.append(model)
            self.events.append(f"warm {model}")
            if model == broken:
                raise OllamaModelError("model requires more system memory than is available")

        async def ps(base_url: str, **kwargs: object) -> set[str] | None:
            return loaded

        monkeypatch.setattr(preflight, "warm_ollama_model", warm)
        monkeypatch.setattr(preflight, "fetch_loaded_ollama_models", ps)

        async def fetch(base_url: str, **kwargs: object) -> set[str] | None:
            self.urls.append(base_url)
            return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]

        def popen(args: list[str], **kwargs: object) -> FakeProcess:
            process = FakeProcess(args, **kwargs)  # type: ignore[arg-type]
            self.started.append(process)
            return process

        monkeypatch.setattr(preflight, "fetch_ollama_models", fetch)
        monkeypatch.setattr(preflight.subprocess, "Popen", popen)
        monkeypatch.setattr(preflight.shutil, "which", lambda name: "/bin/ollama")


async def run_preflight(settings: Settings, *configs: LLMConfig) -> list[str]:
    lines: list[str] = []
    async with llm_preflight(configs, settings, report=lines.append):
        lines.append("<match>")
    return lines


async def test_fetch_lists_models_and_reports_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen3:14b"}]})

    transport = httpx.MockTransport(handler)
    assert await fetch_ollama_models("http://localhost:11434", transport=transport) == {"qwen3:14b"}

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed")

    refused = httpx.MockTransport(refuse)
    assert await fetch_ollama_models("http://localhost:11434", transport=refused) is None


async def test_running_ollama_is_used_and_never_restarted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ollama = Ollama(monkeypatch, {"qwen3:14b", "gemma3:12b"})
    lines = await run_preflight(Settings(), QWEN, GEMMA, QWEN)
    assert not ollama.started
    assert lines[0] == "Ollama ready: gemma3:12b, qwen3:14b"
    assert lines[-1] == "<match>"
    assert ollama.urls == ["http://localhost:11434"]  # one probe for all Ollama roles


async def test_local_ollama_is_started_once_and_stopped_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ollama = Ollama(monkeypatch, None, None, {"qwen3:latest"})
    settings = Settings(ollama_base_url="http://127.0.0.1:11500")
    lines: list[str] = []
    async with llm_preflight(
        [LLMConfig(provider="ollama", model="qwen3")], settings, report=lines.append
    ):
        (process,) = ollama.started
        assert process.args == ["/bin/ollama", "serve"]
        assert process.env["OLLAMA_HOST"] == "127.0.0.1:11500"
        assert not process.terminated  # alive for the whole match
    assert process.terminated
    assert "Ollama ready: qwen3" in lines
    assert ollama.warmed == ["qwen3"]


async def test_unreachable_local_ollama_without_executable_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ollama = Ollama(monkeypatch, None)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    with pytest.raises(LLMHTTPError, match="no `ollama` executable"):
        await run_preflight(Settings(), QWEN)
    assert not ollama.started


async def test_remote_ollama_is_never_launched_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    ollama = Ollama(monkeypatch, None)
    settings = Settings(ollama_base_url="http://gpu-box.lan:11434")
    with pytest.raises(LLMHTTPError, match="remote Ollama server at http://gpu-box.lan:11434"):
        await run_preflight(settings, QWEN)
    assert not ollama.started


async def test_started_ollama_that_exits_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    ollama = Ollama(monkeypatch, None)
    original = preflight.subprocess.Popen

    def crashing(args: list[str], **kwargs: object) -> FakeProcess:
        process = original(args, **kwargs)
        process.returncode = 1
        return process

    monkeypatch.setattr(preflight.subprocess, "Popen", crashing)
    with pytest.raises(LLMHTTPError, match="exited with code 1"):
        await run_preflight(Settings(), QWEN)
    assert len(ollama.started) == 1


async def test_missing_model_names_the_pull_command(monkeypatch: pytest.MonkeyPatch) -> None:
    Ollama(monkeypatch, {"qwen3:14b"})
    with pytest.raises(LLMModelError, match="Run: ollama pull gemma3:12b"):
        await run_preflight(Settings(), QWEN, GEMMA)


async def test_started_ollama_is_stopped_when_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ollama = Ollama(monkeypatch, None, set())
    with pytest.raises(LLMModelError):
        await run_preflight(Settings(), QWEN)
    assert ollama.started[0].terminated


async def test_hosted_providers_are_checked_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ollama = Ollama(monkeypatch, set())
    openai = LLMConfig(provider="openai", model="gpt-x")
    lines = await run_preflight(Settings(openai_api_key="sk-openai-SECRET"), openai)
    assert lines == ["OpenAI configured: gpt-x", "<match>"]
    assert not ollama.urls  # no Ollama role, no probe
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY is not set"):
        await run_preflight(Settings(), LLMConfig(provider="anthropic", model="claude-x"))
    assert "sk-openai-SECRET" not in "".join(lines)


async def test_ollama_failure_stops_arena_before_any_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Ollama(monkeypatch, None)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)

    class NoBrowser:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("a browser opened before the LLM preflight passed")

    monkeypatch.setattr(main_module, "BrowserClient", NoBrowser)
    args = argparse.Namespace(room=None, host_nickname="ArenaHost", wait_seconds=5)
    with pytest.raises(LLMHTTPError, match="Cannot reach Ollama"):
        await main_module.run_arena(args, Settings())


WARM_LINE = re.compile(r"^Ollama warm-up: (\S+) \.\.\. \d+\.\ds( \(already loaded\))?$")


async def test_each_distinct_ollama_model_warms_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ollama = Ollama(monkeypatch, {"qwen3:latest", "gemma3:12b"})
    configs = (
        LLMConfig(provider="ollama", model="qwen3"),
        LLMConfig(provider="ollama", model="qwen3:latest"),
        GEMMA,
        GEMMA,
        LLMConfig(provider="openai", model="gpt-x"),
    )
    lines = await run_preflight(Settings(openai_api_key="sk-test-key-123"), *configs)
    # "qwen3" is "qwen3:latest": one model, one warm-up; hosted models are never called.
    assert sorted(ollama.warmed) == ["gemma3:12b", "qwen3"]
    warm_lines = [line for line in lines if line.startswith("Ollama warm-up")]
    assert len(warm_lines) == 2 and all(WARM_LINE.match(line) for line in warm_lines)
    # Warm-ups come after the reachability/model checks and before the match.
    assert lines.index("Ollama ready: gemma3:12b, qwen3") < lines.index(warm_lines[0])
    assert lines.index(warm_lines[-1]) < lines.index("<match>")


async def test_warmups_report_already_loaded_models_and_are_yielded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Ollama(monkeypatch, {"qwen3:14b", "gemma3:12b"}, loaded={"qwen3:14b"})
    async with llm_preflight([QWEN, GEMMA], Settings(), report=lambda line: None) as warmups:
        assert {(w.model, w.already_loaded) for w in warmups} == {
            ("qwen3:14b", True),
            ("gemma3:12b", False),
        }
        assert all(w.seconds >= 0 for w in warmups)


async def test_warmup_failure_stops_preflight_and_stops_started_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ollama = Ollama(monkeypatch, None, {"qwen3:14b"}, broken="qwen3:14b")
    with pytest.raises(LLMModelError, match="qwen3:14b failed its warm-up request.*memory"):
        await run_preflight(Settings(), QWEN)
    assert ollama.started[0].terminated


async def test_running_ollama_is_warmed_without_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    ollama = Ollama(monkeypatch, {"qwen3:14b"}, loaded={"qwen3:14b"})
    lines = await run_preflight(Settings(), QWEN)
    assert not ollama.started
    assert ollama.warmed == ["qwen3:14b"]
    assert lines[1].endswith("(already loaded)")


@contextlib.asynccontextmanager
async def no_clients(*args: object) -> AsyncIterator[dict]:
    yield {}


def browser_recorder(events: list[str]) -> object:
    def open_browser(**kwargs: object) -> object:
        events.append("browser")
        raise RuntimeError("browser reached")

    return open_browser


async def test_warmup_happens_before_any_browser_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    ollama = Ollama(monkeypatch, {"qwen3:14b"})
    monkeypatch.setattr(main_module, "BrowserClient", browser_recorder(ollama.events))
    monkeypatch.setattr(main_module, "open_llm_clients", no_clients)
    args = argparse.Namespace(room=None, host_nickname="ArenaHost", wait_seconds=5)
    with pytest.raises(RuntimeError, match="browser reached"):
        await main_module.run_arena(args, Settings())
    assert ollama.events == ["warm qwen3:14b", "browser"]


async def test_warmup_failure_stops_arena_before_any_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ollama = Ollama(monkeypatch, {"qwen3:14b"}, broken="qwen3:14b")
    monkeypatch.setattr(main_module, "BrowserClient", browser_recorder(ollama.events))
    monkeypatch.setattr(main_module, "open_llm_clients", no_clients)
    args = argparse.Namespace(room=None, host_nickname="ArenaHost", wait_seconds=5)
    with pytest.raises(LLMModelError, match="warm-up"):
        await main_module.run_arena(args, Settings())
    assert ollama.events == ["warm qwen3:14b"]


async def test_warm_request_uses_match_keep_alive_and_one_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "OK"}})

    class Routed(OllamaClient):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(preflight, "OllamaClient", Routed)
    await warm_ollama_model(Settings(ollama_keep_alive="45m"), "gemma3:12b")
    await warm_ollama_model(Settings(ollama_keep_alive="-1"), "gemma3:12b")
    first, second = bodies
    assert first["model"] == "gemma3:12b" and first["keep_alive"] == "45m"
    assert second["keep_alive"] == -1  # whole seconds go to Ollama as a number
    # Same option names as gameplay (no num_ctx): Ollama must not reload for the match.
    assert first["options"] == {"num_predict": 1}
    assert first["think"] is False and first["stream"] is False
    assert "format" not in first


async def test_loaded_models_come_from_api_ps_and_errors_mean_unknown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(200, json={"models": [{"name": "qwen3:14b"}]})

    ok = httpx.MockTransport(handler)
    assert await fetch_loaded_ollama_models("http://localhost:11434", transport=ok) == {"qwen3:14b"}
    old = httpx.MockTransport(lambda request: httpx.Response(404, text="404 page not found"))
    assert await fetch_loaded_ollama_models("http://localhost:11434", transport=old) is None
