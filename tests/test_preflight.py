import argparse

import httpx
import pytest

import codenames_ai.main as main_module
from codenames_ai.config import Settings
from codenames_ai.llm import preflight
from codenames_ai.llm.base import LLMConfig, LLMError, LLMHTTPError, LLMModelError
from codenames_ai.llm.preflight import fetch_ollama_models, llm_preflight

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
    """Scripted /api/tags answers (None = unreachable) plus any ``ollama serve`` started."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *answers: set[str] | None) -> None:
        self.answers = list(answers)
        self.urls: list[str] = []
        self.started: list[FakeProcess] = []

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
    assert lines == ["Ollama ready: gemma3:12b, qwen3:14b", "<match>"]
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
    assert lines[-1] == "Ollama ready: qwen3"


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
