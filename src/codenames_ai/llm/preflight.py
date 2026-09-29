"""Validate every configured model before any game browser opens.

Ollama is probed over HTTP (and started locally when it is simply not running), then
each distinct Ollama model is warmed with one tiny request so the first spymaster call
does not pay for loading it. Hosted providers are checked locally only, so startup
never spends paid tokens.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import time
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from codenames_ai.config import Settings

from .base import LLMConfig, LLMError, LLMHTTPError, LLMModelError
from .factory import api_key
from .ollama import OllamaClient, keep_alive_value

logger = logging.getLogger(__name__)

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
OLLAMA_START_TIMEOUT = 30.0
# Loading a large model from disk can take minutes on a cold start.
OLLAMA_WARMUP_TIMEOUT = 300.0
PROVIDER_NAMES = {"openai": "OpenAI", "anthropic": "Anthropic"}


async def fetch_ollama_models(
    base_url: str, *, timeout: float = 3.0, transport: httpx.AsyncBaseTransport | None = None
) -> set[str] | None:
    """Installed model names, or None when no Ollama server answers at ``base_url``."""
    try:
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/", timeout=timeout, transport=transport
        ) as http:
            response = await http.get("api/tags")
    except httpx.RequestError:
        return None
    if response.is_error:
        raise LLMHTTPError(f"Ollama at {base_url} answered HTTP {response.status_code}")
    try:
        models = response.json().get("models", [])
        return {str(m.get("name") or m.get("model")) for m in models if isinstance(m, dict)}
    except (ValueError, AttributeError) as exc:
        raise LLMHTTPError(f"Ollama at {base_url} sent an unexpected model list") from exc


def ollama_name(model: str) -> str:
    """Ollama's full model name: "qwen3" means "qwen3:latest"."""
    return model if ":" in model else f"{model}:latest"


def installed(model: str, names: set[str]) -> bool:
    return ollama_name(model) in names or model in names


async def fetch_loaded_ollama_models(
    base_url: str, *, timeout: float = 3.0, transport: httpx.AsyncBaseTransport | None = None
) -> set[str] | None:
    """Models Ollama currently holds in memory (api/ps), or None when unknown."""
    try:
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/", timeout=timeout, transport=transport
        ) as http:
            response = await http.get("api/ps")
        models = response.json().get("models", []) if not response.is_error else None
        if not isinstance(models, list):
            return None
        return {str(m.get("name") or m.get("model")) for m in models if isinstance(m, dict)}
    except (httpx.RequestError, ValueError, AttributeError):
        return None


@dataclass(frozen=True)
class WarmUp:
    model: str
    seconds: float
    already_loaded: bool | None  # None: the server did not say


async def warm_ollama_model(settings: Settings, model: str) -> None:
    """One tiny inference with the match's keep_alive; raises if the model cannot run."""
    async with OllamaClient(
        settings.ollama_base_url,
        model,
        timeout=max(settings.ollama_timeout, OLLAMA_WARMUP_TIMEOUT),
        keep_alive=keep_alive_value(settings.ollama_keep_alive),
    ) as client:
        await client.warm_up()


async def warm_ollama(
    models: Iterable[str], settings: Settings, report: Callable[[str], None]
) -> list[WarmUp]:
    """Warm each distinct model once, one after another (never two loads at once)."""
    loaded = await fetch_loaded_ollama_models(settings.ollama_base_url)
    results: list[WarmUp] = []
    for model in dict.fromkeys(models):
        already = None if loaded is None else installed(model, loaded)
        started = time.perf_counter()
        try:
            await warm_ollama_model(settings, model)
        except LLMError as exc:
            raise LLMModelError(f"Ollama model {model} failed its warm-up request: {exc}") from exc
        seconds = time.perf_counter() - started
        results.append(WarmUp(model, seconds, already))
        note = " (already loaded)" if already else ""
        report(f"Ollama warm-up: {model} ... {seconds:.1f}s{note}")
        logger.debug(
            "Ollama warm-up: model=%s seconds=%.2f already_loaded=%s", model, seconds, already
        )
    return results


def is_local(base_url: str) -> bool:
    return urlparse(base_url).hostname in LOCAL_HOSTS


def start_ollama(base_url: str) -> subprocess.Popen[bytes]:
    """Start ``ollama serve`` for a local URL; never called for a remote server."""
    executable = shutil.which("ollama")
    if executable is None:
        raise LLMHTTPError(
            f"Cannot reach Ollama at {base_url} and no `ollama` executable is on PATH. "
            "Install Ollama or start it, then retry."
        )
    parsed = urlparse(base_url)
    host = f"[{parsed.hostname}]" if ":" in str(parsed.hostname) else parsed.hostname
    env = {**os.environ, "OLLAMA_HOST": f"{host}:{parsed.port or 11434}"}
    logger.info("Starting %s serve for %s", executable, base_url)
    return subprocess.Popen(
        [executable, "serve"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_ollama(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


async def wait_for_ollama(
    base_url: str, process: subprocess.Popen[bytes], *, timeout: float = OLLAMA_START_TIMEOUT
) -> set[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        names = await fetch_ollama_models(base_url, timeout=1.0)
        if names is not None:
            return names
        if process.poll() is not None:
            raise LLMHTTPError(
                f"`ollama serve` exited with code {process.returncode} before {base_url} "
                "answered; run `ollama serve` yourself to see why."
            )
        await asyncio.sleep(0.25)
    raise LLMHTTPError(f"Started `ollama serve`, but {base_url} did not answer in {timeout:.0f}s")


@asynccontextmanager
async def llm_preflight(
    configs: Iterable[LLMConfig], settings: Settings, *, report: Callable[[str], None] = print
) -> AsyncIterator[list[WarmUp]]:
    """Fail before the match on an unreachable Ollama, a missing model, a model that
    cannot run, or a missing key. Yields the Ollama warm-ups (for the match log).

    Ollama started here is stopped on exit; an already running server is left alone.
    """
    distinct = sorted(set(configs), key=str)
    started: subprocess.Popen[bytes] | None = None
    try:
        # "qwen3" and "qwen3:latest" are one model: probed and warmed once.
        spellings: dict[str, str] = {}
        for config in distinct:
            if config.provider == "ollama":
                spellings.setdefault(ollama_name(config.model), config.model)
        ollama = list(spellings.values())
        if ollama:
            base_url = settings.ollama_base_url
            names = await fetch_ollama_models(base_url)
            if names is None:
                if not is_local(base_url):
                    raise LLMHTTPError(
                        f"Cannot reach the remote Ollama server at {base_url}; "
                        "start it there or fix OLLAMA_BASE_URL."
                    )
                report(f"Ollama is not running at {base_url}; starting `ollama serve`...")
                started = start_ollama(base_url)
                names = await wait_for_ollama(base_url, started)
            missing = [model for model in ollama if not installed(model, names)]
            if missing:
                pulls = "; ".join(f"ollama pull {model}" for model in missing)
                raise LLMModelError(
                    f"Ollama model not installed: {', '.join(missing)}. Run: {pulls}"
                )
            report(f"Ollama ready: {', '.join(ollama)}")
        for config in distinct:
            if config.provider != "ollama":
                api_key(config.provider, settings)
                report(f"{PROVIDER_NAMES[config.provider]} configured: {config.model}")
        yield await warm_ollama(ollama, settings, report) if ollama else []
    finally:
        if started is not None:
            stop_ollama(started)
