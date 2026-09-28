import argparse
import sys
from typing import Self

import pytest
from pydantic import ValidationError

import codenames_ai.config as config_module
import codenames_ai.main as main_module
from codenames_ai.config import Settings
from codenames_ai.domain.enums import Role, Team
from codenames_ai.llm import factory
from codenames_ai.llm.anthropic_client import AnthropicClient
from codenames_ai.llm.base import LLMConfig, LLMError
from codenames_ai.llm.factory import create_llm_client, open_llm_clients
from codenames_ai.llm.ollama import OllamaClient
from codenames_ai.llm.openai_client import OpenAIClient

BS, BO = (Team.BLUE, Role.SPYMASTER), (Team.BLUE, Role.OPERATIVE)
RS, RO = (Team.RED, Role.SPYMASTER), (Team.RED, Role.OPERATIVE)
ROLES = [BS, BO, RS, RO]
OPENAI_KEY = "sk-openai-SECRET"
ANTHROPIC_KEY = "sk-ant-SECRET"


def resolved(settings: Settings) -> dict[tuple[Team, Role], str]:
    return {key: str(settings.llm_for(*key)) for key in ROLES}


def keyed(**overrides: object) -> Settings:
    return Settings(openai_api_key=OPENAI_KEY, anthropic_api_key=ANTHROPIC_KEY, **overrides)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Settings.from_env with only the variables a test sets (no local .env file)."""
    monkeypatch.setattr(config_module, "load_dotenv", lambda: None)
    names = ["LLM_PROVIDER", "LLM_MODEL", "OLLAMA_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    names += [f"{t.value}_{r.value}_{s}".upper() for t, r in ROLES for s in ("PROVIDER", "MODEL")]
    for name in names:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_default_is_the_existing_ollama_model(env: pytest.MonkeyPatch) -> None:
    assert set(resolved(Settings.from_env()).values()) == {"ollama / qwen3:14b"}
    env.setenv("OLLAMA_MODEL", "gemma3:12b")
    assert set(resolved(Settings.from_env()).values()) == {"ollama / gemma3:12b"}


def test_default_provider_and_model_apply_to_every_role() -> None:
    settings = Settings(llm_provider="anthropic", llm_model="claude-x")
    assert set(resolved(settings).values()) == {"anthropic / claude-x"}


def test_one_role_override_keeps_the_others_on_the_default() -> None:
    settings = Settings(
        role_providers={"red_spymaster": "openai"}, role_models={"red_spymaster": "gpt-x"}
    )
    assert resolved(settings) == {
        BS: "ollama / qwen3:14b",
        BO: "ollama / qwen3:14b",
        RS: "openai / gpt-x",
        RO: "ollama / qwen3:14b",
    }


def test_model_only_override_uses_the_default_provider() -> None:
    settings = Settings(role_models={"blue_operative": "gemma3:12b"})
    assert settings.llm_for(*BO) == LLMConfig(provider="ollama", model="gemma3:12b")


def test_provider_override_requires_a_model_unless_ollama() -> None:
    with pytest.raises(ValueError, match="--blue-spymaster-model"):
        Settings(role_providers={"blue_spymaster": "anthropic"}).llm_for(*BS)
    # Back to Ollama from a hosted default: Ollama's own default model applies.
    settings = Settings(
        llm_provider="openai", llm_model="gpt-x", role_providers={"red_operative": "ollama"}
    )
    assert str(settings.llm_for(*RO)) == "ollama / qwen3:14b"
    with pytest.raises(ValueError, match="LLM_MODEL"):
        Settings(llm_provider="openai").llm_for(*RO)


def test_all_four_roles_from_environment(env: pytest.MonkeyPatch) -> None:
    for (team, role), (provider, model) in {
        BS: ("anthropic", "claude-x"),
        BO: ("ollama", "qwen3:14b"),
        RS: ("OpenAI", "gpt-x"),  # provider names are case-insensitive in the environment
        RO: ("ollama", "gemma3:12b"),
    }.items():
        env.setenv(f"{team.value}_{role.value}_PROVIDER".upper(), provider)
        env.setenv(f"{team.value}_{role.value}_MODEL".upper(), model)
    env.setenv("OPENAI_API_KEY", OPENAI_KEY)
    settings = Settings.from_env()
    assert resolved(settings) == {
        BS: "anthropic / claude-x",
        BO: "ollama / qwen3:14b",
        RS: "openai / gpt-x",
        RO: "ollama / gemma3:12b",
    }
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == OPENAI_KEY
    assert settings.anthropic_api_key is None


def test_invalid_provider_in_environment_fails(env: pytest.MonkeyPatch) -> None:
    env.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValidationError):
        Settings.from_env()
    env.setenv("LLM_PROVIDER", "ollama")
    env.setenv("RED_OPERATIVE_PROVIDER", "gemini")
    with pytest.raises(ValidationError):
        Settings.from_env()


def test_factory_selects_provider_and_model() -> None:
    settings = keyed()
    ollama = create_llm_client("ollama", "gemma3:12b", settings)
    openai = create_llm_client("openai", "gpt-x", settings)
    claude = create_llm_client("anthropic", "claude-x", settings)
    assert isinstance(ollama, OllamaClient) and ollama.model == "gemma3:12b"
    assert isinstance(openai, OpenAIClient) and openai.model == "gpt-x"
    assert isinstance(claude, AnthropicClient) and claude.model == "claude-x"
    assert openai._client.api_key == OPENAI_KEY
    assert claude._client.api_key == ANTHROPIC_KEY


def test_factory_rejects_unknown_provider_and_missing_keys() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider 'gemini'"):
        create_llm_client("gemini", "m", keyed())
    with pytest.raises(LLMError, match="OPENAI_API_KEY is not set"):
        create_llm_client("openai", "gpt-x", Settings())
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY is not set"):
        create_llm_client("anthropic", "claude-x", Settings(anthropic_api_key=""))


class FakeClient:
    def __init__(self, provider: str, model: str, log: list[str]) -> None:
        self.name, self.log = f"{provider}/{model}", log

    async def __aenter__(self) -> Self:
        self.log.append(f"open {self.name}")
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.log.append(f"close {self.name}")

    async def chat_json(self, **kwargs: object) -> dict:
        return {}


async def test_clients_per_distinct_config_all_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[str] = []
    monkeypatch.setattr(factory, "create_llm_client", lambda p, m, settings: FakeClient(p, m, log))
    qwen = LLMConfig(provider="ollama", model="qwen3:14b")
    configs = {
        BS: LLMConfig(provider="anthropic", model="claude-x"),
        BO: qwen,
        RS: LLMConfig(provider="openai", model="gpt-x"),
        RO: qwen,
    }
    async with open_llm_clients(configs, Settings()) as clients:
        assert clients[BO] is clients[RO]  # identical stateless config may share
        assert len({id(clients[key]) for key in ROLES}) == 3
        assert [c.name for c in (clients[BS], clients[RS])] == [
            "anthropic/claude-x",
            "openai/gpt-x",
        ]
    assert sorted(entry for entry in log if entry.startswith("close")) == [
        "close anthropic/claude-x",
        "close ollama/qwen3:14b",
        "close openai/gpt-x",
    ]


async def test_clients_opened_before_a_failure_are_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []

    def create(provider: str, model: str, settings: Settings) -> FakeClient:
        if provider == "openai":
            raise LLMError("OPENAI_API_KEY is not set")
        return FakeClient(provider, model, log)

    monkeypatch.setattr(factory, "create_llm_client", create)
    configs = {
        BS: LLMConfig(provider="anthropic", model="claude-x"),
        RS: LLMConfig(provider="openai", model="gpt-x"),
    }
    with pytest.raises(LLMError):
        async with open_llm_clients(configs, Settings()):
            pass
    assert log == ["open anthropic/claude-x", "close anthropic/claude-x"]


async def test_real_clients_are_closed() -> None:
    configs = {
        BS: LLMConfig(provider="anthropic", model="claude-x"),
        BO: LLMConfig(provider="ollama", model="qwen3:14b"),
        RS: LLMConfig(provider="openai", model="gpt-x"),
        RO: LLMConfig(provider="ollama", model="gemma3:12b"),
    }
    async with open_llm_clients(configs, keyed()) as clients:
        pass
    assert clients[BS]._client.is_closed()  # type: ignore[attr-defined]
    assert clients[RS]._client.is_closed()  # type: ignore[attr-defined]
    assert clients[BO]._http.is_closed and clients[RO]._http.is_closed  # type: ignore[attr-defined]


def run_main(monkeypatch: pytest.MonkeyPatch, settings: Settings, *argv: str) -> Settings:
    """Parse a real command line; stop before any browser or model is touched."""
    seen: list[Settings] = []

    async def fake_run(args: argparse.Namespace, settings: Settings) -> None:
        seen.append(settings)

    monkeypatch.setattr(main_module.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main_module, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["codenames_ai.main", *argv])
    assert main_module.main() == 0
    return seen[0]


def test_cli_mixed_providers_for_all_four_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = run_main(
        monkeypatch,
        keyed(),
        "arena",
        "--room", "https://codenames.game/r/test",
        "--blue-spymaster-provider", "anthropic",
        "--blue-spymaster-model", "claude-x",
        "--blue-operative-provider", "ollama",
        "--blue-operative-model", "qwen3:14b",
        "--red-spymaster-provider", "openai",
        "--red-spymaster-model", "gpt-x",
        "--red-operative-provider", "ollama",
        "--red-operative-model", "gemma3:12b",
    )  # fmt: skip
    assert resolved(settings) == {
        BS: "anthropic / claude-x",
        BO: "ollama / qwen3:14b",
        RS: "openai / gpt-x",
        RO: "ollama / gemma3:12b",
    }


def test_cli_default_arena_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = run_main(monkeypatch, Settings(), "arena", "--room", "https://x/r/1")
    assert set(resolved(settings).values()) == {"ollama / qwen3:14b"}


def test_cli_default_then_role_override(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = run_main(
        monkeypatch,
        Settings(role_models={"red_operative": "from-env"}),
        "arena", "--room", "https://x/r/1",
        "--provider", "openai", "--model", "gpt-x",
        "--blue-operative-model", "gpt-y",
    )  # fmt: skip
    assert resolved(settings) == {
        BS: "openai / gpt-x",
        BO: "openai / gpt-y",
        RS: "openai / gpt-x",
        RO: "openai / from-env",  # environment role override survives CLI defaults
    }


def test_cli_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["m", "arena", "--red-operative-provider", "gemini"])
    with pytest.raises(SystemExit) as info:
        main_module.main()
    assert info.value.code == 2


def test_startup_diagnostics_list_roles_without_keys() -> None:
    settings = keyed(
        role_providers={"blue_spymaster": "anthropic", "red_spymaster": "openai"},
        role_models={
            "blue_spymaster": "claude-x",
            "red_spymaster": "gpt-x",
            "red_operative": "gemma3:12b",
        },
    )
    text = main_module.format_llms({key: settings.llm_for(*key) for key in ROLES})
    assert text.splitlines() == [
        "BLUE SPYMASTER:  anthropic / claude-x",
        "BLUE OPERATIVE:  ollama / qwen3:14b",
        "RED SPYMASTER:   openai / gpt-x",
        "RED OPERATIVE:   ollama / gemma3:12b",
    ]
    for shown in (text, repr(settings), str(settings)):
        assert OPENAI_KEY not in shown and ANTHROPIC_KEY not in shown
