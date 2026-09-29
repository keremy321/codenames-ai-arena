import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

from codenames_ai.domain.enums import Role, Team
from codenames_ai.llm.base import LLMConfig, Provider


def role_key(team: Team, role: Role) -> str:
    """``blue_spymaster``: names the CLI flags, env variables, and overrides of one role."""
    return f"{team.value}_{role.value}"


class Settings(BaseModel):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"
    room_url: str = ""
    headless: bool = False
    ollama_timeout: float = Field(default=120, gt=0)
    # How long Ollama keeps a model loaded after each request (Ollama syntax: "30m", "-1").
    ollama_keep_alive: str = "30m"
    # Default for every role; an empty model means OLLAMA_MODEL when the provider is ollama.
    llm_provider: Provider = "ollama"
    llm_model: str = ""
    # Per-role overrides keyed by role_key().
    role_providers: dict[str, Provider] = Field(default_factory=dict)
    role_models: dict[str, str] = Field(default_factory=dict)
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    def llm_for(self, team: Team, role: Role) -> LLMConfig:
        key = role_key(team, role)
        provider = self.role_providers.get(key, self.llm_provider)
        model = self.role_models.get(key, "")
        if not model and provider == self.llm_provider:
            model = self.llm_model
        if not model and provider == "ollama":
            model = self.ollama_model
        if not model:
            raise ValueError(
                f"No {provider} model for {team.value} {role.value}: pass "
                f"--{key.replace('_', '-')}-model (or {key.upper()}_MODEL)"
                + (", or --model (LLM_MODEL)" if provider == self.llm_provider else "")
            )
        return LLMConfig(provider=provider, model=model)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        keys = [role_key(team, role) for team in Team for role in Role]
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:14b"),
            ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "").strip() or "30m",
            room_url=os.getenv("CODENAMES_ROOM_URL", ""),
            headless=os.getenv("HEADLESS", "false"),
            llm_provider=os.getenv("LLM_PROVIDER", "").strip().lower() or "ollama",
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            role_providers={
                key: value.strip().lower()
                for key in keys
                if (value := os.getenv(f"{key.upper()}_PROVIDER", "").strip())
            },
            role_models={
                key: value.strip()
                for key in keys
                if (value := os.getenv(f"{key.upper()}_MODEL", "").strip())
            },
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        )
