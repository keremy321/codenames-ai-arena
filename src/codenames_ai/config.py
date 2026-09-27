import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"
    room_url: str = ""
    headless: bool = False
    ollama_timeout: float = Field(default=120, gt=0)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:14b"),
            room_url=os.getenv("CODENAMES_ROOM_URL", ""),
            headless=os.getenv("HEADLESS", "false"),
        )
