"""
core/config.py

Centralised, type-safe configuration for Ask Karthik.

All runtime settings are read from environment variables (and from a .env
file when running locally).  pydantic-settings validates types at startup,
so a missing required variable produces a clear error rather than a
confusing KeyError later in the application.

Usage
-----
    from portfolio_chat_ai.core.config import get_settings

    settings = get_settings()
    print(settings.LLM_MODEL)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default origins used when ALLOWED_ORIGINS is not set in the environment.
_DEFAULT_ORIGINS = [
    "https://karthik-007-code.github.io",
    "http://localhost:5173",
    "http://localhost:3000",
]


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    Fields
    ------
    GROQ_API_KEY
        API key for the Groq LLM provider.  Never exposed to the frontend.

    LLM_MODEL
        Model identifier to use (e.g. "llama-3.3-70b-versatile").

    LLM_BASE_URL
        Base URL for the OpenAI-compatible API endpoint.
        Default: Groq's API base.

    LLM_TIMEOUT_SECONDS
        How long (in seconds) to wait for the LLM before giving up.

    ALLOWED_ORIGINS
        Comma-separated list of frontend origins allowed to open WebSocket
        connections.  Stored as a raw string in env/dotenv; accessed as a
        list via the ALLOWED_ORIGINS property.
        Example: "https://myportfolio.com,http://localhost:5173"

    MAX_MESSAGE_LENGTH
        Maximum number of characters accepted in a single visitor message.

    MAX_CONVERSATION_TURNS
        Maximum number of user/assistant back-and-forth pairs per session
        before the conversation is reset to prevent unbounded token usage.

    MAX_MESSAGES_PER_MINUTE
        Rate limit: maximum messages a single session can send per 60 seconds.

    KNOWLEDGE_DIR
        Path to the directory containing the JSON knowledge files.
    """

    model_config = SettingsConfigDict(
        # Look for a .env file starting from the current working directory
        # and walking up — works whether you run `uvicorn` from the repo
        # root or from inside src/.
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore extra fields in .env so the file can carry comments /
        # unrelated vars without crashing startup.
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # LLM provider                                                         #
    # ------------------------------------------------------------------ #
    GROQ_API_KEY: str

    LLM_MODEL: str = "qwen/qwen3.8-27b"

    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"

    LLM_TIMEOUT_SECONDS: int = 30

    # ------------------------------------------------------------------ #
    # Security / CORS                                                      #
    # ------------------------------------------------------------------ #
    # Stored as a raw string in the env / .env file (comma-separated).
    # Parsed into a list by the validator below.
    # Example in .env:
    #   ALLOWED_ORIGINS=https://mysite.com,http://localhost:5173
    ALLOWED_ORIGINS_RAW: str = ",".join(_DEFAULT_ORIGINS)

    @field_validator("ALLOWED_ORIGINS_RAW", mode="before")
    @classmethod
    def _strip_origins(cls, v: object) -> object:
        """Accept any string value as-is; parsing happens in the property."""
        return v

    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        """Return the parsed list of allowed origins."""
        return [
            o.strip()
            for o in self.ALLOWED_ORIGINS_RAW.split(",")
            if o.strip()
        ]

    # ------------------------------------------------------------------ #
    # Conversation limits                                                  #
    # ------------------------------------------------------------------ #
    MAX_MESSAGE_LENGTH: int = 1000  # characters
    MAX_CONVERSATION_TURNS: int = 20  # user+assistant pairs
    MAX_MESSAGES_PER_MINUTE: int = 15

    # ------------------------------------------------------------------ #
    # Paths                                                                #
    # ------------------------------------------------------------------ #
    KNOWLEDGE_DIR: Path = Path("knowledge")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings instance.

    The @lru_cache decorator ensures the .env file is read and validated
    only once — not on every function call.  This keeps config access
    cheap everywhere in the application.
    """
    return Settings()
