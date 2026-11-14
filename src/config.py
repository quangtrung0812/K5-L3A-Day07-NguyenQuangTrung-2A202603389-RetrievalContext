"""Configuration loader for Day 7 Agentic RAG (reusing Day 6 pgvector)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    database_url: str
    day6_strategy: str
    openai_api_key: str
    embedding_model: str
    agent_model: str


def load_settings(env_path: Path | None = None) -> Settings:
    """Load configuration from .env and environment variables."""
    load_dotenv(dotenv_path=env_path or Path(".env"), override=False)
    return Settings(
        database_url=(os.getenv("DATABASE_URL") or "").strip(),
        day6_strategy=(os.getenv("DAY6_STRATEGY") or "B").strip(),
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip(),
        embedding_model=(os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small").strip(),
        agent_model=(os.getenv("AGENT_MODEL") or "gpt-6-luna").strip(),
    )
