"""Typed application configuration loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Gemini
    google_api_key: str = ""
    gemini_chat_model: str = "gemini-2.5-flash"
    # 'pro' is unavailable on Gemini free tier (quota 0). Default reasoning to
    # flash; override GEMINI_REASONING_MODEL=gemini-2.5-pro on a paid key.
    gemini_reasoning_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "models/gemini-embedding-001"
    embed_dim: int = 3072

    # Qdrant (cloud or local). Set qdrant_api_key for Qdrant Cloud.
    qdrant_url: str = "http://localhost:6533"
    qdrant_api_key: str = ""
    qdrant_collection: str = "documents"

    # Postgres
    database_url: str = "postgresql://a2z:a2z@localhost:5442/a2z_hunter"

    # Retrieval / graph tuning
    retrieval_top_k: int = 20
    rerank_top_n: int = 6
    score_threshold: float = 0.45
    max_attempts: int = 2

    # Web search
    web_search_provider: str = "duckduckgo"  # duckduckgo | tavily
    tavily_api_key: str = ""

    # Models
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    sparse_model: str = "Qdrant/bm25"

    @property
    def dense_vector_name(self) -> str:
        return "dense"

    @property
    def sparse_vector_name(self) -> str:
        return "sparse"


@lru_cache
def get_settings() -> Settings:
    return Settings()
