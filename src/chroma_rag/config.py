"""Typed application configuration loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM provider selection: "gemini", "ollama", or "nvidia". Drives the agent nodes.
    llm_provider: str = "gemini"

    # Embedding provider: "gemini", "ollama", or "nvidia". Because providers produce
    # different vector dimensions, EACH gets its own Chroma collection
    # (collection_base + "_" + provider). Dimension is auto-detected.
    embed_provider: str = "gemini"

    # Gemini
    google_api_key: str = ""
    gemini_chat_model: str = "gemini-2.5-flash"
    # 'pro' is unavailable on Gemini free tier (quota 0). Default reasoning to
    # flash; override GEMINI_REASONING_MODEL=gemini-2.5-pro on a paid key.
    gemini_reasoning_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "models/gemini-embedding-001"
    embed_dim: int = 3072  # Gemini dimension; Ollama is auto-detected.

    # Ollama (self-hosted LLM + embeddings). Used when *_provider="ollama".
    # Set OLLAMA_BASE_URL in .env to your host.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_embed_model: str = "mxbai-embed-large"

    # NVIDIA NIM (hosted; chat only). Used when llm_provider="nvidia".
    # OpenAI-compatible endpoint at integrate.api.nvidia.com. Get a key at
    # https://build.nvidia.com (nvapi-...). No embeddings wired here.
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.3-70b-instruct"
    # NVIDIA embeddings (used when embed_provider="nvidia"). 1024-dim;
    # auto-detected like Ollama. Gets its own Chroma collection.
    nvidia_embed_model: str = "nvidia/nv-embedqa-e5-v5"

    # Chroma Cloud. tenant_id + database + api_key from `chroma login` /
    # `chroma db create`. The base collection name gets a per-provider suffix
    # applied (e.g. "documents_ollama"); see clients.collection_name().
    chroma_tenant: str = ""
    chroma_database: str = "a2z_hunter"
    chroma_api_key: str = ""
    collection_base: str = "documents"  # base name; per-provider suffix applied

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
    # Sparse model for the client-side BM25 half of hybrid search. Chroma has no
    # server-side sparse vectors, so BM25 scoring is fused with Chroma's dense
    # results via RRF in Python (see retriever.py). "Qdrant/bm25" is fastembed's
    # HuggingFace model id for BM25 (no Qdrant runtime dependency).
    sparse_model: str = "Qdrant/bm25"

    # RRF rank constant for fusing dense (Chroma) + sparse (BM25) rankings.
    rrf_k: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
