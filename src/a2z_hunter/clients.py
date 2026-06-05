"""Shared client factories: Gemini LLM/embeddings, Qdrant, sparse/rerank models.

All factories are cached so models (which can be expensive to load) are
instantiated once per process.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseVectorParams,
    VectorParams,
)

from .config import get_settings

# Per-request LLM selection set by the API/graph runner. None => fall back to
# the configured defaults. Lets the UI dropdown pick provider+model per query
# without changing any agent node's call site.
#   {"provider": "gemini"|"ollama", "model": str|None}
_llm_override: ContextVar[dict | None] = ContextVar("llm_override", default=None)


def set_llm_override(provider: str | None, model: str | None) -> object:
    """Set the active provider/model for the current context. Returns a token
    to reset with reset_llm_override()."""
    if not provider and not model:
        return _llm_override.set(None)
    return _llm_override.set({"provider": provider, "model": model})


def reset_llm_override(token: object) -> None:
    _llm_override.reset(token)


def _resolve(default_model: str) -> tuple[str, str]:
    """Resolve (provider, model) from override → settings → arg default."""
    s = get_settings()
    override = _llm_override.get()
    provider = (override or {}).get("provider") or s.llm_provider
    model = (override or {}).get("model")
    if not model:
        model = default_model if provider == "gemini" else s.ollama_model
    return provider, model


@lru_cache
def _gemini(model: str, temperature: float) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model, google_api_key=get_settings().google_api_key, temperature=temperature
    )


@lru_cache
def _ollama(model: str, temperature: float) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model, base_url=get_settings().ollama_base_url, temperature=temperature
    )


def _build(default_model: str, temperature: float) -> BaseChatModel:
    provider, model = _resolve(default_model)
    if provider == "ollama":
        return _ollama(model, temperature)
    return _gemini(model, temperature)


def chat_llm() -> BaseChatModel:
    """Fast nodes (planner, rewriter, response)."""
    return _build(get_settings().gemini_chat_model, 0.0)


def reasoning_llm() -> BaseChatModel:
    """Reasoning + verification nodes."""
    return _build(get_settings().gemini_reasoning_model, 0.2)


# Per-request embedding-provider selection (mirrors the chat-LLM override).
_embed_override: ContextVar[str | None] = ContextVar("embed_override", default=None)


def set_embed_override(provider: str | None) -> object:
    return _embed_override.set(provider or None)


def reset_embed_override(token: object) -> None:
    _embed_override.reset(token)


def active_embed_provider() -> str:
    return _embed_override.get() or get_settings().embed_provider


@lru_cache
def _gemini_embeddings(model: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=model, google_api_key=get_settings().google_api_key
    )


@lru_cache
def _ollama_embeddings(model: str):
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=model, base_url=get_settings().ollama_base_url)


def embeddings():
    """Embedding model for the active embed provider (gemini | ollama)."""
    s = get_settings()
    if active_embed_provider() == "ollama":
        return _ollama_embeddings(s.ollama_embed_model)
    return _gemini_embeddings(s.gemini_embed_model)


def collection_name(provider: str | None = None) -> str:
    """Per-provider collection name, e.g. 'documents_gemini'."""
    s = get_settings()
    return f"{s.qdrant_collection}_{provider or active_embed_provider()}"


@lru_cache
def qdrant_client() -> QdrantClient:
    s = get_settings()
    return QdrantClient(
        url=s.qdrant_url,
        api_key=s.qdrant_api_key or None,
        # Qdrant Cloud requires gRPC disabled / HTTPS via the REST URL.
        prefer_grpc=False,
    )


@lru_cache
def sparse_embedder():
    """Lazy import — fastembed pulls heavy deps; only load when needed."""
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=get_settings().sparse_model)


@lru_cache
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=get_settings().rerank_model)


def embed_dimension() -> int:
    """Dimension of the active embedding provider.

    Gemini is fixed (config); Ollama is auto-detected by embedding a probe
    string (model dimensions vary, e.g. mxbai-embed-large=1024).
    """
    s = get_settings()
    if active_embed_provider() == "ollama":
        return len(embeddings().embed_query("dimension probe"))
    return s.embed_dim


def ensure_collection() -> str:
    """Create the per-provider hybrid (dense + sparse) collection if missing.

    Returns the resolved collection name. Verifies an existing collection's
    dense size matches the active provider's dimension.
    """
    s = get_settings()
    client = qdrant_client()
    name = collection_name()
    dim = embed_dimension()

    if client.collection_exists(name):
        info = client.get_collection(name)
        dense = info.config.params.vectors[s.dense_vector_name]
        if dense.size != dim:
            raise ValueError(
                f"Collection '{name}' dense size {dense.size} != provider "
                f"dimension {dim}. Drop the collection to re-create it."
            )
        return name

    client.create_collection(
        collection_name=name,
        vectors_config={
            s.dense_vector_name: VectorParams(size=dim, distance=Distance.COSINE)
        },
        sparse_vectors_config={s.sparse_vector_name: SparseVectorParams()},
    )
    return name
