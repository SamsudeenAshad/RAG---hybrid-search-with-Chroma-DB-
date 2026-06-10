"""Shared client factories: Gemini LLM/embeddings, Chroma, sparse/rerank models.

All factories are cached so models (which can be expensive to load) are
instantiated once per process.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from .config import get_settings

# Per-request LLM selection set by the API/graph runner. None => fall back to
# the configured defaults. Lets the UI dropdown pick provider+model per query
# without changing any agent node's call site.
#   {"provider": "gemini"|"ollama"|"nvidia", "model": str|None}
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
        if provider == "gemini":
            model = default_model
        elif provider == "nvidia":
            model = s.nvidia_model
        else:
            model = s.ollama_model
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


@lru_cache
def _nvidia(model: str, temperature: float) -> BaseChatModel:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    s = get_settings()
    return ChatNVIDIA(
        model=model,
        api_key=s.nvidia_api_key,
        base_url=s.nvidia_base_url,
        temperature=temperature,
    )


def _build(default_model: str, temperature: float) -> BaseChatModel:
    provider, model = _resolve(default_model)
    if provider == "ollama":
        return _ollama(model, temperature)
    if provider == "nvidia":
        return _nvidia(model, temperature)
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


@lru_cache
def _nvidia_embeddings(model: str):
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    s = get_settings()
    return NVIDIAEmbeddings(
        model=model, api_key=s.nvidia_api_key, base_url=s.nvidia_base_url
    )


def embeddings():
    """Embedding model for the active embed provider (gemini | ollama | nvidia)."""
    s = get_settings()
    provider = active_embed_provider()
    if provider == "ollama":
        return _ollama_embeddings(s.ollama_embed_model)
    if provider == "nvidia":
        return _nvidia_embeddings(s.nvidia_embed_model)
    return _gemini_embeddings(s.gemini_embed_model)


def collection_name(provider: str | None = None) -> str:
    """Per-provider collection name, e.g. 'documents_gemini'."""
    s = get_settings()
    return f"{s.collection_base}_{provider or active_embed_provider()}"


@lru_cache
def chroma_client():
    """Cached Chroma Cloud client. Auth via tenant + database + api_key."""
    import chromadb

    s = get_settings()
    return chromadb.CloudClient(
        tenant=s.chroma_tenant or None,
        database=s.chroma_database,
        api_key=s.chroma_api_key or None,
    )


@lru_cache
def sparse_embedder():
    """Lazy import — fastembed pulls heavy deps; only load when needed.

    Used for the client-side BM25 half of hybrid search (Chroma has no
    server-side sparse vectors)."""
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=get_settings().sparse_model)


@lru_cache
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=get_settings().rerank_model)


def embed_dimension() -> int:
    """Dimension of the active embedding provider.

    Gemini is fixed (config); Ollama and NVIDIA are auto-detected by embedding
    a probe string (model dimensions vary, e.g. mxbai-embed-large=1024,
    nv-embedqa-e5-v5=1024).
    """
    s = get_settings()
    if active_embed_provider() in ("ollama", "nvidia"):
        return len(embeddings().embed_query("dimension probe"))
    return s.embed_dim


def chroma_collection(name: str | None = None):
    """Return the per-provider Chroma collection, creating it if missing.

    Cosine space matches the embedding providers. Chroma fixes the vector
    dimension on the first add, so no explicit size is set here.
    """
    name = name or collection_name()
    return chroma_client().get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def ensure_collection() -> str:
    """Ensure the per-provider collection exists. Returns its name.

    Verifies an existing, non-empty collection's vector dimension matches the
    active provider's dimension (a mismatch means it was built with a different
    embedding model — delete the collection to rebuild).
    """
    name = collection_name()
    coll = chroma_collection(name)

    if coll.count() > 0:
        peek = coll.peek(limit=1)
        existing = peek.get("embeddings")
        if existing is not None and len(existing) > 0:
            existing_dim = len(existing[0])
            dim = embed_dimension()
            if existing_dim != dim:
                raise ValueError(
                    f"Collection '{name}' vector size {existing_dim} != provider "
                    f"dimension {dim}. Delete the collection to re-create it."
                )
    return name
