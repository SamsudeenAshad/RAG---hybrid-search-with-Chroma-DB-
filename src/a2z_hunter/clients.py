"""Shared client factories: Gemini LLM/embeddings, Qdrant, sparse/rerank models.

All factories are cached so models (which can be expensive to load) are
instantiated once per process.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseVectorParams,
    VectorParams,
)

from .config import get_settings


@lru_cache
def chat_llm() -> ChatGoogleGenerativeAI:
    s = get_settings()
    return ChatGoogleGenerativeAI(
        model=s.gemini_chat_model, google_api_key=s.google_api_key, temperature=0
    )


@lru_cache
def reasoning_llm() -> ChatGoogleGenerativeAI:
    s = get_settings()
    return ChatGoogleGenerativeAI(
        model=s.gemini_reasoning_model, google_api_key=s.google_api_key, temperature=0.2
    )


@lru_cache
def embeddings() -> GoogleGenerativeAIEmbeddings:
    s = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=s.gemini_embed_model, google_api_key=s.google_api_key
    )


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


def ensure_collection() -> None:
    """Create the hybrid (dense + sparse) Qdrant collection if missing.

    Asserts the configured embedding dimension matches the dense vector size.
    """
    s = get_settings()
    client = qdrant_client()
    if client.collection_exists(s.qdrant_collection):
        info = client.get_collection(s.qdrant_collection)
        dense = info.config.params.vectors[s.dense_vector_name]
        if dense.size != s.embed_dim:
            raise ValueError(
                f"Collection '{s.qdrant_collection}' dense size {dense.size} "
                f"!= configured EMBED_DIM {s.embed_dim}"
            )
        return

    client.create_collection(
        collection_name=s.qdrant_collection,
        vectors_config={
            s.dense_vector_name: VectorParams(size=s.embed_dim, distance=Distance.COSINE)
        },
        sparse_vectors_config={s.sparse_vector_name: SparseVectorParams()},
    )
