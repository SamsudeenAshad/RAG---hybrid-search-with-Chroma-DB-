"""Hybrid retrieval: dense (Gemini) + sparse (BM25) fused with RRF in Qdrant."""
from __future__ import annotations

from langchain_core.documents import Document
from qdrant_client.models import FusionQuery, Fusion, Prefetch, SparseVector

from .clients import embeddings, qdrant_client, sparse_embedder
from .config import get_settings


def _sparse_query(text: str) -> SparseVector:
    emb = next(iter(sparse_embedder().query_embed(text)))
    return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())


def hybrid_search(query: str, *, top_k: int | None = None) -> list[Document]:
    """Return up to top_k chunks, fusing dense + sparse via Reciprocal Rank Fusion."""
    s = get_settings()
    limit = top_k or s.retrieval_top_k
    client = qdrant_client()

    dense_vec = embeddings().embed_query(query)
    sparse_vec = _sparse_query(query)

    result = client.query_points(
        collection_name=s.qdrant_collection,
        prefetch=[
            Prefetch(query=dense_vec, using=s.dense_vector_name, limit=limit),
            Prefetch(query=sparse_vec, using=s.sparse_vector_name, limit=limit),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    docs: list[Document] = []
    for point in result.points:
        payload = point.payload or {}
        docs.append(
            Document(
                page_content=payload.get("text", ""),
                metadata={
                    "id": str(point.id),
                    "score": point.score,
                    "document_id": payload.get("document_id"),
                    "ordinal": payload.get("ordinal"),
                    "title": payload.get("title"),
                    "source_uri": payload.get("source_uri"),
                    "source": "internal",
                },
            )
        )
    return docs


def top_score(docs: list[Document]) -> float:
    return max((d.metadata.get("score", 0.0) for d in docs), default=0.0)
