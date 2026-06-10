"""Hybrid retrieval for Chroma: dense (Chroma) + sparse (client-side BM25),
fused with Reciprocal Rank Fusion in Python.

Chroma has no server-side sparse vectors, so the BM25 half is built client-side
from the collection's documents. The BM25 index (and the id/text/metadata it
needs) is cached per collection in-process and invalidated by invalidate_bm25()
after an ingest.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from .clients import (
    chroma_collection,
    collection_name,
    embeddings,
    sparse_embedder,
)
from .config import get_settings


@dataclass
class _Bm25Index:
    ids: list[str]
    texts: list[str]
    metadatas: list[dict]
    # One sparse embedding per document, kept as {indices, values} lists.
    doc_terms: list[dict]


# Per-collection BM25 cache. Cleared by invalidate_bm25() after ingestion.
_bm25_cache: dict[str, _Bm25Index] = {}


def invalidate_bm25(name: str | None = None) -> None:
    """Drop the cached BM25 index so the next query rebuilds it from Chroma."""
    if name is None:
        _bm25_cache.clear()
    else:
        _bm25_cache.pop(name, None)


def _build_bm25(name: str) -> _Bm25Index:
    """Fetch all documents from the Chroma collection and build a BM25 index."""
    coll = chroma_collection(name)
    got = coll.get(include=["documents", "metadatas"])
    ids = got.get("ids") or []
    texts = got.get("documents") or []
    metadatas = got.get("metadatas") or [{} for _ in ids]

    doc_terms: list[dict] = []
    if texts:
        for emb in sparse_embedder().embed(texts):
            doc_terms.append(
                {"indices": emb.indices.tolist(), "values": emb.values.tolist()}
            )
    return _Bm25Index(ids=ids, texts=texts, metadatas=metadatas, doc_terms=doc_terms)


def _bm25_index(name: str) -> _Bm25Index:
    idx = _bm25_cache.get(name)
    if idx is None:
        idx = _build_bm25(name)
        _bm25_cache[name] = idx
    return idx


def _bm25_rank(query: str, idx: _Bm25Index, limit: int) -> list[str]:
    """Return chunk ids ranked by BM25 dot-product score (descending)."""
    if not idx.ids:
        return []
    q = next(iter(sparse_embedder().query_embed(query)))
    q_weights = dict(zip(q.indices.tolist(), q.values.tolist()))

    scored: list[tuple[float, str]] = []
    for chunk_id, terms in zip(idx.ids, idx.doc_terms):
        score = 0.0
        for i, v in zip(terms["indices"], terms["values"]):
            w = q_weights.get(i)
            if w is not None:
                score += w * v
        if score > 0:
            scored.append((score, chunk_id))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [chunk_id for _, chunk_id in scored[:limit]]


def _dense_rank(query: str, name: str, limit: int) -> tuple[list[str], dict]:
    """Query Chroma by dense vector. Returns (ranked ids, id->payload map)."""
    coll = chroma_collection(name)
    dense_vec = embeddings().embed_query(query)
    res = coll.query(
        query_embeddings=[dense_vec],
        n_results=limit,
        include=["documents", "metadatas"],
    )
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    payloads = {
        cid: {"text": doc, **(meta or {})}
        for cid, doc, meta in zip(ids, docs, metas)
    }
    return ids, payloads


def _rrf(rankings: list[list[str]], k: int) -> dict[str, float]:
    """Reciprocal Rank Fusion over multiple ranked id lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def hybrid_search(query: str, *, top_k: int | None = None) -> list[Document]:
    """Return up to top_k chunks, fusing dense (Chroma) + sparse (BM25) via RRF."""
    s = get_settings()
    limit = top_k or s.retrieval_top_k
    name = collection_name()

    dense_ids, payloads = _dense_rank(query, name, limit)
    bm25 = _bm25_index(name)
    sparse_ids = _bm25_rank(query, bm25, limit)

    # Backfill payloads for ids surfaced only by BM25 (not in the dense result).
    bm25_payload = {
        cid: {"text": txt, **(meta or {})}
        for cid, txt, meta in zip(bm25.ids, bm25.texts, bm25.metadatas)
    }
    for cid in sparse_ids:
        payloads.setdefault(cid, bm25_payload.get(cid, {}))

    fused = _rrf([dense_ids, sparse_ids], s.rrf_k)
    ranked = sorted(fused.items(), key=lambda t: t[1], reverse=True)[:limit]

    docs: list[Document] = []
    for chunk_id, score in ranked:
        payload = payloads.get(chunk_id, {})
        docs.append(
            Document(
                page_content=payload.get("text", ""),
                metadata={
                    "id": str(chunk_id),
                    "score": score,
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
