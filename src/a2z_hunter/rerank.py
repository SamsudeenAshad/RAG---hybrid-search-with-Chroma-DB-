"""Cross-encoder reranking over a fused candidate pool (keyless, local fastembed)."""
from __future__ import annotations

from langchain_core.documents import Document

from .clients import reranker
from .config import get_settings


def rerank(query: str, docs: list[Document], *, top_n: int | None = None) -> list[Document]:
    """Re-score docs against the query with a cross-encoder; keep top_n.

    Operates on the *combined* internal + web candidate pool so the surviving
    set is globally best rather than best-per-source.
    """
    if not docs:
        return []
    n = top_n or get_settings().rerank_top_n
    scores = list(reranker().rerank(query, [d.page_content for d in docs]))
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    out: list[Document] = []
    for doc, score in ranked[:n]:
        doc.metadata["rerank_score"] = float(score)
        out.append(doc)
    return out
