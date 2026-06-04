"""Reranker node: cross-encoder re-scores the combined internal + web pool."""
from __future__ import annotations

from ..rerank import rerank
from ..state import AgentState


def rerank_node(state: AgentState) -> dict:
    pool = list(state.get("retrieved", [])) + list(state.get("web_results", []))
    pool = [d for d in pool if not d.metadata.get("error")]
    reranked = rerank(state["question"], pool)
    return {"reranked": reranked}
