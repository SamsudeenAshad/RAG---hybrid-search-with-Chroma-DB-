"""Hybrid Retrieval node: run each rewritten query through dense+sparse RRF search,
then dedupe by chunk id across queries."""
from __future__ import annotations

from langchain_core.documents import Document

from ..retriever import hybrid_search
from ..state import AgentState


def hybrid_retrieval_node(state: AgentState) -> dict:
    queries = state.get("rewritten_queries") or [state["question"]]
    seen: dict[str, Document] = {}
    for q in queries:
        for doc in hybrid_search(q):
            key = doc.metadata.get("id", doc.page_content[:80])
            # Keep the highest-scoring instance of each chunk.
            if key not in seen or doc.metadata.get("score", 0) > seen[key].metadata.get(
                "score", 0
            ):
                seen[key] = doc
    return {"retrieved": list(seen.values())}
