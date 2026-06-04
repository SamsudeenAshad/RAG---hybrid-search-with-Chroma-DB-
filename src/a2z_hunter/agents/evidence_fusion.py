"""Evidence Fusion: assemble reranked passages into a numbered context block
with a parallel citation map the response agent can reference."""
from __future__ import annotations

from ..state import AgentState


def evidence_fusion_node(state: AgentState) -> dict:
    docs = state.get("reranked", [])
    lines: list[str] = []
    citations: list[dict] = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title") or doc.metadata.get("source_uri") or "untitled"
        src = doc.metadata.get("source", "internal")
        lines.append(f"[{i}] ({src}: {title})\n{doc.page_content.strip()}")
        citations.append(
            {
                "ref": i,
                "title": title,
                "source": src,
                "source_uri": doc.metadata.get("source_uri", ""),
            }
        )
    evidence = "\n\n".join(lines) if lines else "(no evidence retrieved)"
    return {"evidence": evidence, "citations": citations}
