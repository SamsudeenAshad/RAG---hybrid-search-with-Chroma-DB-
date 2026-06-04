"""Reasoning agent: draft an answer grounded in the fused evidence, with [n] refs."""
from __future__ import annotations

from ..clients import reasoning_llm
from ..state import AgentState

_PROMPT = (
    "You are a careful research assistant. Answer the question using ONLY the numbered "
    "evidence below. Cite supporting passages inline as [n]. If the evidence is "
    "insufficient, say so explicitly rather than guessing.\n\n"
    "Question: {question}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Draft answer:"
)


def reasoning_node(state: AgentState) -> dict:
    text = reasoning_llm().invoke(
        _PROMPT.format(question=state["question"], evidence=state.get("evidence", ""))
    ).content
    return {"draft_answer": text}
