"""Response agent: produce the final, user-facing answer with a citations list."""
from __future__ import annotations

from ..clients import chat_llm
from ..state import AgentState

_PROMPT = (
    "Polish the verified draft into a clear, well-structured final answer for the user. "
    "Preserve inline [n] citations. Do not introduce facts beyond the draft.\n\n"
    "Question: {question}\n\n"
    "Verified draft:\n{draft}"
)


def response_node(state: AgentState) -> dict:
    draft = state.get("draft_answer", "")
    answer = chat_llm().invoke(
        _PROMPT.format(question=state["question"], draft=draft)
    ).content

    citations = state.get("citations", [])
    if citations:
        refs = "\n".join(
            f"[{c['ref']}] {c['title']}"
            + (f" — {c['source_uri']}" if c.get("source_uri") else "")
            for c in citations
        )
        answer = f"{answer}\n\nSources:\n{refs}"
    return {"answer": answer}
