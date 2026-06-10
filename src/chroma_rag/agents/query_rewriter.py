"""Query Rewriter: expand into multiple retrieval-friendly queries.

On a verification loop-back, unsupported claims are folded in to broaden recall.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..clients import chat_llm
from ..state import AgentState

_PROMPT = (
    "Rewrite the user's question into 2-4 diverse, self-contained search queries that "
    "maximize retrieval recall. Use synonyms and alternate phrasings. "
    "Return only the queries.\n\n"
    "Question: {question}\n"
    "Planner subqueries: {subqueries}\n"
    "{retry_hint}"
)


class Rewrites(BaseModel):
    queries: list[str] = Field(default_factory=list)


def query_rewriter_node(state: AgentState) -> dict:
    plan = state.get("plan", {})
    unsupported = state.get("verification", {}).get("unsupported_claims", [])
    retry_hint = (
        f"A previous answer left these claims unsupported; craft queries to find "
        f"evidence for them: {unsupported}"
        if unsupported
        else ""
    )
    llm = chat_llm().with_structured_output(Rewrites)
    out: Rewrites = llm.invoke(
        _PROMPT.format(
            question=state["question"],
            subqueries=plan.get("subqueries", []),
            retry_hint=retry_hint,
        )
    )
    queries = out.queries or [state["question"]]
    return {"rewritten_queries": queries}
