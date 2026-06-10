"""Planner agent: decompose the question and decide whether web search is needed."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..clients import chat_llm
from ..state import AgentState

_PROMPT = (
    "You are a retrieval planner. Given a user question, break it into 1-3 focused "
    "search subqueries and decide whether an internal knowledge base alone is likely "
    "to answer it, or whether live web search is also needed (e.g. for recent events, "
    "real-time data, or topics unlikely to be in a private corpus).\n\n"
    "Question: {question}"
)


class PlanModel(BaseModel):
    subqueries: list[str] = Field(default_factory=list)
    needs_web: bool = False
    rationale: str = ""


def planner_node(state: AgentState) -> dict:
    llm = chat_llm().with_structured_output(PlanModel)
    plan: PlanModel = llm.invoke(_PROMPT.format(question=state["question"]))
    return {
        "plan": plan.model_dump(),
        "retrieval_attempts": state.get("retrieval_attempts", 0),
    }
