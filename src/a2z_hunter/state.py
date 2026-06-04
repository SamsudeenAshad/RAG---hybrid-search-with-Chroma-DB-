"""Shared LangGraph state for the agentic retrieval pipeline."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class Plan(TypedDict, total=False):
    subqueries: list[str]
    needs_web: bool
    rationale: str


class Verification(TypedDict, total=False):
    supported: bool
    unsupported_claims: list[str]
    rationale: str


class AgentState(TypedDict, total=False):
    question: str
    plan: Plan
    rewritten_queries: list[str]
    retrieved: list[Document]        # hybrid hits, pre-rerank
    web_results: list[Document]      # optional external hits
    reranked: list[Document]         # top-N after cross-encoder
    evidence: str                    # fused context block
    citations: list[dict]
    draft_answer: str
    verification: Verification
    retrieval_attempts: int
    answer: str
    messages: Annotated[list, add_messages]
