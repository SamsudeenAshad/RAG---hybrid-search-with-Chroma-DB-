"""Wire the agent nodes into a LangGraph StateGraph with Postgres checkpointing.

Flow:
    planner -> query_rewriter -> hybrid_retrieval
        -> (web_search | rerank)          # conditional on plan/score
    web_search -> rerank
    rerank -> evidence_fusion -> reasoning -> verification
        -> (query_rewriter | response)    # loop-back gate
    response -> END
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from langgraph.graph import END, START, StateGraph

from .agents.evidence_fusion import evidence_fusion_node
from .agents.hybrid_retrieval import hybrid_retrieval_node
from .agents.planner import planner_node
from .agents.query_rewriter import query_rewriter_node
from .agents.reasoning import reasoning_node
from .agents.rerank_node import rerank_node
from .agents.response import response_node
from .agents.verification import verification_node
from .agents.web_search import web_search_node
from .config import get_settings
from .retriever import top_score
from .state import AgentState


def _route_after_retrieval(state: AgentState) -> str:
    """Branch to web search when the planner flagged it or internal hits are weak."""
    s = get_settings()
    plan = state.get("plan", {})
    weak = top_score(state.get("retrieved", [])) < s.score_threshold
    return "web_search" if (plan.get("needs_web") or weak) else "rerank"


def _route_after_verification(state: AgentState) -> str:
    """Loop back to rewrite when unsupported and under the attempt cap; else finish."""
    s = get_settings()
    verification = state.get("verification", {})
    attempts = state.get("retrieval_attempts", 0)
    if not verification.get("supported", False) and attempts < s.max_attempts:
        return "query_rewriter"
    return "response"


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("query_rewriter", query_rewriter_node)
    g.add_node("hybrid_retrieval", hybrid_retrieval_node)
    g.add_node("web_search", web_search_node)
    g.add_node("rerank", rerank_node)
    g.add_node("evidence_fusion", evidence_fusion_node)
    g.add_node("reasoning", reasoning_node)
    g.add_node("verification", verification_node)
    g.add_node("response", response_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "query_rewriter")
    g.add_edge("query_rewriter", "hybrid_retrieval")
    g.add_conditional_edges(
        "hybrid_retrieval",
        _route_after_retrieval,
        {"web_search": "web_search", "rerank": "rerank"},
    )
    g.add_edge("web_search", "rerank")
    g.add_edge("rerank", "evidence_fusion")
    g.add_edge("evidence_fusion", "reasoning")
    g.add_edge("reasoning", "verification")
    g.add_conditional_edges(
        "verification",
        _route_after_verification,
        {"query_rewriter": "query_rewriter", "response": "response"},
    )
    g.add_edge("response", END)
    return g


@contextmanager
def compiled_graph() -> Iterator:
    """Compile the graph with a Postgres checkpointer.

    PostgresSaver requires autocommit=True + dict_row; .setup() creates the
    checkpoint tables on first use.
    """
    from psycopg import Connection
    from psycopg.rows import dict_row
    from langgraph.checkpoint.postgres import PostgresSaver

    s = get_settings()
    conn = Connection.connect(
        s.database_url, autocommit=True, row_factory=dict_row, prepare_threshold=0
    )
    try:
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        yield build_graph().compile(checkpointer=checkpointer)
    finally:
        conn.close()


def run_query(question: str, *, thread_id: str) -> AgentState:
    """Convenience runner for one question on a given conversation thread."""
    with compiled_graph() as graph:
        config = {"configurable": {"thread_id": thread_id}}
        return graph.invoke({"question": question, "messages": []}, config=config)
