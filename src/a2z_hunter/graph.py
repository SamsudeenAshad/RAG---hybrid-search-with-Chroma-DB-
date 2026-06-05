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
from .clients import reset_llm_override, set_llm_override
from .config import get_settings
from .retriever import top_score
from .state import AgentState


@contextmanager
def _llm_selection(provider: str | None, model: str | None) -> Iterator[None]:
    """Apply a per-run provider/model selection to all agent nodes."""
    token = set_llm_override(provider, model)
    try:
        yield
    finally:
        reset_llm_override(token)


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


def run_query(
    question: str,
    *,
    thread_id: str,
    provider: str | None = None,
    model: str | None = None,
) -> AgentState:
    """Convenience runner for one question on a given conversation thread."""
    with _llm_selection(provider, model), compiled_graph() as graph:
        config = {"configurable": {"thread_id": thread_id}}
        return graph.invoke({"question": question, "messages": []}, config=config)


# Human-readable labels + emoji for each node, used by the live "thinking" stream.
_NODE_LABELS: dict[str, str] = {
    "planner": "🧭 Planning — decomposing the question",
    "query_rewriter": "✍️ Rewriting — expanding into multiple queries",
    "hybrid_retrieval": "🔎 Retrieving — dense + sparse search (RRF)",
    "web_search": "🌐 Web search — fetching external sources",
    "rerank": "📊 Reranking — cross-encoder scoring",
    "evidence_fusion": "🧩 Fusing evidence — building the context block",
    "reasoning": "💭 Reasoning — drafting a grounded answer",
    "verification": "✅ Verifying — fact-checking the draft",
    "response": "📝 Responding — polishing + citations",
}


def _summarize_node(node: str, update: dict) -> str:
    """Produce a short detail line describing what a node just produced."""
    try:
        if node == "planner":
            plan = update.get("plan", {}) or {}
            subs = plan.get("subqueries", []) or []
            web = "web search needed" if plan.get("needs_web") else "no web search"
            return f"{len(subs)} sub-quer{'y' if len(subs) == 1 else 'ies'} · {web}"
        if node == "query_rewriter":
            qs = update.get("rewritten_queries", []) or []
            return f"{len(qs)} query variant(s)"
        if node == "hybrid_retrieval":
            hits = update.get("retrieved", []) or []
            return f"{len(hits)} candidate passage(s)"
        if node == "web_search":
            hits = update.get("web_results", []) or []
            return f"{len(hits)} web result(s)"
        if node == "rerank":
            kept = update.get("reranked", []) or []
            return f"kept top {len(kept)} passage(s)"
        if node == "evidence_fusion":
            cites = update.get("citations", []) or []
            return f"{len(cites)} citation(s) assembled"
        if node == "reasoning":
            draft = update.get("draft_answer", "") or ""
            return f"draft answer ({len(draft)} chars)"
        if node == "verification":
            v = update.get("verification", {}) or {}
            if v.get("supported"):
                return "supported ✓"
            n = len(v.get("unsupported_claims", []) or [])
            return f"{n} unsupported claim(s) — may retry"
        if node == "response":
            return "final answer ready"
    except Exception:  # never let summarization break the stream
        pass
    return ""


def run_query_stream(
    question: str,
    *,
    thread_id: str,
    provider: str | None = None,
    model: str | None = None,
) -> Iterator[dict]:
    """Run the pipeline, yielding one event dict per node as it completes.

    Event shapes:
      {"type": "step",  "node", "label", "detail", "attempts"}
      {"type": "final", "answer", "citations", "verification", "attempts", "thread_id"}
      {"type": "error", "message"}
    """
    with _llm_selection(provider, model), compiled_graph() as graph:
        config = {"configurable": {"thread_id": thread_id}}
        final_state: AgentState = {}
        try:
            for chunk in graph.stream(
                {"question": question, "messages": []},
                config=config,
                stream_mode="updates",
            ):
                # updates mode: {node_name: {state_delta}}
                for node, update in chunk.items():
                    update = update or {}
                    final_state.update(update)
                    yield {
                        "type": "step",
                        "node": node,
                        "label": _NODE_LABELS.get(node, node),
                        "detail": _summarize_node(node, update),
                        "attempts": final_state.get("retrieval_attempts", 0),
                    }
            yield {
                "type": "final",
                "thread_id": thread_id,
                "answer": final_state.get("answer", ""),
                "citations": final_state.get("citations", []),
                "verification": final_state.get("verification", {}),
                "attempts": final_state.get("retrieval_attempts", 0),
            }
        except Exception as e:  # surface pipeline errors to the client
            yield {"type": "error", "message": str(e)}
