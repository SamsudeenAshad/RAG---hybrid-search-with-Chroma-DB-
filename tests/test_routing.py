"""Unit tests for graph routing + evidence fusion — no live services required."""
from __future__ import annotations

import os

os.environ.setdefault("GOOGLE_API_KEY", "test")

from langchain_core.documents import Document  # noqa: E402

from a2z_hunter.agents.evidence_fusion import evidence_fusion_node  # noqa: E402
from a2z_hunter.config import get_settings  # noqa: E402
from a2z_hunter.graph import (  # noqa: E402
    _route_after_retrieval,
    _route_after_verification,
)


def _doc(score: float, source: str = "internal") -> Document:
    return Document(page_content="x", metadata={"score": score, "source": source, "id": str(score)})


def test_route_to_web_when_planner_flags_it():
    state = {"plan": {"needs_web": True}, "retrieved": [_doc(0.9)]}
    assert _route_after_retrieval(state) == "web_search"


def test_route_to_web_when_scores_weak():
    thresh = get_settings().score_threshold
    state = {"plan": {"needs_web": False}, "retrieved": [_doc(thresh - 0.1)]}
    assert _route_after_retrieval(state) == "web_search"


def test_route_to_rerank_when_strong_internal_hits():
    thresh = get_settings().score_threshold
    state = {"plan": {"needs_web": False}, "retrieved": [_doc(thresh + 0.2)]}
    assert _route_after_retrieval(state) == "rerank"


def test_verification_loops_back_when_unsupported_under_cap():
    state = {"verification": {"supported": False}, "retrieval_attempts": 1}
    # default MAX_ATTEMPTS = 2, so attempt 1 still has budget
    assert _route_after_verification(state) == "query_rewriter"


def test_verification_finishes_when_supported():
    state = {"verification": {"supported": True}, "retrieval_attempts": 1}
    assert _route_after_verification(state) == "response"


def test_verification_finishes_when_cap_reached():
    cap = get_settings().max_attempts
    state = {"verification": {"supported": False}, "retrieval_attempts": cap}
    assert _route_after_verification(state) == "response"


def test_evidence_fusion_numbers_and_cites():
    state = {
        "question": "q",
        "reranked": [
            Document(page_content="alpha", metadata={"title": "A", "source": "internal"}),
            Document(page_content="beta", metadata={"title": "B", "source": "web", "source_uri": "http://x"}),
        ],
    }
    out = evidence_fusion_node(state)
    assert "[1]" in out["evidence"] and "[2]" in out["evidence"]
    assert len(out["citations"]) == 2
    assert out["citations"][1]["source_uri"] == "http://x"
