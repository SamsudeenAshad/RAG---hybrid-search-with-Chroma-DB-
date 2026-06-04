"""Web Search node (conditional): fill gaps via DuckDuckGo (keyless) or Tavily."""
from __future__ import annotations

from langchain_core.documents import Document

from ..config import get_settings
from ..state import AgentState


def _duckduckgo(query: str, max_results: int) -> list[Document]:
    from duckduckgo_search import DDGS

    docs: list[Document] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            docs.append(
                Document(
                    page_content=f"{r.get('title', '')}\n{r.get('body', '')}",
                    metadata={
                        "source": "web",
                        "source_uri": r.get("href", ""),
                        "title": r.get("title", ""),
                    },
                )
            )
    return docs


def _tavily(query: str, max_results: int) -> list[Document]:
    from langchain_tavily import TavilySearch

    tool = TavilySearch(max_results=max_results, api_key=get_settings().tavily_api_key)
    results = tool.invoke({"query": query}).get("results", [])
    return [
        Document(
            page_content=f"{r.get('title', '')}\n{r.get('content', '')}",
            metadata={
                "source": "web",
                "source_uri": r.get("url", ""),
                "title": r.get("title", ""),
            },
        )
        for r in results
    ]


def web_search_node(state: AgentState) -> dict:
    s = get_settings()
    queries = state.get("rewritten_queries") or [state["question"]]
    provider = _tavily if s.web_search_provider == "tavily" else _duckduckgo

    docs: list[Document] = []
    for q in queries[:2]:  # cap external calls
        try:
            docs.extend(provider(q, max_results=4))
        except Exception as exc:  # web search is best-effort; never fail the graph
            docs.append(
                Document(
                    page_content=f"[web search unavailable: {exc}]",
                    metadata={"source": "web", "error": True},
                )
            )
    return {"web_results": docs}
