# a2z_hunter — Agentic Vector Search

A multi-agent retrieval system built with **LangGraph** (orchestration), **LangChain**
(LLM/embedding abstractions), **Qdrant** (hybrid vector store), **PostgreSQL** (graph
checkpointer + document metadata), and **Google Gemini** (LLM + embeddings).

A user question flows through a pipeline of specialized agents that plan, rewrite,
retrieve (dense + sparse), rerank, fuse evidence, reason, **verify**, and respond — with
the conversation state durably checkpointed in Postgres so runs are resumable.

```
User Query → Planner → Query Rewriter → Hybrid Retrieval (Qdrant + BM25, RRF)
   ├─ Internal Knowledge
   └─ Web Search (conditional)
→ Reranker → Evidence Fusion → Reasoning → Verification ──(loop back if unsupported)
→ Response (grounded + citations)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design and
[architecture.excalidraw](architecture.excalidraw) for the diagram.

## Requirements

- Python 3.10–3.13 (tested on 3.12)
- A **Google Gemini API key** (`gemini-2.5-flash` works on the free tier;
  `gemini-2.5-pro` needs a paid key)
- **Qdrant** — local (Docker) or Qdrant Cloud
- **PostgreSQL** — local (Docker) or remote
- Optional: a **Tavily API key** for higher-quality web search (else keyless DuckDuckGo)

## Setup

```bash
# 1. Config
cp .env.example .env
#    edit .env: set GOOGLE_API_KEY (+ QDRANT_URL/QDRANT_API_KEY, DATABASE_URL, etc.)

# 2. (Optional) local infra
docker compose up -d            # Qdrant on :6533, Postgres on :5442

# 3. Install
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Key environment variables (`.env`)

| Var | Purpose | Default |
|-----|---------|---------|
| `GOOGLE_API_KEY` | Gemini auth (**required**) | — |
| `GEMINI_CHAT_MODEL` | fast nodes | `gemini-2.5-flash` |
| `GEMINI_REASONING_MODEL` | reasoning + verification | `gemini-2.5-flash` |
| `GEMINI_EMBED_MODEL` / `EMBED_DIM` | embeddings | `models/gemini-embedding-001` / `3072` |
| `QDRANT_URL` / `QDRANT_API_KEY` | vector store (local or cloud) | `http://localhost:6533` |
| `DATABASE_URL` | Postgres (checkpoints + metadata) | local Docker on `:5442` |
| `WEB_SEARCH_PROVIDER` | `duckduckgo` or `tavily` | `duckduckgo` |
| `RETRIEVAL_TOP_K` / `RERANK_TOP_N` / `SCORE_THRESHOLD` / `MAX_ATTEMPTS` | tuning | `20 / 6 / 0.45 / 2` |

> **Embedding dimension must match the Qdrant collection.** If you change
> `EMBED_DIM`, drop and recreate the `documents` collection.

## Usage

### Ingest documents

```bash
# files / directories (.txt, .md)
python -m a2z_hunter.ingest ./data

# raw text
python -m a2z_hunter.ingest --title "My Note" --text "some content to index"
```

Ingestion is idempotent (sha256 dedupe), batches dense embeddings at Gemini's 100-string
limit, and writes both a dense (Gemini) and sparse (BM25) vector per chunk.

### Query (programmatic)

```python
from a2z_hunter.graph import run_query

final = run_query("What is Qdrant and how does it do hybrid search?", thread_id="abc")
print(final["answer"])          # grounded answer + citations
print(final["verification"])    # {supported, unsupported_claims, rationale}
```

### Query (HTTP)

```bash
python -m a2z_hunter.api          # FastAPI on :8000
curl -X POST localhost:8000/query -H 'content-type: application/json' \
     -d '{"question": "What is Qdrant?"}'
```

Endpoints: `POST /ingest`, `POST /query` (optional `thread_id` for multi-turn), `GET /health`.

## How it works

| Stage | What it does |
|-------|--------------|
| **Planner** | decomposes the question; flags whether web search is needed |
| **Query Rewriter** | multi-query expansion (folds in unsupported claims on retry) |
| **Hybrid Retrieval** | dense (Gemini) + sparse (BM25) fused server-side with RRF in Qdrant |
| **Web Search** | conditional — runs when planner flags it or internal scores are weak |
| **Reranker** | cross-encoder re-scores the combined internal+web pool, keeps top-N |
| **Evidence Fusion** | dedupes/merges passages into a numbered context block + citation map |
| **Reasoning** | drafts an answer grounded in the evidence, with inline `[n]` refs |
| **Verification** | fact-checks the draft; loops back to rewrite if claims are unsupported (≤ `MAX_ATTEMPTS`) |
| **Response** | polishes the verified draft and appends a sources list |

State persists in Postgres via LangGraph's `PostgresSaver`, keyed by `thread_id` —
runs are resumable and inspectable.

## Tests

```bash
pytest -q          # routing, evidence fusion, ingest helpers (no live services needed)
```

## Layout

```
src/a2z_hunter/
├── config.py        clients.py     db.py
├── retriever.py     rerank.py      ingest.py
├── state.py         graph.py       api.py
└── agents/          planner · query_rewriter · hybrid_retrieval · web_search
                     rerank_node · evidence_fusion · reasoning · verification · response
```
