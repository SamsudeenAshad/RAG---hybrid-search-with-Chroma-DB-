# chroma_rag — Agentic Vector Search

A multi-agent retrieval system built with **LangGraph** (orchestration), **LangChain**
(LLM/embedding abstractions), **Chroma** (vector store), **PostgreSQL** (graph
checkpointer + document metadata), and pluggable model providers (**Gemini**, **Ollama**,
or **NVIDIA NIM**).

A user question flows through a pipeline of specialized agents that plan, rewrite,
retrieve (dense + sparse), rerank, fuse evidence, reason, **verify**, and respond — with
the conversation state durably checkpointed in Postgres so runs are resumable.

```
User Query → Planner → Query Rewriter → Hybrid Retrieval (Chroma dense + BM25, RRF)
   ├─ Internal Knowledge
   └─ Web Search (conditional)
→ Reranker → Evidence Fusion → Reasoning → Verification ──(loop back if unsupported)
→ Response (grounded + citations)
```

Chroma is dense-only, so the **sparse (BM25) half of hybrid search runs client-side**
and is fused with Chroma's dense results via **Reciprocal Rank Fusion (RRF) in Python**.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design and
[architecture.excalidraw](architecture.excalidraw) for the diagram.

## Requirements

- Python 3.10–3.13 (tested on 3.12)
- **Chroma** — local (Docker, default) or Chroma Cloud
- **PostgreSQL** — local (Docker) or remote
- An **embedding + LLM provider** — one of:
  - **Google Gemini** (`gemini-2.5-flash` works on the free tier; `gemini-2.5-pro` needs a paid key)
  - **Ollama** (self-hosted LLM + embeddings)
  - **NVIDIA NIM** (hosted chat; embeddings via `nv-embedqa-e5-v5`)
- Optional: a **Tavily API key** for higher-quality web search (else keyless DuckDuckGo)

## Setup

```bash
# 1. Config
cp .env.example .env
#    edit .env: set the provider keys + CHROMA_* / DATABASE_URL, etc.

# 2. Local infra (Chroma on :8800, Postgres on :5442)
docker compose up -d

# 3. Install
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Key environment variables (`.env`)

| Var | Purpose | Default |
|-----|---------|---------|
| `LLM_PROVIDER` / `EMBED_PROVIDER` | `gemini` \| `ollama` \| `nvidia` | `gemini` |
| `GOOGLE_API_KEY` | Gemini auth (required when provider is `gemini`) | — |
| `GEMINI_EMBED_MODEL` / `EMBED_DIM` | Gemini embeddings | `models/gemini-embedding-001` / `3072` |
| `CHROMA_HOST` / `CHROMA_PORT` | local Chroma (HttpClient) | `localhost` / `8800` |
| `CHROMA_DATABASE` | Chroma database name | `chroma_rag` |
| `CHROMA_TENANT` / `CHROMA_API_KEY` | Chroma **Cloud** (used only when `CHROMA_HOST` is empty) | — |
| `COLLECTION_BASE` | base collection name (per-provider suffix applied) | `documents` |
| `DATABASE_URL` | Postgres (checkpoints + metadata) | local Docker on `:5442` |
| `WEB_SEARCH_PROVIDER` | `duckduckgo` or `tavily` | `duckduckgo` |
| `RETRIEVAL_TOP_K` / `RERANK_TOP_N` / `SCORE_THRESHOLD` / `MAX_ATTEMPTS` / `RRF_K` | tuning | `20 / 6 / 0.45 / 2 / 60` |

> **Per-provider collections.** Each embedding provider gets its own Chroma
> collection (`<COLLECTION_BASE>_<provider>`, e.g. `documents_gemini`) sized to its
> vector dimension. Query with the **same provider you ingested with**.

> **Local vs. Cloud.** With `CHROMA_HOST` set, the client uses `HttpClient` against a
> local Chroma (no record quota). Leave `CHROMA_HOST` empty to use Chroma Cloud via
> `CloudClient` (`CHROMA_TENANT` + `CHROMA_API_KEY` required).

## Usage

### Ingest documents

```bash
# files / directories (.txt, .md)
python -m chroma_rag.ingest ./data

# raw text
python -m chroma_rag.ingest --title "My Note" --text "some content to index"

# target a specific embedding provider's collection
python -m chroma_rag.ingest ./data --embed-provider gemini
```

Ingestion is idempotent (sha256 dedupe, scoped per provider), batches dense embeddings
at Gemini's 100-string limit, and adds dense vectors + chunk text/metadata to Chroma.
BM25 is **not** stored — it is rebuilt client-side from the collection at query time.

### Query (programmatic)

```python
from chroma_rag.graph import run_query

final = run_query("What is hybrid search and how does RRF combine dense + BM25?", thread_id="abc")
print(final["answer"])          # grounded answer + citations
print(final["verification"])    # {supported, unsupported_claims, rationale}
```

### Query (HTTP)

```bash
python -m chroma_rag.api          # FastAPI on :8000 (override with API_PORT)
curl -X POST localhost:8000/query -H 'content-type: application/json' \
     -d '{"question": "What is hybrid search?"}'
```

Endpoints: `POST /ingest`, `POST /query` (optional `thread_id` for multi-turn),
`GET /models`, `GET /health`. A browser UI is served at `/`.

## How it works

| Stage | What it does |
|-------|--------------|
| **Planner** | decomposes the question; flags whether web search is needed |
| **Query Rewriter** | multi-query expansion (folds in unsupported claims on retry) |
| **Hybrid Retrieval** | dense (Chroma) + sparse (client-side BM25) fused with RRF in Python |
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
src/chroma_rag/
├── config.py        clients.py     db.py
├── retriever.py     rerank.py      ingest.py
├── state.py         graph.py       api.py
└── agents/          planner · query_rewriter · hybrid_retrieval · web_search
                     rerank_node · evidence_fusion · reasoning · verification · response
```
