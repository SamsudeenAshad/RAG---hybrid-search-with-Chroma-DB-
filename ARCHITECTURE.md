# Agentic Vector Search — Architecture & Implementation Plan

> Multi-agent retrieval system using **LangGraph** (orchestration), **LangChain** (LLM/embedding/retriever abstractions), **Chroma** (vector store), **PostgreSQL** (graph checkpointer + document metadata), and pluggable model providers (**Gemini**, **Ollama**, or **NVIDIA NIM**).

---

## 1. Goal

Given a user question, route it through a pipeline of specialized agents that (1) plan, (2) retrieve from a vector store, (3) optionally search the web, (4) reason over the gathered evidence, and (5) produce a grounded, cited answer. The conversation state is durably checkpointed in Postgres so runs are resumable and inspectable.

## 2. Agent Pipeline (from your diagram)

```
User Query
     │
     ▼
Planner Agent              → decompose question, decide which tools/sources are needed
     │
     ▼
Query Rewriter             → expand/clarify/multi-query rewrite for better recall
     │
     ▼
Hybrid Retrieval           → dense (Chroma) + sparse (client-side BM25) fused with RRF
 (Chroma + BM25)
     ├── Internal Knowledge → embedded document corpus
     └── Web Search         → (conditional) fill gaps via external search
     │
     ▼
Reranker                   → cross-encoder re-scores fused candidates, keeps top-N
     │
     ▼
Evidence Fusion            → dedupe, merge, group passages into a coherent context block
     │
     ▼
Reasoning Agent            → synthesize evidence, resolve conflicts, draft answer
     │
     ▼
Verification Agent         → fact-check draft against evidence; gate (loop back if unsupported)
     │
     ▼
Response Agent             → final grounded answer with citations
```

We implement this as a **LangGraph `StateGraph`**. Most edges are linear, but several nodes set flags that drive **conditional edges**:
- **Hybrid Retrieval** triggers **Web Search** only when the planner flagged it or internal hits are weak (top fused RRF score below threshold).
- **Verification Agent** is the loop-back gate: if the draft answer contains claims unsupported by the evidence, it loops back to **Query Rewriter** (broadened query) up to `MAX_ATTEMPTS`; otherwise it proceeds to **Response**.

## 3. Component Responsibilities

| Component | Library | Role |
|-----------|---------|------|
| Orchestration | LangGraph `StateGraph` | Node-per-agent, conditional routing, retry loops |
| State persistence | `langgraph-checkpoint-postgres` (`PostgresSaver`) | Durable checkpoints in Postgres; resume/inspect runs by `thread_id` |
| Vector store | Chroma (`chromadb`, local `HttpClient` or `CloudClient`) | Dense vectors + chunk text/metadata; one collection per embedding provider |
| Sparse / BM25 | `fastembed` (`Qdrant/bm25` model id) sparse embeddings | Lexical recall, computed **client-side** and fused with dense via RRF in Python |
| Embeddings | Gemini `gemini-embedding-001` (3072-dim) / Ollama / NVIDIA `nv-embedqa-e5-v5` (1024-dim) | Provider-selectable; non-Gemini dims auto-detected |
| Reranker | Cross-encoder (`fastembed` reranker, `ms-marco-MiniLM-L-6-v2`) | Re-scores fused candidates, keeps top-N before evidence fusion |
| LLM | Gemini `gemini-2.5-flash` / Ollama / NVIDIA `llama-3.3-70b` via LangChain chat models | Agent reasoning (provider-selectable per request) |
| Web search | Tavily (`langchain-tavily`) or DuckDuckGo fallback | External knowledge |
| Relational store | PostgreSQL (`psycopg`) | Document/source registry, ingestion log, citations |
| API | FastAPI | `/ingest`, `/query`, `/models`, `/health` + browser UI |
| Infra | docker-compose | Chroma + Postgres containers |

## 4. Shared Graph State

```python
class AgentState(TypedDict):
    question: str
    plan: dict                      # planner output: subqueries, needs_web, k
    rewritten_queries: list[str]    # query rewriter output (multi-query)
    retrieved: list[Document]       # hybrid (dense+sparse) hits, pre-rerank
    web_results: list[dict]         # web search hits (optional)
    reranked: list[Document]        # top-N after cross-encoder rerank
    evidence: str                   # fused, deduped context block + provenance map
    draft_answer: str               # reasoning agent draft
    verification: dict              # {supported: bool, unsupported_claims: [...]}
    retrieval_attempts: int         # loop guard
    answer: str                     # final, verified answer
    citations: list[dict]
    messages: Annotated[list, add_messages]
```

## 5. Routing Logic

- **Planner → Query Rewriter**: always.
- **Query Rewriter → Hybrid Retrieval**: always (runs dense Chroma + client-side sparse BM25, fused via RRF in Python).
- **Hybrid Retrieval → (Web Search | Reranker)**: conditional. If `plan.needs_web` or top fused RRF score < threshold → Web Search (then Reranker); else → Reranker directly.
- **Reranker → Evidence Fusion → Reasoning**: linear.
- **Reasoning → Verification**: always.
- **Verification → (Query Rewriter | Response)**: if `verification.supported == False` and `retrieval_attempts < MAX_ATTEMPTS` → loop back to **Query Rewriter** (broaden using `unsupported_claims`); else → Response.
- **Response → END**.

## 6. Data Model (Postgres)

```sql
-- Document/source registry (LangGraph checkpoint tables are auto-created by .setup())
CREATE TABLE documents (
    id          UUID PRIMARY KEY,
    source_uri  TEXT NOT NULL,
    title       TEXT,
    sha256      TEXT UNIQUE,          -- dedupe on re-ingest
    chunk_count INT,
    ingested_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chunks (
    id          UUID PRIMARY KEY,     -- == Chroma record id
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INT,
    text        TEXT,                 -- source of truth; Chroma also stores the chunk text
    token_count INT
);
```

Each embedding provider gets its own Chroma collection (`documents_<provider>`, cosine space), sized to that provider's dimension (Gemini 3072, NVIDIA/Ollama auto-detected). A record carries `{ id, embedding: [...dim], document: "<chunk text>", metadata: { document_id, ordinal, title, source_uri } }`. Chroma is **dense-only**, so there is no sparse vector stored — at query time the BM25 index is rebuilt client-side from the collection's documents and fused with Chroma's dense results via RRF (`rrf_k`, default 60). The BM25 index is cached in-process and invalidated after each ingest.

## 7. Ingestion Flow

1. Load source (file / raw text) → split with `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap).
2. Compute a provider-scoped `sha256`; skip if already in `documents` (idempotent per embedding provider).
3. Compute **dense** vectors per chunk via the active provider (Gemini batched ≤ 100 — API limit).
4. `add` records (dense embedding + chunk text + metadata) to the provider's Chroma collection; insert rows into `documents` + `chunks`.
5. Invalidate the in-process BM25 cache so the next query rebuilds it from the updated collection. (No sparse vectors are stored — BM25 is a query-time, client-side concern.)

## 8. Proposed File Layout

```
chroma_rag/
├── docker-compose.yml          # chroma + postgres
├── .env.example                # provider keys, TAVILY_API_KEY, DB/CHROMA config
├── pyproject.toml              # deps (uv / pip)
├── README.md
├── src/chroma_rag/
│   ├── config.py               # pydantic-settings: env → typed config
│   ├── clients.py              # chroma client (local/cloud), llm/embeddings, sparse, reranker
│   ├── db.py                   # schema bootstrap + document/chunk DAO
│   ├── ingest.py               # ingestion pipeline + CLI (dense → Chroma)
│   ├── retriever.py            # Chroma dense + client-side BM25, RRF fusion in Python
│   ├── rerank.py               # cross-encoder reranker
│   ├── state.py                # AgentState TypedDict
│   ├── models.py               # provider/model listing for the UI
│   ├── agents/
│   │   ├── planner.py
│   │   ├── query_rewriter.py
│   │   ├── hybrid_retrieval.py # internal knowledge (Chroma + client-side BM25)
│   │   ├── web_search.py       # conditional external search
│   │   ├── rerank_node.py      # cross-encoder rerank node
│   │   ├── evidence_fusion.py  # dedupe + merge into context block
│   │   ├── reasoning.py        # draft answer
│   │   ├── verification.py     # fact-check gate / loop-back
│   │   └── response.py
│   ├── graph.py                # StateGraph wiring + PostgresSaver
│   ├── api.py                  # FastAPI app
│   └── static/                 # browser UI
└── tests/
    ├── test_routing.py
    └── test_db.py
```

## 9. Key Dependencies

```
langgraph
langgraph-checkpoint-postgres
langchain
langchain-google-genai          # ChatGoogleGenerativeAI + GoogleGenerativeAIEmbeddings
langchain-ollama                # Ollama chat + embeddings provider
langchain-nvidia-ai-endpoints   # NVIDIA NIM chat + embeddings provider
chromadb                        # vector store (local HttpClient or CloudClient)
fastembed                       # client-side BM25 sparse embeddings + cross-encoder reranker
langchain-tavily                # web search (optional)
psycopg[binary,pool]            # Postgres driver for PostgresSaver + DAO
pydantic-settings
fastapi / uvicorn
```

## 10. Critical Implementation Notes (from current library behavior)

- **PostgresSaver** must be created with `autocommit=True` and `row_factory=dict_row`, and `.setup()` called once to create checkpoint tables. Set `LANGGRAPH_STRICT_MSGPACK=true` for safe deserialization.
- **Gemini embeddings** cap batch size at **100 strings** — chunk batching is required for large ingests. The Gemini dimension is **3072** (`EMBED_DIM`); Ollama and NVIDIA dimensions are auto-detected by embedding a probe string.
- **Per-provider collections.** A collection's dimension is fixed by its first `add`. `ensure_collection()` verifies an existing collection's dimension matches the active provider and refuses a mismatch — delete the collection to rebuild. Always query with the same provider you ingested with.
- **Hybrid search is split.** Chroma serves the dense ranking; the BM25 ranking is computed in Python from the collection's documents (cached in-process, invalidated on ingest) and the two are merged with **Reciprocal Rank Fusion** (`rrf_k`, default 60). There is no server-side fusion.
- **Local vs. Cloud Chroma.** `HttpClient` (local Docker, default) has no record quota; Chroma Cloud free tier caps at ~300 records. The client switches on whether `CHROMA_HOST` is set.
- **Reranker** runs *after* fusing internal + web candidates and *before* Evidence Fusion — it sees the full candidate pool so the top-N is globally best, not per-source.
- **Verification** is a real gate, not cosmetic: it must return structured `{supported, unsupported_claims}` (use structured output) so routing is deterministic. Cap loop-backs with `MAX_ATTEMPTS` to avoid infinite rewrite→retrieve→verify cycles.
- Use distinct models per node: a fast model for planner/rewriter/response, a stronger one for reasoning + verification. Provider + model are selectable per request via the API/UI.

## 11. Build Order (next steps)

1. `docker-compose.yml` + `.env.example` + deps → bring up Chroma & Postgres.
2. `config.py`, `clients.py`, `db.py` → connectivity + schema.
3. `retriever.py` + `rerank.py` → Chroma dense + client-side BM25 (RRF) retriever + cross-encoder.
4. `ingest.py` → load sample docs (dense vectors + text/metadata) into Chroma/Postgres.
5. `state.py` + the agent nodes (planner, query_rewriter, hybrid_retrieval, web_search, rerank_node, evidence_fusion, reasoning, verification, response).
6. `graph.py` → wire StateGraph with PostgresSaver + conditional edges (web-search branch, verification loop-back).
7. `api.py` → FastAPI `/ingest`, `/query`, `/models`, `/health` + browser UI.
8. `tests/` → ingestion helpers, routing, evidence fusion (no live services needed).

---

### Sources consulted
- [GoogleGenerativeAIEmbeddings — LangChain reference](https://reference.langchain.com/python/langchain-google-genai/embeddings/GoogleGenerativeAIEmbeddings) (batch limit 100, `output_dimensionality`)
- [langgraph-checkpoint-postgres — PyPI](https://pypi.org/project/langgraph-checkpoint-postgres/) and [PostgresSaver usage](https://reference.langchain.com/python/langgraph.checkpoint.postgres) (`autocommit`, `dict_row`, `.setup()`, strict msgpack)
