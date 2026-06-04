# Agentic Vector Search — Architecture & Implementation Plan

> Multi-agent retrieval system using **LangGraph** (orchestration), **LangChain** (LLM/embedding/retriever abstractions), **Qdrant** (vector store), **PostgreSQL** (graph checkpointer + document metadata), and **Google Gemini** (LLM + embeddings).

---

## 1. Goal

Given a user question, route it through a pipeline of specialized agents that (1) plan, (2) retrieve from a vector store, (3) optionally search the web, (4) reason over the gathered evidence, and (5) produce a grounded, cited answer. The conversation state is durably checkpointed in Postgres so runs are resumable and inspectable.

## 2. Agent Pipeline (from your diagram)

```
User Request
     │
     ▼
Planner Agent      → decompose question, decide which tools are needed
     │
     ▼
Vector Search Agent → embed query, retrieve top-k chunks from Qdrant
     │
     ▼
Web Search Agent   → (conditional) fill knowledge gaps via web search
     │
     ▼
Reasoning Agent    → synthesize evidence, resolve conflicts, check sufficiency
     │
     ▼
Response Agent     → final grounded answer with citations
```

We implement this as a **LangGraph `StateGraph`**. Edges are mostly linear, but the Planner sets flags that drive **conditional edges** (e.g. skip Web Search when vector hits are strong enough; loop back to retrieval if Reasoning judges evidence insufficient).

## 3. Component Responsibilities

| Component | Library | Role |
|-----------|---------|------|
| Orchestration | LangGraph `StateGraph` | Node-per-agent, conditional routing, retry loops |
| State persistence | `langgraph-checkpoint-postgres` (`PostgresSaver`) | Durable checkpoints in Postgres; resume/inspect runs by `thread_id` |
| Vector store | Qdrant (`langchain-qdrant`) | Stores embedded document chunks + payload metadata |
| Embeddings | Gemini `text-embedding-004` via `GoogleGenerativeAIEmbeddings` | 768-dim vectors (configurable via `output_dimensionality`) |
| LLM | Gemini `gemini-2.0-flash` (fast nodes), `gemini-1.5-pro` (reasoning) via `ChatGoogleGenerativeAI` | Agent reasoning |
| Web search | Tavily (`langchain-tavily`) or DuckDuckGo fallback | External knowledge |
| Relational store | PostgreSQL (`psycopg`) | Document/source registry, ingestion log, citations |
| API | FastAPI | `/ingest`, `/query`, `/threads/{id}` endpoints |
| Infra | docker-compose | Qdrant + Postgres containers |

## 4. Shared Graph State

```python
class AgentState(TypedDict):
    question: str
    plan: dict                      # planner output: subqueries, needs_web, k
    retrieved: list[Document]       # Qdrant hits
    web_results: list[dict]         # web search hits (optional)
    evidence_sufficient: bool       # reasoning verdict
    retrieval_attempts: int         # loop guard
    answer: str
    citations: list[dict]
    messages: Annotated[list, add_messages]
```

## 5. Routing Logic

- **Planner → Vector Search**: always.
- **Vector Search → (Web Search | Reasoning)**: conditional. If `plan.needs_web` or top score < threshold → Web Search; else → Reasoning.
- **Reasoning → (Vector Search | Response)**: if `evidence_sufficient == False` and `retrieval_attempts < MAX_ATTEMPTS` → loop back to Vector Search (broadened query); else → Response.
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
    id          UUID PRIMARY KEY,     -- == Qdrant point id
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INT,
    text        TEXT,                 -- source of truth; Qdrant payload mirrors a snippet
    token_count INT
);
```

Qdrant collection `documents` holds one point per chunk: `{ id, vector(768), payload: { document_id, ordinal, title, source_uri, text } }`.

## 7. Ingestion Flow

1. Load source (file / URL) → `langchain` document loader.
2. Split with `RecursiveCharacterTextSplitter` (e.g. 1000 chars, 150 overlap).
3. Compute `sha256`; skip if already in `documents` (idempotent).
4. Embed chunks with Gemini (batch ≤ 100 — Gemini API limit).
5. Upsert points into Qdrant; insert rows into `documents` + `chunks`.

## 8. Proposed File Layout

```
a2z_hunter/
├── docker-compose.yml          # qdrant + postgres
├── .env.example                # GOOGLE_API_KEY, TAVILY_API_KEY, DB/QDRANT urls
├── pyproject.toml              # deps (uv / pip)
├── README.md
├── src/a2z_hunter/
│   ├── config.py               # pydantic-settings: env → typed config
│   ├── clients.py              # qdrant client, gemini llm/embeddings, pg pool
│   ├── db.py                   # schema bootstrap + document/chunk DAO
│   ├── ingest.py               # ingestion pipeline + CLI
│   ├── retriever.py            # Qdrant retriever wrapper
│   ├── state.py                # AgentState TypedDict
│   ├── agents/
│   │   ├── planner.py
│   │   ├── vector_search.py
│   │   ├── web_search.py
│   │   ├── reasoning.py
│   │   └── response.py
│   ├── graph.py                # StateGraph wiring + PostgresSaver
│   └── api.py                  # FastAPI app
└── tests/
    ├── test_ingest.py
    └── test_graph.py
```

## 9. Key Dependencies

```
langgraph
langgraph-checkpoint-postgres
langchain
langchain-google-genai          # ChatGoogleGenerativeAI + GoogleGenerativeAIEmbeddings
langchain-qdrant
qdrant-client
langchain-tavily                # web search (optional)
psycopg[binary,pool]            # Postgres driver for PostgresSaver + DAO
pydantic-settings
fastapi / uvicorn
```

## 10. Critical Implementation Notes (from current library behavior)

- **PostgresSaver** must be created with `autocommit=True` and `row_factory=dict_row`, and `.setup()` called once to create checkpoint tables. Set `LANGGRAPH_STRICT_MSGPACK=true` for safe deserialization.
- **Gemini embeddings** cap batch size at **100 strings** — chunk batching is required for large ingests. Dimension is 768 by default; `output_dimensionality` can reduce it (must match the Qdrant collection's configured vector size).
- Qdrant collection vector size **must equal** the embedding dimension — assert this at startup.
- Use distinct Gemini models per node: `flash` for planner/response (cheap/fast), `pro` for reasoning (quality).

## 11. Build Order (next steps)

1. `docker-compose.yml` + `.env.example` + deps → bring up Qdrant & Postgres.
2. `config.py`, `clients.py`, `db.py` → connectivity + schema.
3. `ingest.py` → load sample docs into Qdrant/Postgres.
4. `state.py` + the 5 agent nodes.
5. `graph.py` → wire StateGraph with PostgresSaver + conditional edges.
6. `api.py` → FastAPI `/ingest` and `/query`.
7. `tests/` → ingestion idempotency + a graph smoke test.

---

### Sources consulted
- [GoogleGenerativeAIEmbeddings — LangChain reference](https://reference.langchain.com/python/langchain-google-genai/embeddings/GoogleGenerativeAIEmbeddings) (batch limit 100, `output_dimensionality`)
- [langgraph-checkpoint-postgres — PyPI](https://pypi.org/project/langgraph-checkpoint-postgres/) and [PostgresSaver usage](https://reference.langchain.com/python/langgraph.checkpoint.postgres) (`autocommit`, `dict_row`, `.setup()`, strict msgpack)
