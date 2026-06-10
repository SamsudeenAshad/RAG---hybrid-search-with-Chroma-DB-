"""FastAPI surface: ingest documents and query the agentic pipeline."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import ingest as ingest_mod
from .graph import run_query, run_query_stream
from .models import list_providers

app = FastAPI(title="chroma_rag — Agentic Vector Search")

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


class IngestRequest(BaseModel):
    text: str
    title: str = "inline"
    source_uri: str = "inline"
    embed_provider: str | None = None  # "gemini" | "ollama"; None => default


class QueryRequest(BaseModel):
    question: str
    thread_id: str | None = None
    provider: str | None = None        # chat LLM: "gemini" | "ollama"
    model: str | None = None           # specific model id; None => provider default
    embed_provider: str | None = None  # embeddings: "gemini" | "ollama"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def models() -> dict:
    """Providers + available models for the UI dropdown."""
    return list_providers()


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    return ingest_mod.ingest_text(
        req.text, source_uri=req.source_uri, title=req.title,
        embed_provider=req.embed_provider,
    )


@app.post("/query")
def query(req: QueryRequest) -> dict:
    thread_id = req.thread_id or str(uuid.uuid4())
    final = run_query(
        req.question, thread_id=thread_id, provider=req.provider,
        model=req.model, embed_provider=req.embed_provider,
    )
    return {
        "thread_id": thread_id,
        "answer": final.get("answer", ""),
        "citations": final.get("citations", []),
        "verification": final.get("verification", {}),
        "attempts": final.get("retrieval_attempts", 0),
    }


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/query/stream")
def query_stream(
    question: str,
    thread_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    embed_provider: str | None = None,
) -> StreamingResponse:
    """Server-Sent Events: stream the pipeline's progress, then the final result."""
    tid = thread_id or str(uuid.uuid4())

    def event_gen():
        # Tell the client its thread id up front.
        yield f"data: {json.dumps({'type': 'thread', 'thread_id': tid})}\n\n"
        for event in run_query_stream(
            question, thread_id=tid, provider=provider, model=model,
            embed_provider=embed_provider,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def run() -> None:
    import os
    import uvicorn

    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("chroma_rag.api:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    run()
