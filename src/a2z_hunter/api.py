"""FastAPI surface: ingest documents and query the agentic pipeline."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import ingest as ingest_mod
from .graph import run_query

app = FastAPI(title="a2z_hunter — Agentic Vector Search")

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


class IngestRequest(BaseModel):
    text: str
    title: str = "inline"
    source_uri: str = "inline"


class QueryRequest(BaseModel):
    question: str
    thread_id: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    return ingest_mod.ingest_text(
        req.text, source_uri=req.source_uri, title=req.title
    )


@app.post("/query")
def query(req: QueryRequest) -> dict:
    thread_id = req.thread_id or str(uuid.uuid4())
    final = run_query(req.question, thread_id=thread_id)
    return {
        "thread_id": thread_id,
        "answer": final.get("answer", ""),
        "citations": final.get("citations", []),
        "verification": final.get("verification", {}),
        "attempts": final.get("retrieval_attempts", 0),
    }


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


def run() -> None:
    import os
    import uvicorn

    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("a2z_hunter.api:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    run()
