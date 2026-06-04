"""Ingestion pipeline: load -> split -> dedupe -> embed (dense+sparse) -> upsert.

CLI:
    python -m a2z_hunter.ingest path/to/file.txt [more files/dirs ...]
    python -m a2z_hunter.ingest --text "raw text" --title "My Note"
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct, SparseVector

from . import db
from .clients import embeddings, ensure_collection, qdrant_client, sparse_embedder
from .config import get_settings

GEMINI_BATCH = 100  # Gemini embeddings API caps batch size at 100 strings.


def _split(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_text(text)


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def ingest_text(text: str, *, source_uri: str, title: str) -> dict:
    """Idempotently ingest a single document's text. Returns a summary dict."""
    s = get_settings()
    ensure_collection()
    db.init_schema()

    sha = db.sha256_text(text)
    if db.document_exists(sha):
        return {"status": "skipped", "reason": "duplicate", "source_uri": source_uri}

    chunks = _split(text)
    if not chunks:
        return {"status": "skipped", "reason": "empty", "source_uri": source_uri}

    # Dense embeddings (batched to respect Gemini's 100-string limit).
    dense_vectors: list[list[float]] = []
    for batch in _batched(chunks, GEMINI_BATCH):
        dense_vectors.extend(embeddings().embed_documents(batch))

    # Sparse (BM25) embeddings.
    sparse_vectors = list(sparse_embedder().embed(chunks))

    document_id = uuid.uuid4()
    points: list[PointStruct] = []
    chunk_rows: list[dict] = []
    for ordinal, (chunk, dense, sparse) in enumerate(
        zip(chunks, dense_vectors, sparse_vectors)
    ):
        chunk_id = uuid.uuid4()
        points.append(
            PointStruct(
                id=str(chunk_id),
                vector={
                    s.dense_vector_name: dense,
                    s.sparse_vector_name: SparseVector(
                        indices=sparse.indices.tolist(), values=sparse.values.tolist()
                    ),
                },
                payload={
                    "document_id": str(document_id),
                    "ordinal": ordinal,
                    "title": title,
                    "source_uri": source_uri,
                    "text": chunk,
                },
            )
        )
        chunk_rows.append(
            {"id": chunk_id, "ordinal": ordinal, "text": chunk, "token_count": len(chunk.split())}
        )

    qdrant_client().upsert(collection_name=s.qdrant_collection, points=points)
    # Insert document row first (FK target), then its chunks.
    db.insert_document(
        source_uri=source_uri, title=title, sha256=sha,
        chunk_count=len(chunks), doc_id=document_id,
    )
    db.insert_chunks(document_id, chunk_rows)
    return {"status": "ingested", "chunks": len(chunks), "source_uri": source_uri}


def ingest_path(path: Path) -> list[dict]:
    results: list[dict] = []
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix in {".txt", ".md"}:
                results.append(
                    ingest_text(
                        child.read_text(encoding="utf-8", errors="ignore"),
                        source_uri=str(child),
                        title=child.name,
                    )
                )
    elif path.is_file():
        results.append(
            ingest_text(
                path.read_text(encoding="utf-8", errors="ignore"),
                source_uri=str(path),
                title=path.name,
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("paths", nargs="*", help="files or directories (.txt/.md)")
    parser.add_argument("--text", help="ingest a raw text string instead of files")
    parser.add_argument("--title", default="inline", help="title for --text")
    args = parser.parse_args(argv)

    results: list[dict] = []
    if args.text:
        results.append(ingest_text(args.text, source_uri="inline", title=args.title))
    for p in args.paths:
        results.extend(ingest_path(Path(p)))

    if not results:
        parser.print_help()
        return 1
    for r in results:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
