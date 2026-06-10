"""Ingestion pipeline: load -> split -> dedupe -> embed (dense+sparse) -> upsert.

CLI:
    python -m chroma_rag.ingest path/to/file.txt [more files/dirs ...]
    python -m chroma_rag.ingest --text "raw text" --title "My Note"
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


def ingest_text(
    text: str, *, source_uri: str, title: str, embed_provider: str | None = None
) -> dict:
    """Idempotently ingest a single document's text. Returns a summary dict.

    embed_provider (gemini|ollama) selects the embedding model + target
    collection; None uses the configured default. Dedupe is scoped per
    provider so the same document can live in both collections.
    """
    from .clients import reset_embed_override, set_embed_override

    token = set_embed_override(embed_provider)
    try:
        return _ingest_text(text, source_uri=source_uri, title=title)
    finally:
        reset_embed_override(token)


def _ingest_text(text: str, *, source_uri: str, title: str) -> dict:
    from .clients import active_embed_provider

    s = get_settings()
    coll = ensure_collection()
    db.init_schema()

    provider = active_embed_provider()
    # Provider-scoped sha so the same text can index under each provider.
    sha = db.sha256_text(f"{provider}:{text}")
    if db.document_exists(sha):
        return {
            "status": "skipped", "reason": "duplicate",
            "source_uri": source_uri, "embed_provider": provider,
        }

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

    qdrant_client().upsert(collection_name=coll, points=points)
    # Insert document row first (FK target), then its chunks.
    db.insert_document(
        source_uri=source_uri, title=title, sha256=sha,
        chunk_count=len(chunks), doc_id=document_id,
    )
    db.insert_chunks(document_id, chunk_rows)
    return {
        "status": "ingested", "chunks": len(chunks),
        "source_uri": source_uri, "collection": coll,
    }


def ingest_path(path: Path, *, embed_provider: str | None = None) -> list[dict]:
    results: list[dict] = []
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix in {".txt", ".md"}:
                results.append(
                    ingest_text(
                        child.read_text(encoding="utf-8", errors="ignore"),
                        source_uri=str(child),
                        title=child.name,
                        embed_provider=embed_provider,
                    )
                )
    elif path.is_file():
        results.append(
            ingest_text(
                path.read_text(encoding="utf-8", errors="ignore"),
                source_uri=str(path),
                title=path.name,
                embed_provider=embed_provider,
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("paths", nargs="*", help="files or directories (.txt/.md)")
    parser.add_argument("--text", help="ingest a raw text string instead of files")
    parser.add_argument("--title", default="inline", help="title for --text")
    parser.add_argument(
        "--embed-provider", choices=["gemini", "ollama", "nvidia"], default=None,
        help="embedding provider (default: configured EMBED_PROVIDER)",
    )
    args = parser.parse_args(argv)

    results: list[dict] = []
    if args.text:
        results.append(
            ingest_text(
                args.text, source_uri="inline", title=args.title,
                embed_provider=args.embed_provider,
            )
        )
    for p in args.paths:
        results.extend(ingest_path(Path(p), embed_provider=args.embed_provider))

    if not results:
        parser.print_help()
        return 1
    for r in results:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
