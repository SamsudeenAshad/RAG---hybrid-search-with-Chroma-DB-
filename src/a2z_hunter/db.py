"""PostgreSQL schema bootstrap + document/chunk data-access helpers.

LangGraph's checkpoint tables are created separately by PostgresSaver.setup()
(see graph.py). Here we only manage the application's document registry.
"""
from __future__ import annotations

import hashlib
import uuid
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY,
    source_uri  TEXT NOT NULL,
    title       TEXT,
    sha256      TEXT UNIQUE,
    chunk_count INT,
    ingested_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INT,
    text        TEXT,
    token_count INT
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
"""


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(get_settings().database_url, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


def init_schema() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_exists(sha256: str) -> bool:
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE sha256 = %s", (sha256,)
        ).fetchone()
        return row is not None


def insert_document(
    *, source_uri: str, title: str, sha256: str, chunk_count: int
) -> uuid.UUID:
    doc_id = uuid.uuid4()
    with connect() as conn:
        conn.execute(
            """INSERT INTO documents (id, source_uri, title, sha256, chunk_count)
               VALUES (%s, %s, %s, %s, %s)""",
            (str(doc_id), source_uri, title, sha256, chunk_count),
        )
        conn.commit()
    return doc_id


def insert_chunks(document_id: uuid.UUID, chunks: list[dict]) -> None:
    """chunks: list of {id, ordinal, text, token_count}."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (id, document_id, ordinal, text, token_count)
                   VALUES (%s, %s, %s, %s, %s)""",
                [
                    (str(c["id"]), str(document_id), c["ordinal"], c["text"], c["token_count"])
                    for c in chunks
                ],
            )
        conn.commit()
