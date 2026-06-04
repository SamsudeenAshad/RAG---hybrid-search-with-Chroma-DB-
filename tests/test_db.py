"""Pure helpers that don't need a live database."""
from __future__ import annotations

import os

os.environ.setdefault("GOOGLE_API_KEY", "test")

from a2z_hunter.db import sha256_text  # noqa: E402
from a2z_hunter.ingest import _batched, _split  # noqa: E402


def test_sha256_is_stable_and_dedupes():
    assert sha256_text("hello") == sha256_text("hello")
    assert sha256_text("hello") != sha256_text("world")


def test_split_overlaps_long_text():
    text = "sentence. " * 500
    chunks = _split(text)
    assert len(chunks) > 1
    assert all(chunks)


def test_batched_respects_gemini_limit():
    items = list(range(250))
    batches = list(_batched(items, 100))
    assert [len(b) for b in batches] == [100, 100, 50]
