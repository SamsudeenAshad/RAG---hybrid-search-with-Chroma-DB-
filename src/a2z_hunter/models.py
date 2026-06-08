"""Enumerate selectable LLM providers + models for the UI dropdown.

Gemini models are a curated static list; Ollama models are fetched live from the
configured server (gracefully empty if unreachable).
"""
from __future__ import annotations

import urllib.request
import json

from .config import get_settings

# Curated Gemini chat models known to support generateContent.
_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-pro-latest",
]

# Curated NVIDIA NIM chat models (hosted at integrate.api.nvidia.com).
_NVIDIA_MODELS = [
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "nvidia/llama-3.1-nemotron-70b-instruct",
]


def _ollama_models() -> list[str]:
    s = get_settings()
    url = s.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.load(resp)
        return sorted(m["name"] for m in data.get("models", []))
    except Exception:
        return []


def list_providers() -> dict:
    """Return providers, their available models, and the configured defaults."""
    s = get_settings()
    ollama = _ollama_models()
    ollama_up = bool(ollama)
    return {
        "default_provider": s.llm_provider,
        "default_embed_provider": s.embed_provider,
        # Embedding-provider selection (separate from chat LLM; each provider
        # uses its own Qdrant collection, sized to its dimension).
        "embed_providers": [
            {
                "id": "gemini",
                "label": f"Gemini ({s.gemini_embed_model.split('/')[-1]}, {s.embed_dim}d)",
                "available": bool(s.google_api_key),
            },
            {
                "id": "ollama",
                "label": f"Ollama ({s.ollama_embed_model})",
                "available": ollama_up,
            },
            {
                "id": "nvidia",
                "label": f"NVIDIA ({s.nvidia_embed_model.split('/')[-1]})",
                "available": bool(s.nvidia_api_key),
            },
        ],
        "providers": [
            {
                "id": "gemini",
                "label": "Google Gemini",
                "available": bool(s.google_api_key),
                "models": _GEMINI_MODELS,
                "default_model": s.gemini_chat_model,
            },
            {
                "id": "ollama",
                "label": "Ollama (self-hosted)",
                "available": bool(ollama),
                "models": ollama,
                "default_model": s.ollama_model,
                "base_url": s.ollama_base_url,
            },
            {
                "id": "nvidia",
                "label": "NVIDIA NIM",
                "available": bool(s.nvidia_api_key),
                "models": _NVIDIA_MODELS,
                "default_model": s.nvidia_model,
            },
        ],
    }
