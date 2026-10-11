from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter

from backend.config import settings
from backend.stores.chroma_store import ChromaVectorStore

router = APIRouter(tags=["health"])


def _check_ollama() -> dict[str, Any]:
    """проверяет доступность Ollama и список моделей"""
    base_url = settings.embedding.base_url.rstrip("/")
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            return {
                "ok": True,
                "models": models,
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "models": [],
        }


def _check_vector_store() -> dict[str, Any]:
    """проверяет доступность Chroma (или другого бэка)"""
    try:
        if settings.vector_store.backend == "chroma":
            store = ChromaVectorStore(
                persist_directory=settings.vector_store.persist_directory,
                collection_name=settings.vector_store.collection_name,
            )
            count = store.count()
            return {
                "ok": True,
                "backend": "chroma",
                "count": count,
            }
        return {
            "ok": False,
            "backend": settings.vector_store.backend,
            "error": "backend not implemented in health check",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "backend": settings.vector_store.backend,
            "error": str(exc),
        }


@router.get("/health")
def health() -> dict[str, Any]:
    ollama = _check_ollama()
    vector_store = _check_vector_store()

    status = "ok" if ollama["ok"] and vector_store["ok"] else "degraded"

    return {
        "status": status,
        "ollama": ollama["ok"],
        "vector_store": vector_store["ok"],
        "details": {
            "ollama": ollama,
            "vector_store": vector_store,
            "embedding_model": settings.embedding.model,
            "llm_model": settings.llm.model,
        },
    }
