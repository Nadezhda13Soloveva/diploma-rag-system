from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config import settings
from backend.ingestion.indexer import Indexer
from backend.observability.logging import get_logger
from backend.stores.chroma_store import ChromaVectorStore
from backend.stores.metadata_store import MetadataStore

logger = get_logger(__name__)

router = APIRouter(tags=["indexing"])


class ReindexRequest(BaseModel):
    force: bool = Field(
        default=False,
        description="Если true - полная переиндексация (очистка metadata/vector/BM25)",
    )


class ReindexResponse(BaseModel):
    indexed: int
    duration_ms: float
    stats: dict[str, Any]
    force: bool


def _clear_indexes() -> None:
    """Полная очистка stores перед force-reindex"""
    vector = ChromaVectorStore()
    vector.delete_collection()
    logger.info("reindex_vector_cleared")

    meta_path = f"{settings.paths.index}/metadata.json"
    meta = MetadataStore(meta_path)
    for doc in list(meta.list_documents()):
        meta.remove(doc.document_id)
    meta.save()
    logger.info("reindex_metadata_cleared")

    bm25_path = f"{settings.paths.index}/bm25.pkl"
    from pathlib import Path

    path = Path(bm25_path)
    if path.exists():
        path.unlink()
        logger.info("reindex_bm25_cleared", path=bm25_path)


@router.post("/reindex", response_model=ReindexResponse)
def reindex(body: ReindexRequest | None = None) -> ReindexResponse:
    """
    Запускает offline Indexer по data/documents/
    force=true - удаляет текущие индексы и строит заново
    """
    req = body or ReindexRequest()
    start = time.perf_counter()

    try:
        if req.force:
            logger.info("reindex_force_start")
            _clear_indexes()

        indexer = Indexer()
        stats = indexer.run()
    except Exception as exc:  # noqa: BLE001
        logger.exception("reindex_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Reindex failed: {exc}") from exc

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    indexed = int(stats.get("chunks_indexed", 0))

    logger.info(
        "reindex_done",
        indexed=indexed,
        duration_ms=duration_ms,
        force=req.force,
        stats=stats,
    )

    return ReindexResponse(
        indexed=indexed,
        duration_ms=duration_ms,
        stats=stats,
        force=req.force,
    )
