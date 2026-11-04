from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query
from pydantic import BaseModel

from backend.config import settings
from backend.stores.metadata_store import MetadataStore

router = APIRouter(tags=["documents"])


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    source: str
    file_type: str
    chunk_count: int
    date_indexed: str | None = None


class DocumentsResponse(BaseModel):
    items: list[DocumentInfo]
    total: int
    limit: int
    offset: int


@router.get("/documents", response_model=DocumentsResponse)
def list_documents(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> DocumentsResponse:
    meta_path = Path(settings.paths.index) / "metadata.json"
    store = MetadataStore(meta_path)

    docs = store.list_documents()
    # сначала новые, если даты нет - по filename
    docs_sorted = sorted(
        docs,
        key=lambda d: (d.indexed_at or "", d.filename),
        reverse=True,
    )
    total = len(docs_sorted)
    page = docs_sorted[offset : offset + limit]

    items = [
        DocumentInfo(
            document_id=d.document_id,
            filename=d.filename,
            source=d.source,
            file_type=d.file_type,
            chunk_count=len(d.chunk_ids),
            date_indexed=d.indexed_at,
        )
        for d in page
    ]
    return DocumentsResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )
