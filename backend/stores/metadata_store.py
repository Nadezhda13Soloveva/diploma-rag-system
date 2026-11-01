from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentRecord:
    document_id: str
    source: str
    filename: str
    file_type: str
    chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MetadataStore:
    """JSON-хранилище метаданных документов (согласованность с vector/BM25)."""

    def __init__(self, path: str | Path = "./data/index/metadata.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._docs: dict[str, DocumentRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._docs = {}
            return
        with self.path.open(encoding="utf-8") as f:
            raw = json.load(f)
        self._docs = {doc_id: DocumentRecord(**payload) for doc_id, payload in raw.items()}

    def save(self) -> None:
        payload = {doc_id: asdict(rec) for doc_id, rec in self._docs.items()}
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("metadata_saved", path=str(self.path), documents=len(self._docs))

    def has_document(self, document_id: str) -> bool:
        return document_id in self._docs

    def upsert(self, record: DocumentRecord) -> None:
        self._docs[record.document_id] = record

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._docs.get(document_id)

    def all_chunk_ids(self) -> list[str]:
        ids: list[str] = []
        for rec in self._docs.values():
            ids.extend(rec.chunk_ids)
        return ids

    def list_documents(self) -> list[DocumentRecord]:
        return list(self._docs.values())

    def remove(self, document_id: str) -> None:
        self._docs.pop(document_id, None)
