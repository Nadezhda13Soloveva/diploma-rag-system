from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from backend.ingestion.parsers.base import ParsedDocument


@dataclass
class DocumentChunk:
    content: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    source: str = ""
    filename: str = ""
    file_type: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    page: int | None = None
    title: str | None = None
    section: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "filename": self.filename,
            "file_type": self.file_type,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "page": self.page,
            "title": self.title,
            "section": self.section,
            "char_start": self.char_start,
            "char_end": self.char_end,
            **self.metadata,
        }


class BaseChunker(ABC):
    """Единый интерфейс стратегий чанкинга."""

    @abstractmethod
    def chunk(self, parsed: ParsedDocument) -> list[DocumentChunk]:
        """Разбивает ParsedDocument на список DocumentChunk."""
        ...
