from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from backend.ingestion.chunker.base import DocumentChunk


@dataclass
class RetrievalResult:
    """Результат векторного поиска по одному чанку"""

    content: str
    score: float  # similarity (чем выше, тем лучше)
    rank: int  # 1-based
    chunk_id: str = ""
    document_id: str = ""
    source: str = ""
    filename: str = ""
    file_type: str = ""
    page: int | None = None
    title: str | None = None
    section: str | None = None
    distance: float | None = None  # сырое расстояние из backend
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "filename": self.filename,
            "file_type": self.file_type,
            "page": self.page,
            "title": self.title,
            "section": self.section,
            "distance": self.distance,
            "metadata": self.metadata,
        }


class BaseVectorStore(ABC):
    """Единый интерфейс векторных хранилищ"""

    @abstractmethod
    def add_documents(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Индексация чанков вместе с эмбеддингами и metadata"""
        ...

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Поиск top_k ближайших чанков со score и rank"""
        ...

    @abstractmethod
    def delete_collection(self) -> None:
        """Удаление всей коллекции/индекс"""
        ...

    @abstractmethod
    def count(self) -> int: ...
