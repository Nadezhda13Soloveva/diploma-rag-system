from __future__ import annotations

from backend.config import settings
from backend.ingestion.chunker.base import BaseChunker, DocumentChunk
from backend.ingestion.parsers.base import ParsedDocument


class FixedChunker(BaseChunker):
    """Разбиение текста на окна фиксированной длины"""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunking.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunking.chunk_overlap
        )
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def chunk(self, parsed: ParsedDocument) -> list[DocumentChunk]:
        text = parsed.content or ""
        if not text.strip():
            return []

        step = max(self.chunk_size - self.chunk_overlap, 1)
        pieces: list[str] = []
        for start in range(0, len(text), step):
            end = min(start + self.chunk_size, len(text))
            piece = text[start:end]
            if piece.strip():
                pieces.append(piece)
            if end >= len(text):
                break

        title = parsed.metadata.get("title")
        title_val = title if isinstance(title, str) else None

        chunks: list[DocumentChunk] = []
        for i, content in enumerate(pieces):
            chunks.append(
                DocumentChunk(
                    content=content,
                    document_id=parsed.document_id,
                    source=parsed.source,
                    filename=parsed.filename,
                    file_type=parsed.file_type,
                    chunk_index=i,
                    total_chunks=len(pieces),
                    title=title_val,
                    section=parsed.metadata.get("section")
                    if isinstance(parsed.metadata.get("section"), str)
                    else None,
                    char_start=i * step,
                    char_end=min(i * step + len(content), len(text)),
                    metadata={
                        "chunker": "fixed",
                        "chunk_size_config": self.chunk_size,
                        "chunk_overlap_config": self.chunk_overlap,
                    },
                )
            )
        return chunks
