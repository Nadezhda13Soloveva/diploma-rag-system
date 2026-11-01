from __future__ import annotations

from backend.config import settings
from backend.ingestion.chunker.base import BaseChunker
from backend.ingestion.chunker.fixed import FixedChunker
from backend.ingestion.chunker.recursive import RecursiveChunker
from backend.ingestion.chunker.semantic import EmbedFn, SemanticChunker


def get_chunker(
    strategy: str | None = None,
    *,
    embed_fn: EmbedFn | None = None,
) -> BaseChunker:
    name = (strategy or settings.chunking.strategy).lower()

    if name == "fixed":
        return FixedChunker()
    if name == "recursive":
        return RecursiveChunker()
    if name == "semantic":
        return SemanticChunker(embed_fn=embed_fn)
    if name == "parent_document":
        return RecursiveChunker()

    raise ValueError(f"Unknown chunking strategy: {name}")
