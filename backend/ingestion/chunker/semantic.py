from __future__ import annotations

import math
import re
from collections.abc import Callable

from backend.config import settings
from backend.ingestion.chunker.base import BaseChunker, DocumentChunk
from backend.ingestion.parsers.base import ParsedDocument

# embed_fn: список строк -> список векторов
EmbedFn = Callable[[list[str]], list[list[float]]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _split_sentences(text: str) -> list[str]:
    # простая сегментация; достаточно для MVP
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class SemanticChunker(BaseChunker):
    """
    Разбиение по смысловым границам:
    считаем эмбеддинги соседних предложений и режем там,
    где косинусная близость падает ниже threshold
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        similarity_threshold: float = 0.55,
        chunk_size: int | None = None,
        min_chunk_sentences: int = 1,
    ) -> None:
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunking.chunk_size
        self.min_chunk_sentences = min_chunk_sentences

    def chunk(self, parsed: ParsedDocument) -> list[DocumentChunk]:
        text = parsed.content or ""
        if not text.strip():
            return []

        sentences = _split_sentences(text)
        if not sentences:
            return []

        if self.embed_fn is None:
            # fallback: группируем по chunk_size
            return self._fallback_by_size(parsed, sentences)

        embeddings = self.embed_fn(sentences)
        if len(embeddings) != len(sentences):
            raise ValueError("embed_fn must return one vector per sentence")

        groups: list[list[str]] = []
        current: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = _cosine(embeddings[i - 1], embeddings[i])
            candidate = current + [sentences[i]]
            candidate_text = " ".join(candidate)

            should_split = (
                sim < self.similarity_threshold and len(current) >= self.min_chunk_sentences
            ) or len(candidate_text) > self.chunk_size

            if should_split:
                groups.append(current)
                current = [sentences[i]]
            else:
                current = candidate

        if current:
            groups.append(current)

        return self._to_chunks(parsed, groups)

    def _fallback_by_size(
        self, parsed: ParsedDocument, sentences: list[str]
    ) -> list[DocumentChunk]:
        groups: list[list[str]] = []
        current: list[str] = []
        for s in sentences:
            candidate = current + [s]
            if current and len(" ".join(candidate)) > self.chunk_size:
                groups.append(current)
                current = [s]
            else:
                current = candidate
        if current:
            groups.append(current)
        return self._to_chunks(parsed, groups)

    def _to_chunks(self, parsed: ParsedDocument, groups: list[list[str]]) -> list[DocumentChunk]:
        title = parsed.metadata.get("title")
        title_val = title if isinstance(title, str) else None
        chunks: list[DocumentChunk] = []
        for i, group in enumerate(groups):
            content = " ".join(group).strip()
            if not content:
                continue
            chunks.append(
                DocumentChunk(
                    content=content,
                    document_id=parsed.document_id,
                    source=parsed.source,
                    filename=parsed.filename,
                    file_type=parsed.file_type,
                    chunk_index=i,
                    total_chunks=0,
                    title=title_val,
                    metadata={
                        "chunker": "semantic",
                        "similarity_threshold": self.similarity_threshold,
                        "sentences": len(group),
                    },
                )
            )
        for i, c in enumerate(chunks):
            c.chunk_index = i
            c.total_chunks = len(chunks)
        return chunks
