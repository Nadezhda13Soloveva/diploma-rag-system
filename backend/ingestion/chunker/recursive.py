from __future__ import annotations

import re
from typing import Any

from backend.config import settings
from backend.ingestion.chunker.base import BaseChunker, DocumentChunk
from backend.ingestion.parsers.base import ParsedDocument

DEFAULT_SEPARATORS = ["\n## ", "\n### ", "\n\n", ". ", " "]


class RecursiveChunker(BaseChunker):
    """
    Разделители по приоритету: заголовки -> абзацы -> предложения -> пробелы
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunking.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunking.chunk_overlap
        )
        self.separators = separators or DEFAULT_SEPARATORS

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def chunk(self, parsed: ParsedDocument) -> list[DocumentChunk]:
        text = parsed.content or ""
        if not text.strip():
            return []

        raw_parts = self._split_text(text)
        merged = self._merge_with_overlap(raw_parts)

        title = parsed.metadata.get("title")
        if isinstance(title, str):
            title_val: str | None = title
        else:
            title_val = None

        chunks: list[DocumentChunk] = []
        search_from = 0
        for idx, part in enumerate(merged):
            content = part.strip()
            if not content:
                continue

            # позиция в исходном тексте (приблизительно)
            pos = text.find(content[: min(64, len(content))], search_from)
            if pos < 0:
                pos = search_from
            char_start = pos
            char_end = pos + len(content)
            search_from = max(char_start + 1, char_end - self.chunk_overlap)

            section = self._detect_section(content, parsed.metadata)
            page = self._detect_page(content, parsed.metadata)

            chunks.append(
                DocumentChunk(
                    content=content,
                    document_id=parsed.document_id,
                    source=parsed.source,
                    filename=parsed.filename,
                    file_type=parsed.file_type,
                    chunk_index=idx,
                    total_chunks=0,  # заполним ниже
                    page=page,
                    title=title_val,
                    section=section,
                    char_start=char_start,
                    char_end=char_end,
                    metadata={
                        "chunker": "recursive",
                        "chunk_size_config": self.chunk_size,
                        "chunk_overlap_config": self.chunk_overlap,
                    },
                )
            )

        total = len(chunks)
        for c in chunks:
            c.total_chunks = total
            # перенумерация после фильтра пустых
        for i, c in enumerate(chunks):
            c.chunk_index = i

        return chunks

    def _split_text(self, text: str) -> list[str]:
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            step = max(self.chunk_size - self.chunk_overlap, 1)
            return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

        sep = separators[0]
        rest = separators[1:]

        if sep == " ":
            pieces = text.split(" ")
        else:
            pieces = text.split(sep)

        # собираем куски, не превышая chunk_size
        # слишком большие - рекурсия
        results: list[str] = []
        current = ""

        for i, piece in enumerate(pieces):
            # возвращаем разделитель всем, кроме первого куска
            piece_with_sep = piece if i == 0 or sep == " " else sep + piece
            if sep == " " and current:
                candidate = f"{current} {piece}"
            else:
                candidate = piece_with_sep if not current else current + piece_with_sep

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    results.append(current.strip())
                if len(piece_with_sep) > self.chunk_size:
                    results.extend(self._recursive_split(piece_with_sep.strip(), rest))
                    current = ""
                else:
                    current = piece_with_sep.strip()

        if current.strip():
            results.append(current.strip())
        return results

    def _merge_with_overlap(self, parts: list[str]) -> list[str]:
        """Склеивает мелкие части и добавляет overlap между финальными чанками"""
        if not parts:
            return []

        # мержим  пока влезает в chunk_size
        merged: list[str] = []
        current = ""
        for part in parts:
            if not current:
                current = part
                continue
            candidate = f"{current}\n\n{part}" if not current.endswith("\n") else current + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                merged.append(current)
                current = part
        if current:
            merged.append(current)

        # overlap
        if self.chunk_overlap <= 0 or len(merged) <= 1:
            return merged

        with_overlap: list[str] = [merged[0]]
        for i in range(1, len(merged)):
            prev = with_overlap[-1]
            overlap = prev[-self.chunk_overlap :]
            nxt = (overlap + "\n" + merged[i]).strip()
            if len(nxt) > self.chunk_size * 2:
                # safety
                with_overlap.append(merged[i])
            else:
                with_overlap.append(
                    nxt if len(nxt) <= self.chunk_size + self.chunk_overlap else merged[i]
                )
        return with_overlap

    def _detect_section(self, content: str, metadata: dict[str, Any]) -> str | None:
        # заголовок в начале чанка
        m = re.search(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)
        if m:
            return m.group(2).strip()
        sections = metadata.get("sections")
        if isinstance(sections, list) and sections:
            first = sections[0]
            return first if isinstance(first, str) else None
        section = metadata.get("section")
        return section if isinstance(section, str) else None

    def _detect_page(self, content: str, metadata: dict[str, Any]) -> int | None:
        m = re.search(r"---\s*Page\s+(\d+)\s*---", content)
        if m:
            return int(m.group(1))
        page_texts = metadata.get("page_texts")
        if isinstance(page_texts, list) and page_texts:
            snippet = content[:80]
            for p in page_texts:
                if not isinstance(p, dict):
                    continue
                ptext = p.get("text") or ""
                if snippet and snippet[:40] in ptext:
                    page = p.get("page")
                    return int(page) if page is not None else None
        return None
