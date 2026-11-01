from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pymupdf as fitz

from backend.ingestion.parsers.base import BaseParser, ParsedDocument
from backend.observability.logging import get_logger

logger = get_logger(__name__)

_MIN_CHARS_PER_PAGE = 50


class PdfParser(BaseParser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def parse(self, path: Path) -> ParsedDocument:
        path = Path(path)
        document_id = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]

        try:
            doc = fitz.open(path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("pdf_open_failed", path=str(path))
            raise ValueError(f"Не удалось открыть PDF: {path}") from exc

        try:
            page_count = doc.page_count
            page_texts: list[dict[str, Any]] = []
            total_chars = 0

            for i in range(page_count):
                page = doc.load_page(i)
                text = page.get_text("text") or ""
                text = text.replace("\r\n", "\n").replace("\r", "\n")
                char_count = len(text.strip())
                total_chars += char_count
                page_texts.append(
                    {
                        "page": i + 1,
                        "text": text,
                        "char_count": char_count,
                    }
                )

            avg_chars = (total_chars / page_count) if page_count > 0 else 0.0
            is_scanned = page_count > 0 and avg_chars < _MIN_CHARS_PER_PAGE

            if is_scanned:
                logger.warning(
                    "pdf_scanned_detected",
                    path=str(path),
                    page_count=page_count,
                    avg_chars_per_page=round(avg_chars, 2),
                    threshold=_MIN_CHARS_PER_PAGE,
                    message="Текстовый слой почти отсутствует; OCR пока не поддерживается",
                )

            structured = self._extract_structured_markdown(path, doc)

            if structured and structured.strip():
                content = structured.strip() + "\n"
            else:
                content = self._join_pages(page_texts)

            title = self._extract_title(doc, content)

            metadata: dict[str, Any] = {
                "page_count": page_count,
                "is_scanned": is_scanned,
                "avg_chars_per_page": round(avg_chars, 2),
                "pages": [{"page": p["page"], "char_count": p["char_count"]} for p in page_texts],
            }
            if title:
                metadata["title"] = title

            # полные тексты страниц для чанкера (page в metadata чанка)
            metadata["page_texts"] = [{"page": p["page"], "text": p["text"]} for p in page_texts]

            return ParsedDocument(
                document_id=document_id,
                source=str(path),
                filename=path.name,
                file_type="pdf",
                content=content,
                metadata=metadata,
                raw_metadata={
                    "page_count": page_count,
                    "is_scanned": is_scanned,
                    "parser": "PdfParser",
                    "engine": "pymupdf",
                },
            )
        finally:
            doc.close()

    def _join_pages(self, page_texts: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for p in page_texts:
            body = (p["text"] or "").strip()
            parts.append(f"--- Page {p['page']} ---\n{body}")
        return "\n\n".join(parts).strip() + ("\n" if parts else "")

    def _extract_structured_markdown(self, path: Path, doc: fitz.Document) -> str | None:
        """Пытаемся получить Markdown со структурой заголовков через pymupdf4llm"""
        try:
            import pymupdf4llm

            md: str = pymupdf4llm.to_markdown(str(path))
            return md
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "pymupdf4llm_unavailable_or_failed",
                path=str(path),
                error=str(exc),
            )
            return None

    def _extract_title(self, doc: fitz.Document, content: str) -> str | None:
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip()
        if title:
            return title

        # fallback: первая непустая строка / markdown H1
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip() or None
            if line and not line.startswith("---"):
                return line[:200]
        return None
