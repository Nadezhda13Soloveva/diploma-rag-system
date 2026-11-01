from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from backend.ingestion.parsers.pdf_parser import PdfParser


@pytest.fixture
def parser() -> PdfParser:
    return PdfParser()


def _make_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text, fontsize=12)
    doc.save(path)
    doc.close()
    return path


def test_can_parse(parser: PdfParser) -> None:
    assert parser.can_parse(Path("doc.pdf")) is True
    assert parser.can_parse(Path("doc.PDF")) is True
    assert parser.can_parse(Path("doc.md")) is False


def test_extracts_text_per_page(parser: PdfParser, tmp_path: Path) -> None:
    # Длинный латинице-текст, чтобы avg_chars точно > threshold
    page1 = (
        "PostgreSQL recovery runbook. "
        "Check service status, inspect logs, verify connections. "
        "Step one: systemctl status postgresql. "
        "Step two: read /var/log/postgresql/postgresql.log."
    )
    page2 = (
        "Page two contains additional diagnostic procedures. "
        "Verify disk space, check replication lag, restart if needed."
    )
    pdf = _make_pdf(tmp_path / "sample.pdf", [page1, page2])
    doc = parser.parse(pdf)

    assert doc.file_type == "pdf"
    assert doc.filename == "sample.pdf"
    assert doc.metadata["page_count"] == 2
    assert doc.metadata["is_scanned"] is False

    page_texts = doc.metadata["page_texts"]
    assert len(page_texts) == 2
    assert page_texts[0]["page"] == 1
    assert "PostgreSQL" in page_texts[0]["text"]
    assert page_texts[1]["page"] == 2
    assert "Page two" in page_texts[1]["text"] or "additional" in page_texts[1]["text"]


def test_scanned_pdf_warning(parser: PdfParser, tmp_path: Path) -> None:
    # Страницы без текста → scanned
    pdf = _make_pdf(tmp_path / "scanned.pdf", ["", ""])
    doc = parser.parse(pdf)

    assert doc.metadata["is_scanned"] is True
    assert doc.metadata["page_count"] == 2


def test_empty_pdf(parser: PdfParser, tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    doc_pdf = pymupdf.open()
    doc_pdf.new_page()
    doc_pdf.save(path)
    doc_pdf.close()

    doc = parser.parse(path)

    assert doc.file_type == "pdf"
    assert doc.metadata["page_count"] == 1
    assert doc.metadata["is_scanned"] is True
    assert doc.metadata["pages"][0]["char_count"] == 0
    assert doc.metadata["page_texts"][0]["text"].strip() == ""


def test_page_numbers_in_metadata(parser: PdfParser, tmp_path: Path) -> None:
    pdf = _make_pdf(
        tmp_path / "pages.pdf",
        ["Enough text on this page so the parser does not mark it as scanned content."],
    )
    doc = parser.parse(pdf)

    pages_meta = doc.metadata["pages"]
    assert pages_meta[0]["page"] == 1
    assert pages_meta[0]["char_count"] > 0
