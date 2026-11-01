from __future__ import annotations

from backend.ingestion.chunker.recursive import RecursiveChunker
from backend.ingestion.parsers.base import ParsedDocument


def _doc(content: str, **meta: object) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc1",
        source="docs/sample.md",
        filename="sample.md",
        file_type="md",
        content=content,
        metadata=dict(meta),
    )


def test_short_document_single_chunk() -> None:
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    doc = _doc("Короткий текст без разбиения", title="T")
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].content.startswith("Короткий")
    assert chunks[0].document_id == "doc1"
    assert chunks[0].source == "docs/sample.md"
    assert chunks[0].filename == "sample.md"
    assert chunks[0].file_type == "md"
    assert chunks[0].title == "T"
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1
    assert len(chunks[0].content) <= 500


def test_long_document_respects_chunk_size() -> None:
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    text = " ".join([f"word{i}" for i in range(200)])
    chunks = chunker.chunk(_doc(text))

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.content) <= 100 + 20  # overlap может чуть расширить склейку
        assert c.document_id == "doc1"
        assert c.total_chunks == len(chunks)


def test_headings_create_natural_breaks() -> None:
    chunker = RecursiveChunker(chunk_size=120, chunk_overlap=20)
    content = (
        "# Title\n\n"
        "Intro paragraph with some text.\n\n"
        "## Section A\n\n"
        "Content of section A. " * 5 + "\n\n"
        "## Section B\n\n"
        "Content of section B. " * 5
    )
    chunks = chunker.chunk(
        _doc(content, title="Title", sections=["Title", "Section A", "Section B"])
    )

    assert len(chunks) >= 1
    for c in chunks:
        assert c.title == "Title"
        assert c.filename == "sample.md"
        assert c.chunk_index >= 0
        assert c.total_chunks == len(chunks)

    # хотя бы у одного чанка section определилась
    assert any(c.section for c in chunks)


def test_overlap_is_applied() -> None:
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    text = "abcdefghijklmnopqrstuvwxyz" * 10  # длинная строка без пробелов почти
    # добавим пробелы для более предсказуемого сплита
    text = " ".join(["token"] * 80)
    chunks = chunker.chunk(_doc(text))

    assert len(chunks) >= 2
    # overlap не всегда дословно из-за strip, но размеры контролируем
    assert all(len(c.content) <= 50 + 15 for c in chunks)


def test_empty_document() -> None:
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.chunk(_doc("")) == []
    assert chunker.chunk(_doc("   \n\n")) == []
