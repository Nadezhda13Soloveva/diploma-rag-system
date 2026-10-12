from __future__ import annotations

from pathlib import Path

import pytest

from backend.ingestion.parsers.markdown_parser import MarkdownParser

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_MD = FIXTURES / "sample.md"


@pytest.fixture
def parser() -> MarkdownParser:
    return MarkdownParser()


def test_can_parse_md(parser: MarkdownParser) -> None:
    assert parser.can_parse(Path("docs/runbook.md")) is True
    assert parser.can_parse(Path("readme.markdown")) is True
    assert parser.can_parse(Path("file.txt")) is False
    assert parser.can_parse(Path("file.pdf")) is False


def test_parse_sample_md(parser: MarkdownParser) -> None:
    doc = parser.parse(SAMPLE_MD)

    assert doc.file_type == "md"
    assert doc.filename == "sample.md"
    assert doc.source.endswith("sample.md")
    assert doc.document_id
    assert "Восстановление PostgreSQL" in doc.content
    assert "2.3 Диагностика" in doc.content


def test_title_from_frontmatter_or_h1(parser: MarkdownParser) -> None:
    doc = parser.parse(SAMPLE_MD)
    assert doc.metadata.get("title") == "Восстановление PostgreSQL"


def test_sections_extracted(parser: MarkdownParser) -> None:
    doc = parser.parse(SAMPLE_MD)

    sections = doc.metadata.get("sections", [])
    assert "Восстановление PostgreSQL" in sections
    assert "2.3 Диагностика" in sections
    assert "Проверка подключений" in sections
    assert "Откат" in sections


def test_frontmatter_preserved(parser: MarkdownParser) -> None:
    doc = parser.parse(SAMPLE_MD)

    fm = doc.metadata.get("frontmatter") or doc.raw_metadata.get("frontmatter")
    assert isinstance(fm, dict)
    assert fm.get("author") == "SRE Team"
    assert "database" in (fm.get("tags") or [])


def test_parse_without_frontmatter(parser: MarkdownParser, tmp_path: Path) -> None:
    md_file = tmp_path / "simple.md"
    md_file.write_text("# Только заголовок\n\nТекст без frontmatter.\n", encoding="utf-8")

    doc = parser.parse(md_file)

    assert doc.metadata.get("title") == "Только заголовок"
    assert "Текст без frontmatter" in doc.content
    assert doc.raw_metadata.get("frontmatter") == {}
