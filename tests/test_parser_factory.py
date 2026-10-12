from __future__ import annotations

from pathlib import Path

import pytest

from backend.ingestion.parsers.base import BaseParser, ParsedDocument
from backend.ingestion.parsers.factory import ParserFactory, UnknownFormatError


class DummyMarkdownParser(BaseParser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == ".md"

    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(
            document_id="dummy",
            source=str(path),
            filename=path.name,
            file_type="md",
            content="# dummy",
        )


class DummyTextParser(BaseParser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == ".txt"

    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(
            document_id="dummy",
            source=str(path),
            filename=path.name,
            file_type="txt",
            content="dummy text",
        )


def test_factory_selects_parser_by_extension() -> None:
    factory = ParserFactory(parsers=[DummyMarkdownParser(), DummyTextParser()])

    md_parser = factory.get_parser("docs/runbook.md")
    txt_parser = factory.get_parser(Path("notes.txt"))

    assert isinstance(md_parser, DummyMarkdownParser)
    assert isinstance(txt_parser, DummyTextParser)


def test_factory_raises_on_unknown_format() -> None:
    factory = ParserFactory(parsers=[DummyMarkdownParser()])

    with pytest.raises(UnknownFormatError) as exc_info:
        factory.get_parser("file.docx")

    assert "docx" in str(exc_info.value).lower() or "неизвестный" in str(exc_info.value).lower()


def test_factory_with_no_parsers_raises() -> None:
    factory = ParserFactory()

    with pytest.raises(UnknownFormatError):
        factory.get_parser("anything.md")


def test_register_parser() -> None:
    factory = ParserFactory()
    factory.register(DummyMarkdownParser())

    parser = factory.get_parser("readme.md")
    assert isinstance(parser, DummyMarkdownParser)
