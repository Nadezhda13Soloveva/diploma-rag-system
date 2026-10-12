from __future__ import annotations

from pathlib import Path

import pytest

from backend.ingestion.parsers.text_parser import TextParser


@pytest.fixture
def parser() -> TextParser:
    return TextParser()


def test_can_parse(parser: TextParser) -> None:
    assert parser.can_parse(Path("notes.txt")) is True
    assert parser.can_parse(Path("file.TXT")) is True
    assert parser.can_parse(Path("file.md")) is False
    assert parser.can_parse(Path("file.pdf")) is False


def test_utf8_with_bom(parser: TextParser, tmp_path: Path) -> None:
    content = "Привет, мир!\r\nВторая строка.\r\n"
    data = b"\xef\xbb\xbf" + content.encode("utf-8")
    path = tmp_path / "utf8_bom.txt"
    path.write_bytes(data)

    doc = parser.parse(path)

    assert doc.file_type == "txt"
    assert doc.filename == "utf8_bom.txt"
    assert "Привет, мир!" in doc.content
    assert "\r" not in doc.content
    assert doc.content.count("\n") >= 1
    assert doc.metadata["had_bom"] is True
    assert doc.metadata["encoding"] in {"utf-8", "ascii"}


def test_utf8_without_bom(parser: TextParser, tmp_path: Path) -> None:
    path = tmp_path / "utf8.txt"
    path.write_text("Hello\r\nWorld\r\n", encoding="utf-8")

    doc = parser.parse(path)

    assert doc.content == "Hello\nWorld\n"
    assert doc.metadata["had_bom"] is False


def test_windows_1251(parser: TextParser, tmp_path: Path) -> None:
    text = "Тестовая строка в Windows-1251\r\nВторая строка"
    data = text.encode("cp1251")
    path = tmp_path / "cp1251.txt"
    path.write_bytes(data)

    doc = parser.parse(path)

    assert "Тестовая строка" in doc.content
    assert "Вторая строка" in doc.content
    assert "\r" not in doc.content
    assert doc.metadata["encoding"] in {"cp1251", "windows-1251"}


def test_newline_normalization(parser: TextParser, tmp_path: Path) -> None:
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"line1\r\nline2\rline3\n")

    doc = parser.parse(path)

    assert doc.content == "line1\nline2\nline3\n"
