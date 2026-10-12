from __future__ import annotations

import hashlib
from pathlib import Path

import chardet

from backend.ingestion.parsers.base import BaseParser, ParsedDocument

# типичные BOM
_BOM_UTF8 = b"\xef\xbb\xbf"
_BOM_UTF16_LE = b"\xff\xfe"
_BOM_UTF16_BE = b"\xfe\xff"


class TextParser(BaseParser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == ".txt"

    def parse(self, path: Path) -> ParsedDocument:
        path = Path(path)
        raw = path.read_bytes()

        raw, had_bom = self._strip_bom(raw)
        encoding = self._detect_encoding(raw)
        text = raw.decode(encoding, errors="replace")
        text = self._normalize_newlines(text)

        document_id = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]

        return ParsedDocument(
            document_id=document_id,
            source=str(path),
            filename=path.name,
            file_type="txt",
            content=text,
            metadata={
                "encoding": encoding,
                "had_bom": had_bom,
            },
            raw_metadata={
                "encoding": encoding,
                "had_bom": had_bom,
                "size_bytes": path.stat().st_size,
            },
        )

    def _strip_bom(self, data: bytes) -> tuple[bytes, bool]:
        if data.startswith(_BOM_UTF8):
            return data[len(_BOM_UTF8) :], True
        if data.startswith(_BOM_UTF16_LE) or data.startswith(_BOM_UTF16_BE):
            return data[2:], True
        return data, False

    def _detect_encoding(self, data: bytes) -> str:
        if not data:
            return "utf-8"

        result = chardet.detect(data)
        encoding = (result.get("encoding") or "utf-8").lower()

        aliases = {
            "windows-1251": "cp1251",
            "win1251": "cp1251",
            "cp1251": "cp1251",
            "utf8": "utf-8",
            "utf-8": "utf-8",
            "ascii": "utf-8",  # ascii читаем как utf-8
        }
        return aliases.get(encoding, encoding)

    def _normalize_newlines(self, text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")
