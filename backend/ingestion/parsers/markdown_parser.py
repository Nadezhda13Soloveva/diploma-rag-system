from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from backend.ingestion.parsers.base import BaseParser, ParsedDocument

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class MarkdownParser(BaseParser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in {".md", ".markdown"}

    def parse(self, path: Path) -> ParsedDocument:
        path = Path(path)
        raw_text = path.read_text(encoding="utf-8")

        frontmatter, body = self._split_frontmatter(raw_text)
        title = self._extract_title(body, frontmatter)
        sections = self._extract_sections(body)

        document_id = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]

        metadata: dict[str, Any] = {
            "title": title,
            "sections": sections,
        }
        if frontmatter:
            metadata["frontmatter"] = frontmatter

        if title:
            metadata["title"] = title
        if sections:
            metadata["section"] = sections[0]

        return ParsedDocument(
            document_id=document_id,
            source=str(path),
            filename=path.name,
            file_type="md",
            content=body.strip() + "\n",
            metadata=metadata,
            raw_metadata={
                "frontmatter": frontmatter,
                "encoding": "utf-8",
            },
        )

    def _split_frontmatter(self, text: str) -> tuple[dict[str, Any], str]:
        match = FRONTMATTER_RE.match(text)
        if not match:
            return {}, text

        raw_fm = match.group(1)
        body = text[match.end() :]
        try:
            data = yaml.safe_load(raw_fm) or {}
            if not isinstance(data, dict):
                data = {"_raw": data}
        except yaml.YAMLError:
            data = {"_raw": raw_fm}

        return data, body

    def _extract_title(self, body: str, frontmatter: dict[str, Any]) -> str | None:
        # 1. title из frontmatter имеет приоритет
        fm_title = frontmatter.get("title")
        if isinstance(fm_title, str) and fm_title.strip():
            return fm_title.strip()

        # 2. первый заголовок уровня # (H1)
        for match in HEADING_RE.finditer(body):
            level = len(match.group(1))
            text = match.group(2).strip()
            if level == 1 and text:
                return text

        return None

    def _extract_sections(self, body: str) -> list[str]:
        sections: list[str] = []
        for match in HEADING_RE.finditer(body):
            text = match.group(2).strip()
            if text:
                sections.append(text)
        return sections
