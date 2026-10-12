from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParsedDocument:
    document_id: str
    source: str
    filename: str
    file_type: str  # md / pdf / txt
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    @abstractmethod
    def can_parse(self, path: Path) -> bool:
        """может ли этот парсер обработать данный файл"""
        ...

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """парсит файл и возвращает структурированный документ"""
        ...
