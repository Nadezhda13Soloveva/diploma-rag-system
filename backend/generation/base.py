from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from backend.stores.base import RetrievalResult


@dataclass
class GenerationResult:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    model: str | None = None
    raw_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "model": self.model,
        }


class BaseGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        query: str,
        retrieved: list[RetrievalResult],
        *,
        refused: bool = False,
        refusal_reason: str | None = None,
    ) -> GenerationResult: ...
