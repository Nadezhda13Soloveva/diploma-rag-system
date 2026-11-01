from __future__ import annotations

from abc import ABC, abstractmethod

from backend.stores.base import RetrievalResult


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 8,
    ) -> list[RetrievalResult]: ...
