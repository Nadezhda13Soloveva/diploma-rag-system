from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any

import httpx

from backend.config import settings
from backend.observability.logging import get_logger

logger = get_logger(__name__)


class EmbeddingError(RuntimeError):
    """Ошибка при получении эмбеддингов"""


class BaseEmbedder(ABC):
    """Единый интерфейс эмбеддеров"""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Векторизация списка документов/чанков"""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Векторизация одного запроса"""
        ...


class OllamaEmbedder(BaseEmbedder):
    """
    Embedder через Ollama /api/embeddings
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        dimension: int | None = None,
        timeout: float = 60.0,
        batch_size: int = 32,
        use_cache: bool = True,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model or settings.embedding.model
        self.base_url = (base_url or settings.embedding.base_url).rstrip("/")
        self.dimension = dimension or settings.embedding.dimension
        self.timeout = timeout
        self.batch_size = batch_size
        self.use_cache = use_cache
        self._cache: dict[str, list[float]] = {}
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> OllamaEmbedder:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        to_compute: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            if self.use_cache:
                key = self._cache_key(text)
                cached = self._cache.get(key)
                if cached is not None:
                    results[i] = cached
                    continue
            to_compute.append((i, text))

        for start in range(0, len(to_compute), self.batch_size):
            batch = to_compute[start : start + self.batch_size]
            for idx, text in batch:
                vector = self._embed_one(text)
                self._validate_dimension(vector)
                results[idx] = vector
                if self.use_cache:
                    self._cache[self._cache_key(text)] = vector

        return [v if v is not None else [] for v in results]

    def _embed_one(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embeddings"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": text,
        }
        try:
            resp = self._get_client().post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as exc:
            logger.error("embed_timeout", model=self.model, error=str(exc))
            raise EmbeddingError(f"Ollama embeddings timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            logger.error("embed_http_error", model=self.model, error=str(exc))
            raise EmbeddingError(f"Ollama embeddings HTTP error: {exc}") from exc
        except ValueError as exc:
            raise EmbeddingError(f"Invalid JSON from Ollama: {exc}") from exc

        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingError("Ollama response missing 'embedding'")
        return [float(x) for x in embedding]

    def _validate_dimension(self, vector: list[float]) -> None:
        if len(vector) != self.dimension:
            raise EmbeddingError(
                f"Unexpected embedding dimension: got {len(vector)}, expected {self.dimension}"
            )

    def _cache_key(self, text: str) -> str:
        raw = f"{self.model}:{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def clear_cache(self) -> None:
        self._cache.clear()
