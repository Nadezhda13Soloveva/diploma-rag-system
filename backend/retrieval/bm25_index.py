from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.ingestion.chunker.base import DocumentChunk
from backend.observability.logging import get_logger

logger = get_logger(__name__)

# слова: латиница, кириллица, цифры; длина >= 2
_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]{2,}", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Простая токенизация RU/EN: lower + regex-слова"""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


@dataclass
class BM25Index:
    """Лексический индекс BM25Okapi с persist на диск"""

    chunk_ids: list[str] = field(default_factory=list)
    corpus_tokens: list[list[str]] = field(default_factory=list)
    _bm25: BM25Okapi | None = field(default=None, repr=False)

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.chunk_ids = []
        self.corpus_tokens = []
        for chunk in chunks:
            tokens = tokenize(chunk.content)
            if not tokens:
                # пустые чанки не индексируем, но сохраняем id с пустым токен-листом
                # чтобы длины совпадали- лучше пропускать
                continue
            self.chunk_ids.append(chunk.chunk_id)
            self.corpus_tokens.append(tokens)

        if not self.corpus_tokens:
            self._bm25 = None
            logger.warning("bm25_build_empty", message="Нет токенов для BM25")
            return

        self._bm25 = BM25Okapi(self.corpus_tokens)
        logger.info("bm25_built", documents=len(self.chunk_ids))

    def search(self, query: str, top_k: int = 8) -> list[tuple[str, float]]:
        if self._bm25 is None or not self.chunk_ids:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(
            zip(self.chunk_ids, scores, strict=True),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        top_k = max(0, top_k)
        # возвращаем top_k даже если score == 0 (редкий крайний случай)
        return [(cid, float(score)) for cid, score in ranked[:top_k]]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_ids": self.chunk_ids,
            "corpus_tokens": self.corpus_tokens,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("bm25_saved", path=str(path), documents=len(self.chunk_ids))

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)  # noqa: S301 - локальный индекс

        index = cls(
            chunk_ids=list(payload["chunk_ids"]),
            corpus_tokens=list(payload["corpus_tokens"]),
        )
        if index.corpus_tokens:
            index._bm25 = BM25Okapi(index.corpus_tokens)
        logger.info("bm25_loaded", path=str(path), documents=len(index.chunk_ids))
        return index

    def __len__(self) -> int:
        return len(self.chunk_ids)
