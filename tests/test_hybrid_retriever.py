from __future__ import annotations

from backend.ingestion.chunker.base import DocumentChunk
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.fusion import reciprocal_rank_fusion
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.stores.base import BaseVectorStore, RetrievalResult


def test_rrf_reference_example() -> None:
    """
    Эталон:
      list A: d1, d2, d3
      list B: d2, d3, d4
      k=60
      score(d1)=1/61
      score(d2)=1/61 + 1/61 = 2/61
      score(d3)=1/62 + 1/62 = 2/62
      score(d4)=1/63
    Порядок: d2 > d3 > d1 > d4
    """
    fused = reciprocal_rank_fusion(
        [["d1", "d2", "d3"], ["d2", "d3", "d4"]],
        k=60,
        weights=[1.0, 1.0],
    )
    ids = [doc_id for doc_id, _ in fused]
    assert ids == ["d2", "d3", "d1", "d4"]

    scores = dict(fused)
    assert abs(scores["d1"] - 1 / 61) < 1e-9
    assert abs(scores["d2"] - (1 / 62 + 1 / 61)) < 1e-9
    assert abs(scores["d3"] - (1 / 63 + 1 / 62)) < 1e-9
    assert abs(scores["d4"] - 1 / 63) < 1e-9


def test_rrf_with_weights() -> None:
    fused = reciprocal_rank_fusion(
        [["a", "b"], ["b", "a"]],
        k=60,
        weights=[1.0, 0.0],
    )
    assert fused[0][0] == "a"


class FakeVectorStore(BaseVectorStore):
    def __init__(self, hits: list[RetrievalResult]) -> None:
        self._hits = hits

    def add_documents(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        return None

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        where: dict[str, object] | None = None,
    ) -> list[RetrievalResult]:
        return self._hits[:top_k]

    def delete_collection(self) -> None:
        return None

    def count(self) -> int:
        return len(self._hits)


def test_hybrid_retriever_source_ranks() -> None:
    vector_hits = [
        RetrievalResult(content="alpha postgres", score=0.9, rank=1, chunk_id="c1"),
        RetrievalResult(content="beta redis", score=0.8, rank=2, chunk_id="c2"),
    ]
    bm25 = BM25Index()
    bm25.build(
        [
            DocumentChunk(content="postgres restart guide", chunk_id="c1"),
            DocumentChunk(content="redis cache flush", chunk_id="c2"),
            DocumentChunk(content="network dns check", chunk_id="c3"),
        ]
    )

    retriever = HybridRetriever(
        FakeVectorStore(vector_hits),
        bm25,
        vector_weight=0.6,
        bm25_weight=0.4,
        rrf_k=60,
    )
    results = retriever.retrieve("postgres", query_embedding=[0.1, 0.2], top_k=3)

    assert len(results) >= 1
    assert results[0].final_rank == 1
    assert results[0].rank == 1
    # c1 должен быть и в vector, и в bm25
    top = next(r for r in results if r.chunk_id == "c1")
    assert top.vector_rank == 1
    assert top.bm25_rank is not None
    assert "vector" in top.metadata.get("sources", [])
    assert "bm25" in top.metadata.get("sources", [])
