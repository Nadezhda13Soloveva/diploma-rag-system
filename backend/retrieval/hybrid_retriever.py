from __future__ import annotations

from backend.config import settings
from backend.observability.logging import get_logger
from backend.retrieval.base import BaseRetriever
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.fusion import reciprocal_rank_fusion
from backend.stores.base import BaseVectorStore, RetrievalResult

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Vector + BM25 + RRF fusion
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        bm25_index: BM25Index,
        *,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
        rrf_k: int = 60,
        fetch_k: int | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.vector_weight = (
            vector_weight if vector_weight is not None else settings.retrieval.vector_weight
        )
        self.bm25_weight = (
            bm25_weight if bm25_weight is not None else settings.retrieval.bm25_weight
        )
        self.rrf_k = rrf_k
        # сколько кандидатов брать у каждого ретривера до fusion
        self.fetch_k = fetch_k

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 8,
    ) -> list[RetrievalResult]:
        fetch_k = self.fetch_k or max(top_k * 3, top_k)

        vector_hits = self.vector_store.similarity_search(query_embedding, top_k=fetch_k)
        bm25_hits = self.bm25_index.search(query, top_k=fetch_k)

        vector_ids = [h.chunk_id for h in vector_hits]
        bm25_ids = [cid for cid, _ in bm25_hits]

        fused = reciprocal_rank_fusion(
            [vector_ids, bm25_ids],
            k=self.rrf_k,
            weights=[self.vector_weight, self.bm25_weight],
        )

        vector_rank_map = {h.chunk_id: h.rank for h in vector_hits}
        bm25_rank_map = {cid: i + 1 for i, (cid, _) in enumerate(bm25_hits)}
        vector_by_id = {h.chunk_id: h for h in vector_hits}

        # контент/metadata: предпочитаем vector hit; для BM25-only — заглушка
        results: list[RetrievalResult] = []
        for final_rank, (chunk_id, rrf_score) in enumerate(fused[:top_k], start=1):
            base = vector_by_id.get(chunk_id)
            if base is not None:
                item = RetrievalResult(
                    content=base.content,
                    score=float(rrf_score),
                    rank=final_rank,
                    chunk_id=base.chunk_id,
                    document_id=base.document_id,
                    source=base.source,
                    filename=base.filename,
                    file_type=base.file_type,
                    page=base.page,
                    title=base.title,
                    section=base.section,
                    distance=base.distance,
                    vector_rank=vector_rank_map.get(chunk_id),
                    bm25_rank=bm25_rank_map.get(chunk_id),
                    final_rank=final_rank,
                    metadata={
                        **base.metadata,
                        "rrf_score": float(rrf_score),
                        "sources": _sources(vector_rank_map, bm25_rank_map, chunk_id),
                    },
                )
            else:
                item = RetrievalResult(
                    content="",  # BM25-only: контент можно добрать из metadata store позже
                    score=float(rrf_score),
                    rank=final_rank,
                    chunk_id=chunk_id,
                    vector_rank=vector_rank_map.get(chunk_id),
                    bm25_rank=bm25_rank_map.get(chunk_id),
                    final_rank=final_rank,
                    metadata={
                        "rrf_score": float(rrf_score),
                        "sources": _sources(vector_rank_map, bm25_rank_map, chunk_id),
                        "content_missing": True,
                    },
                )
            results.append(item)

        logger.info(
            "hybrid_retrieve",
            query_len=len(query),
            vector_hits=len(vector_hits),
            bm25_hits=len(bm25_hits),
            fused=len(results),
        )
        return results


def _sources(
    vector_rank_map: dict[str, int],
    bm25_rank_map: dict[str, int],
    chunk_id: str,
) -> list[str]:
    out: list[str] = []
    if chunk_id in vector_rank_map:
        out.append("vector")
    if chunk_id in bm25_rank_map:
        out.append("bm25")
    return out
