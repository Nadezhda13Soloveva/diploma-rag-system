from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from backend.api.schemas import AskRequest, AskResponse, TimingInfo
from backend.config import settings
from backend.generation.ollama_generator import GenerationError, OllamaGenerator
from backend.ingestion.embedder import EmbeddingError, OllamaEmbedder
from backend.observability.logging import get_logger
from backend.observability.timing import Timings, measure, set_stage
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.filtering import filter_by_score
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.stores.chroma_store import ChromaVectorStore

logger = get_logger(__name__)

router = APIRouter(tags=["ask"])


def _load_bm25() -> BM25Index:
    path = Path(settings.paths.index) / "bm25.pkl"
    if not path.exists():
        return BM25Index()
    try:
        return BM25Index.load(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("bm25_load_failed", path=str(path), error=str(exc))
        return BM25Index()


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    timings = Timings()
    query = body.query.strip()
    top_k = body.top_k or settings.retrieval.top_k

    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    # embed
    try:
        with measure() as t:
            embedder = OllamaEmbedder()
            query_embedding = embedder.embed_query(query)
        set_stage(timings, "embed", t[0])
    except EmbeddingError as exc:
        logger.exception("ask_embed_failed", request_id=request_id)
        raise HTTPException(
            status_code=503,
            detail=f"Embedding service unavailable: {exc}",
        ) from exc

    # retrieve
    try:
        with measure() as t:
            vector_store = ChromaVectorStore()
            bm25 = _load_bm25()

            if vector_store.count() == 0 and len(bm25) == 0:
                timings.mark_total()
                return AskResponse(
                    request_id=request_id,
                    answer=(
                        "Индекс пуст. Загрузите документы и выполните "
                        "POST /reindex перед запросами."
                    ),
                    refused=True,
                    refusal_reason="empty_index",
                    sources=[],
                    scores=[],
                    timings=TimingInfo(**timings.to_dict()),
                    model=settings.llm.model,
                )

            retriever = HybridRetriever(vector_store, bm25)
            retrieved = retriever.retrieve(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
            )
        set_stage(timings, "retrieve", t[0])
    except Exception as exc:  # noqa: BLE001
        logger.exception("ask_retrieve_failed", request_id=request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    # filter
    filtered, refusal_flag, refusal_reason = filter_by_score(retrieved)

    # generate
    try:
        with measure() as t:
            generator = OllamaGenerator()
            result = generator.generate(
                query,
                filtered,
                refused=refusal_flag,
                refusal_reason=refusal_reason,
            )
        set_stage(timings, "generate", t[0])
    except GenerationError as exc:
        logger.exception("ask_generate_failed", request_id=request_id)
        raise HTTPException(
            status_code=503,
            detail=f"LLM service unavailable: {exc}",
        ) from exc

    timings.mark_total()

    scores = [r.score for r in filtered]

    logger.info(
        "ask_complete",
        request_id=request_id,
        refused=result.refused,
        sources=len(result.sources),
        timings=timings.to_dict(),
    )

    return AskResponse(
        request_id=request_id,
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        sources=result.sources,
        scores=scores,
        timings=TimingInfo(**timings.to_dict()),
        model=result.model,
    )
