from __future__ import annotations

from backend.config import settings
from backend.observability.logging import get_logger
from backend.stores.base import RetrievalResult

logger = get_logger(__name__)


def filter_by_score(
    results: list[RetrievalResult],
    min_score: float | None = None,
    refusal_threshold: float | None = None,
    min_relevant_chunks: int | None = None,
) -> tuple[list[RetrievalResult], bool, str | None]:
    """
    Фильтрация retrieval-результатов и решение об отказе
        filtered_results: чанки с score >= min_score
        refusal_flag: True - генерацию запускать не нужно
        refusal_reason: причина отказа или None
    """
    min_score = settings.retrieval.min_score if min_score is None else min_score
    refusal_threshold = (
        settings.retrieval.refusal_threshold if refusal_threshold is None else refusal_threshold
    )
    min_relevant_chunks = (
        settings.retrieval.min_relevant_chunks
        if min_relevant_chunks is None
        else min_relevant_chunks
    )

    if not results:
        reason = "no_results"
        logger.info("retrieval_refusal", reason=reason)
        return [], True, reason

    best_score = max(r.score for r in results)

    if best_score < refusal_threshold:
        reason = "best_score_below_refusal_threshold"
        logger.info(
            "retrieval_refusal",
            reason=reason,
            best_score=best_score,
            refusal_threshold=refusal_threshold,
        )
        return [], True, reason

    filtered = [r for r in results if r.score >= min_score]

    if len(filtered) < min_relevant_chunks:
        reason = "insufficient_relevant_chunks"
        logger.info(
            "retrieval_refusal",
            reason=reason,
            relevant=len(filtered),
            min_relevant_chunks=min_relevant_chunks,
            min_score=min_score,
        )
        return filtered, True, reason

    logger.info(
        "retrieval_accepted",
        best_score=best_score,
        relevant=len(filtered),
    )
    return filtered, False, None
