from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """
    RRF
    score(d) = Σ weight_i / (k + rank_i(d))
    rank = 1-based позиция в списке
    Возвращает [(id, score)] по убыванию score
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists")

    scores: dict[str, float] = defaultdict(float)

    for weight, ranking in zip(weights, ranked_lists, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += weight * (1.0 / (k + rank))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
