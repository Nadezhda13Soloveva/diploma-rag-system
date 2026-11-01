from __future__ import annotations

from backend.retrieval.filtering import filter_by_score
from backend.stores.base import RetrievalResult


def _r(score: float, cid: str = "c") -> RetrievalResult:
    return RetrievalResult(content=f"text-{cid}", score=score, rank=1, chunk_id=cid)


def test_success() -> None:
    results = [_r(0.9, "a"), _r(0.5, "b"), _r(0.1, "c")]
    filtered, refusal, reason = filter_by_score(
        results,
        min_score=0.32,
        refusal_threshold=0.25,
        min_relevant_chunks=1,
    )
    assert refusal is False
    assert reason is None
    assert len(filtered) == 2
    assert all(r.score >= 0.32 for r in filtered)


def test_refusal_best_score_too_low() -> None:
    results = [_r(0.10, "a"), _r(0.05, "b")]
    filtered, refusal, reason = filter_by_score(
        results,
        min_score=0.32,
        refusal_threshold=0.25,
        min_relevant_chunks=1,
    )
    assert refusal is True
    assert reason == "best_score_below_refusal_threshold"
    assert filtered == []


def test_refusal_insufficient_relevant() -> None:
    # best_score ок, но только 1 чанк выше min_score, а нужно 2
    results = [_r(0.80, "a"), _r(0.10, "b"), _r(0.05, "c")]
    filtered, refusal, reason = filter_by_score(
        results,
        min_score=0.32,
        refusal_threshold=0.25,
        min_relevant_chunks=2,
    )
    assert refusal is True
    assert reason == "insufficient_relevant_chunks"
    assert len(filtered) == 1
    assert filtered[0].chunk_id == "a"


def test_borderline_exact_thresholds() -> None:
    # best_score == refusal_threshold - не отказываем по этому правилу
    # score == min_score - проходит фильтр
    results = [_r(0.25, "a"), _r(0.32, "b")]
    filtered, refusal, reason = filter_by_score(
        results,
        min_score=0.32,
        refusal_threshold=0.25,
        min_relevant_chunks=1,
    )
    assert refusal is False
    assert reason is None
    assert len(filtered) == 1
    assert filtered[0].chunk_id == "b"


def test_empty_results() -> None:
    filtered, refusal, reason = filter_by_score(
        [],
        min_score=0.32,
        refusal_threshold=0.25,
        min_relevant_chunks=1,
    )
    assert refusal is True
    assert reason == "no_results"
    assert filtered == []
