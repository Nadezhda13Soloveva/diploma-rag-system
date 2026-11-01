from __future__ import annotations

from pathlib import Path

from backend.ingestion.chunker.base import DocumentChunk
from backend.retrieval.bm25_index import BM25Index, tokenize


def _chunk(cid: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        content=text,
        chunk_id=cid,
        document_id="d1",
        source="s.md",
        filename="s.md",
        file_type="md",
        chunk_index=0,
        total_chunks=1,
    )


def test_tokenize_ru_en() -> None:
    tokens = tokenize("Hello, PostgreSQL! Перезапуск сервиса 42.")
    assert "hello" in tokens
    assert "postgresql" in tokens
    assert "перезапуск" in tokens
    assert "сервиса" in tokens
    assert "42" in tokens


def test_build_and_search() -> None:
    chunks = [
        _chunk("c1", "PostgreSQL recovery and restart procedure"),
        _chunk("c2", "Redis cache flush and invalidation"),
        _chunk("c3", "Kubernetes pod restart and rollout"),
    ]
    index = BM25Index()
    index.build(chunks)

    assert len(index) == 3
    results = index.search("PostgreSQL restart", top_k=2)
    assert len(results) >= 1
    assert results[0][0] == "c1"
    assert results[0][1] > 0


def test_search_russian() -> None:
    chunks = [
        _chunk("r1", "Инструкция по перезапуску PostgreSQL"),
        _chunk("r2", "Очистка кэша Redis"),
    ]
    index = BM25Index()
    index.build(chunks)
    # те же словоформы, что в документе (без стемминга)
    results = index.search("перезапуску PostgreSQL", top_k=5)
    assert any(cid == "r1" for cid, _ in results)
    assert results[0][0] == "r1"


def test_save_and_load(tmp_path: Path) -> None:
    chunks = [
        _chunk("a", "alpha document about databases"),
        _chunk("b", "beta document about networking"),
    ]
    index = BM25Index()
    index.build(chunks)

    path = tmp_path / "bm25.pkl"
    index.save(path)
    assert path.exists()

    loaded = BM25Index.load(path)
    assert len(loaded) == 2
    results = loaded.search("databases", top_k=1)
    assert len(results) == 1
    assert results[0][0] == "a"
    assert results[0][1] >= 0


def test_empty_query_or_index() -> None:
    index = BM25Index()
    assert index.search("anything") == []

    index.build([_chunk("x", "some text here")])
    assert index.search("") == []
    assert index.search("   ") == []
