from __future__ import annotations

from pathlib import Path

import pytest

from backend.ingestion.chunker.base import DocumentChunk
from backend.stores.chroma_store import ChromaVectorStore


def _chunk(i: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        content=text,
        chunk_id=f"chunk-{i}",
        document_id="doc-1",
        source="docs/runbook.md",
        filename="runbook.md",
        file_type="md",
        chunk_index=i,
        total_chunks=2,
        page=1,
        title="Runbook",
        section="Diagnostics",
    )


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_docs",
    )


def test_add_and_search(store: ChromaVectorStore) -> None:
    chunks = [
        _chunk(0, "PostgreSQL recovery procedure"),
        _chunk(1, "Redis cache invalidation steps"),
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    store.add_documents(chunks, embeddings)
    assert store.count() == 2

    results = store.similarity_search([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[0].score >= results[1].score
    assert "PostgreSQL" in results[0].content
    assert results[0].filename == "runbook.md"
    assert results[0].document_id == "doc-1"
    assert results[0].title == "Runbook"
    assert results[0].section == "Diagnostics"
    assert results[0].metadata.get("source") == "docs/runbook.md"


def test_persist_between_restarts(tmp_path: Path) -> None:
    path = str(tmp_path / "chroma_persist")
    s1 = ChromaVectorStore(persist_directory=path, collection_name="persist_test")
    s1.add_documents(
        [_chunk(0, "persistent chunk")],
        [[0.2, 0.3, 0.4, 0.5]],
    )
    assert s1.count() == 1

    s2 = ChromaVectorStore(persist_directory=path, collection_name="persist_test")
    assert s2.count() == 1
    results = s2.similarity_search([0.2, 0.3, 0.4, 0.5], top_k=1)
    assert len(results) == 1
    assert "persistent" in results[0].content


def test_delete_collection(store: ChromaVectorStore) -> None:
    store.add_documents([_chunk(0, "x")], [[0.1, 0.1, 0.1, 0.1]])
    assert store.count() == 1
    store.delete_collection()
    assert store.count() == 0
