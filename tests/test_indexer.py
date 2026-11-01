from __future__ import annotations

from pathlib import Path

from backend.ingestion.embedder import BaseEmbedder
from backend.ingestion.indexer import Indexer
from backend.retrieval.bm25_index import BM25Index
from backend.stores.chroma_store import ChromaVectorStore
from backend.stores.metadata_store import MetadataStore


class FakeEmbedder(BaseEmbedder):
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float((i + 1) % 7) for _ in range(self.dim)] for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _write_docs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.md").write_text(
        "# Postgres\n\nRestart PostgreSQL service carefully\n",
        encoding="utf-8",
    )
    (root / "b.txt").write_text(
        "Redis cache flush procedure and invalidation steps\n",
        encoding="utf-8",
    )
    (root / "c.md").write_text(
        "# Network\n\nCheck DNS and firewall rules for the cluster\n",
        encoding="utf-8",
    )


def test_indexer_run_and_idempotent(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    _write_docs(docs)

    chroma_dir = tmp_path / "chroma"
    index_dir = tmp_path / "index"
    bm25_path = index_dir / "bm25.pkl"
    meta_path = index_dir / "metadata.json"

    embedder = FakeEmbedder(dim=8)
    store = ChromaVectorStore(
        persist_directory=str(chroma_dir),
        collection_name="test_index",
    )

    indexer = Indexer(
        documents_dir=docs,
        embedder=embedder,
        vector_store=store,
        bm25_path=bm25_path,
        metadata_path=meta_path,
        batch_size=2,
    )

    stats1 = indexer.run()
    assert stats1["documents_indexed"] == 3
    assert stats1["chunks_indexed"] >= 3
    assert stats1["errors"] == 0
    assert store.count() >= 3
    assert meta_path.exists()
    assert bm25_path.exists()

    meta = MetadataStore(meta_path)
    assert len(meta.list_documents()) == 3

    bm25 = BM25Index.load(bm25_path)
    assert len(bm25) >= 3
    hits = bm25.search("PostgreSQL", top_k=3)
    assert len(hits) >= 1

    # повторный запуск - без дублей
    indexer2 = Indexer(
        documents_dir=docs,
        embedder=FakeEmbedder(dim=8),
        vector_store=store,
        bm25_path=bm25_path,
        metadata_path=meta_path,
    )
    stats2 = indexer2.run()
    assert stats2["documents_indexed"] == 0
    assert stats2["skipped"] == 3
    assert len(MetadataStore(meta_path).list_documents()) == 3
