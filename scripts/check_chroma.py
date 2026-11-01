"""Проверка работоспособности persistent Chroma"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings
from backend.ingestion.chunker.base import DocumentChunk
from backend.stores.chroma_store import ChromaVectorStore


def main() -> None:
    print("Chroma persistent check")
    print(f"persist_directory : {settings.vector_store.persist_directory}")
    print(f"collection_name : {settings.vector_store.collection_name}")

    store = ChromaVectorStore(
        persist_directory=settings.vector_store.persist_directory,
        collection_name=settings.vector_store.collection_name,
    )

    store.delete_collection()
    assert store.count() == 0, "Коллекция должна быть пустой после delete_collection"
    print("[OK] delete_collection / count=0")

    dim = settings.embedding.dimension
    chunks = [
        DocumentChunk(
            content="Локальный RAG на Ollama и Chroma",
            chunk_id="chunk-1",
            document_id="doc-1",
            source="test1.md",
            filename="test1.md",
            file_type="md",
            chunk_index=0,
            total_chunks=2,
        ),
        DocumentChunk(
            content="PostgreSQL с расширением pgvector",
            chunk_id="chunk-2",
            document_id="doc-1",
            source="test2.md",
            filename="test2.md",
            file_type="md",
            chunk_index=1,
            total_chunks=2,
        ),
    ]
    embeddings = [
        [0.1] * dim,
        [0.9] * dim,
    ]

    store.add_documents(chunks, embeddings)
    assert store.count() == 2, "Ожидалось 2 документа"
    print("[OK] add_documents / count=2")

    results = store.similarity_search([0.1] * dim, top_k=2)
    assert len(results) > 0, "Поиск должен вернуть результаты"
    top = results[0]
    print(f"[OK] similarity_search → {len(results)} results")
    print(f"     top id: {top.chunk_id}, score: {top.score:.4f}, rank: {top.rank}")

    store2 = ChromaVectorStore(
        persist_directory=settings.vector_store.persist_directory,
        collection_name=settings.vector_store.collection_name,
    )
    assert store2.count() == 2, "Данные должны сохраниться между перезапусками"
    print("[OK] persist between restarts")

    print("\nВсе проверки Chroma пройдены успешно")


if __name__ == "__main__":
    main()
