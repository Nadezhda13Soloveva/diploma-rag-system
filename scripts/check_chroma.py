#!/usr/bin/env python3
"""Проверка работоспособности persistent Chroma"""

from __future__ import annotations

import sys
from pathlib import Path

# чтобы можно было запускать скрипт из корня проекта
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings
from backend.stores.chroma_store import ChromaVectorStore


def main() -> None:
    print("Chroma persistent check")
    print(f"persist_directory: {settings.vector_store.persist_directory}")
    print(f"collection_name: {settings.vector_store.collection_name}")

    store = ChromaVectorStore(
        persist_directory=settings.vector_store.persist_directory,
        collection_name=settings.vector_store.collection_name,
    )

    # 1. очистка (чтобы проверка была повторяемой)
    store.reset()
    assert store.count() == 0, "Коллекция должна быть пустой после reset"
    print("[OK] reset / count=0")

    # 2. вставка тестовых данных
    dim = settings.embedding.dimension
    ids = ["chunk-1", "chunk-2"]
    documents = [
        "Локальный RAG на Ollama и Chroma",
        "PostgreSQL с расширением pgvector",
    ]
    embeddings = [
        [0.1] * dim,
        [0.9] * dim,
    ]
    metadatas = [
        {"source": "test1.md", "chunk_index": 0},
        {"source": "test2.md", "chunk_index": 1},
    ]

    store.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    assert store.count() == 2, "Ожидалось 2 документа"
    print("[OK] add / count=2")

    # 3. поиск
    query_embedding = [0.1] * dim
    results = store.similarity_search(query_embedding, top_k=2)
    assert len(results) > 0, "Поиск должен вернуть результаты"
    print(f"[OK] similarity_search -> {len(results)} results")
    print(f" top id: {results[0]['id']}, distance: {results[0]['distance']:.4f}")

    # 4. проверка persist: создаём новый экземпляр того же пути
    store2 = ChromaVectorStore(
        persist_directory=settings.vector_store.persist_directory,
        collection_name=settings.vector_store.collection_name,
    )
    assert store2.count() == 2, "Данные должны сохраниться между перезапусками"
    print("[OK] persist between restarts")

    print("\nВсе проверки Chroma пройдены")


if __name__ == "__main__":
    main()
