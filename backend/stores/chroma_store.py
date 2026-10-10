from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection


class BaseVectorStore(ABC):
    @abstractmethod
    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None: ...

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def count(self) -> int: ...


class ChromaVectorStore(BaseVectorStore):
    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "documents",
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        items: list[dict[str, Any]] = []
        if not result["ids"] or not result["ids"][0]:
            return items

        for i, doc_id in enumerate(result["ids"][0]):
            items.append(
                {
                    "id": doc_id,
                    "content": (result["documents"][0][i] if result["documents"] else None),
                    "metadata": (result["metadatas"][0][i] if result["metadatas"] else {}),
                    "distance": (result["distances"][0][i] if result["distances"] else None),
                }
            )
        return items

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        """удаляет коллекцию и создаёт заново (для тестов)"""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
