from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from backend.config import settings
from backend.ingestion.chunker.base import DocumentChunk
from backend.stores.base import BaseVectorStore, RetrievalResult


def _distance_to_score(distance: float | None) -> float:
    """Chroma cosine space: distance = 1 - cosine_similarity -> score = 1 - distance"""
    if distance is None:
        return 0.0
    return max(0.0, 1.0 - float(distance))


def _chunk_metadata(chunk: DocumentChunk) -> dict[str, Any]:
    """Chroma принимает плоские metadata (str/int/float/bool)"""
    meta: dict[str, Any] = {
        "document_id": chunk.document_id or "",
        "source": chunk.source or "",
        "filename": chunk.filename or "",
        "file_type": chunk.file_type or "",
        "chunk_index": int(chunk.chunk_index),
        "total_chunks": int(chunk.total_chunks),
    }
    if chunk.page is not None:
        meta["page"] = int(chunk.page)
    if chunk.title:
        meta["title"] = str(chunk.title)
    if chunk.section:
        meta["section"] = str(chunk.section)
    # дополнительные простые поля
    for k, v in chunk.metadata.items():
        if isinstance(v, str | int | float | bool):
            meta[f"extra_{k}"] = v
    return meta


class ChromaVectorStore(BaseVectorStore):
    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.persist_directory = persist_directory or settings.vector_store.persist_directory
        self.collection_name = collection_name or settings.vector_store.collection_name

        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection: Collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [_chunk_metadata(c) for c in chunks]

        self._collection.upsert(
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
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            return []

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        items: list[RetrievalResult] = []
        if not result["ids"] or not result["ids"][0]:
            return items

        for i, doc_id in enumerate(result["ids"][0]):
            meta = (result["metadatas"][0][i] if result["metadatas"] else {}) or {}
            dist = result["distances"][0][i] if result["distances"] else None
            content = result["documents"][0][i] if result["documents"] else ""

            items.append(
                RetrievalResult(
                    content=content or "",
                    score=_distance_to_score(dist),
                    rank=i + 1,
                    chunk_id=str(doc_id),
                    document_id=str(meta.get("document_id", "")),
                    source=str(meta.get("source", "")),
                    filename=str(meta.get("filename", "")),
                    file_type=str(meta.get("file_type", "")),
                    page=int(meta["page"]) if meta.get("page") is not None else None,
                    title=str(meta["title"]) if meta.get("title") is not None else None,
                    section=str(meta["section"]) if meta.get("section") is not None else None,
                    distance=float(dist) if dist is not None else None,
                    metadata=dict(meta),
                )
            )
        return items

    def delete_collection(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self._collection.count())
