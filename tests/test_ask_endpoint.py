from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.api.main import app
from backend.generation.base import GenerationResult
from backend.stores.base import RetrievalResult

client = TestClient(app)


def _retrieval_hits() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            content="PostgreSQL restart: systemctl restart postgresql",
            score=0.85,
            rank=1,
            chunk_id="c1",
            filename="db.md",
            page=1,
        )
    ]


@patch("backend.api.routes.ask.OllamaGenerator")
@patch("backend.api.routes.ask.HybridRetriever")
@patch("backend.api.routes.ask.ChromaVectorStore")
@patch("backend.api.routes.ask._load_bm25")
@patch("backend.api.routes.ask.OllamaEmbedder")
def test_ask_success(
    mock_embedder_cls: MagicMock,
    mock_bm25: MagicMock,
    mock_store_cls: MagicMock,
    mock_retriever_cls: MagicMock,
    mock_gen_cls: MagicMock,
) -> None:
    mock_embedder_cls.return_value.embed_query.return_value = [0.1] * 8

    store = mock_store_cls.return_value
    store.count.return_value = 1

    mock_bm25.return_value = MagicMock(__len__=lambda self: 1)

    mock_retriever_cls.return_value.retrieve.return_value = _retrieval_hits()

    mock_gen_cls.return_value.generate.return_value = GenerationResult(
        answer="Перезапустите через systemctl restart postgresql",
        sources=[{"filename": "db.md", "page": 1, "score": 0.85}],
        refused=False,
        model="test-model",
    )

    resp = client.post("/ask", json={"query": "Как перезапустить postgres?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["refused"] is False
    assert "systemctl" in data["answer"]
    assert data["request_id"]
    assert "timings" in data
    assert "embed_ms" in data["timings"]
    assert "retrieve_ms" in data["timings"]
    assert "generate_ms" in data["timings"]
    assert "total_ms" in data["timings"]
    assert data["sources"]
    assert data["scores"]


@patch("backend.api.routes.ask.OllamaGenerator")
@patch("backend.api.routes.ask.HybridRetriever")
@patch("backend.api.routes.ask.ChromaVectorStore")
@patch("backend.api.routes.ask._load_bm25")
@patch("backend.api.routes.ask.OllamaEmbedder")
def test_ask_empty_index(
    mock_embedder_cls: MagicMock,
    mock_bm25: MagicMock,
    mock_store_cls: MagicMock,
    mock_retriever_cls: MagicMock,
    mock_gen_cls: MagicMock,
) -> None:
    mock_embedder_cls.return_value.embed_query.return_value = [0.1] * 8
    mock_store_cls.return_value.count.return_value = 0
    mock_bm25.return_value = MagicMock(__len__=lambda self: 0)

    resp = client.post("/ask", json={"query": "Любой вопрос"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["refused"] is True
    assert data["refusal_reason"] == "empty_index"
    mock_gen_cls.return_value.generate.assert_not_called()


def test_ask_validation_empty_query() -> None:
    resp = client.post("/ask", json={"query": ""})
    assert resp.status_code == 422


@patch("backend.api.routes.ask.OllamaEmbedder")
def test_ask_ollama_embed_unavailable(mock_embedder_cls: MagicMock) -> None:
    from backend.ingestion.embedder import EmbeddingError

    mock_embedder_cls.return_value.embed_query.side_effect = EmbeddingError("down")
    resp = client.post("/ask", json={"query": "тест"})
    assert resp.status_code == 503
