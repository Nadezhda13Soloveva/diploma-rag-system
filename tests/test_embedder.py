from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from backend.ingestion.embedder import EmbeddingError, OllamaEmbedder


def _mock_response(vector: list[float], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error",
            request=MagicMock(),
            response=resp,
        )
    resp.json.return_value = {"embedding": vector}
    return resp


def test_embed_query_returns_one_vector() -> None:
    dim = 8
    client = MagicMock()
    client.post.return_value = _mock_response([0.1] * dim)

    emb = OllamaEmbedder(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
        dimension=dim,
        use_cache=False,
        client=client,
    )
    vec = emb.embed_query("hello")

    assert len(vec) == dim
    assert vec == [0.1] * dim
    client.post.assert_called_once()
    args, kwargs = client.post.call_args
    assert args[0].endswith("/api/embeddings")
    assert kwargs["json"]["prompt"] == "hello"


def test_embed_documents_batch() -> None:
    dim = 4
    client = MagicMock()
    client.post.side_effect = [
        _mock_response([0.1] * dim),
        _mock_response([0.2] * dim),
        _mock_response([0.3] * dim),
    ]

    emb = OllamaEmbedder(
        dimension=dim,
        batch_size=2,
        use_cache=False,
        client=client,
    )
    vectors = emb.embed_documents(["a", "b", "c"])

    assert len(vectors) == 3
    assert vectors[0] == [0.1] * dim
    assert vectors[2] == [0.3] * dim
    assert client.post.call_count == 3


def test_dimension_validation() -> None:
    client = MagicMock()
    client.post.return_value = _mock_response([0.1, 0.2])  # dim=2

    emb = OllamaEmbedder(dimension=768, use_cache=False, client=client)

    with pytest.raises(EmbeddingError, match="dimension"):
        emb.embed_query("x")


def test_cache_avoids_second_call() -> None:
    dim = 3
    client = MagicMock()
    client.post.return_value = _mock_response([1.0] * dim)

    emb = OllamaEmbedder(dimension=dim, use_cache=True, client=client)
    v1 = emb.embed_query("same")
    v2 = emb.embed_query("same")

    assert v1 == v2
    assert client.post.call_count == 1


def test_timeout_error() -> None:
    client = MagicMock()
    client.post.side_effect = httpx.TimeoutException("timeout")

    emb = OllamaEmbedder(dimension=3, use_cache=False, client=client)

    with pytest.raises(EmbeddingError, match="timeout"):
        emb.embed_query("x")


def test_empty_list() -> None:
    emb = OllamaEmbedder(dimension=3, client=MagicMock())
    assert emb.embed_documents([]) == []
