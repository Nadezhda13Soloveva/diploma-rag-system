from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from backend.generation.ollama_generator import (
    GenerationError,
    OllamaGenerator,
    postprocess_answer,
)
from backend.stores.base import RetrievalResult


def _chunk(text: str = "Перезапуск PostgreSQL выполняется через systemctl.") -> RetrievalResult:
    return RetrievalResult(
        content=text,
        score=0.9,
        rank=1,
        chunk_id="c1",
        filename="db.md",
        page=2,
        source="docs/db.md",
    )


def _mock_chat_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "message": {"role": "assistant", "content": content},
    }
    return resp


def test_postprocess_strips_answer_prefix() -> None:
    assert postprocess_answer("Ответ: нужно перезапустить сервис") == ("нужно перезапустить сервис")
    assert postprocess_answer("Answer: restart the service") == "restart the service"
    assert postprocess_answer("просто текст") == "просто текст"


def test_generate_with_mock_llm() -> None:
    client = MagicMock()
    client.post.return_value = _mock_chat_response(
        "Ответ: Перезапустите PostgreSQL через systemctl."
    )

    gen = OllamaGenerator(
        model="qwen2.5:7b-instruct-q4_K_M",
        base_url="http://localhost:11434",
        temperature=0.1,
        max_tokens=256,
        client=client,
    )
    result = gen.generate("Как перезапустить postgres?", [_chunk()])

    assert result.refused is False
    assert "systemctl" in result.answer.lower() or "PostgreSQL" in result.answer
    assert not result.answer.lower().startswith("ответ:")
    assert len(result.sources) >= 1
    assert result.sources[0]["filename"] == "db.md"
    assert result.model == "qwen2.5:7b-instruct-q4_K_M"

    args, kwargs = client.post.call_args
    assert args[0].endswith("/api/chat")
    body: dict[str, Any] = kwargs["json"]
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0.1
    assert body["options"]["num_predict"] == 256
    assert "messages" in body


def test_generate_refusal_without_llm_call() -> None:
    client = MagicMock()
    gen = OllamaGenerator(client=client)
    result = gen.generate(
        "Случайный вопрос",
        [],
        refused=True,
        refusal_reason="best_score_below_refusal_threshold",
    )
    assert result.refused is True
    assert result.refusal_reason == "best_score_below_refusal_threshold"
    assert "недостаточно" in result.answer.lower()
    client.post.assert_not_called()


def test_generate_http_error() -> None:
    client = MagicMock()
    client.post.side_effect = httpx.TimeoutException("timeout")
    gen = OllamaGenerator(client=client)

    with pytest.raises(GenerationError, match="timeout"):
        gen.generate("вопрос", [_chunk()])


@pytest.mark.integration
def test_generate_real_ollama() -> None:
    # тут нужен запущенный Ollama с моделью из config
    try:
        with OllamaGenerator(timeout=180.0) as gen:
            result = gen.generate(
                "Что сказано в контексте про PostgreSQL?",
                [_chunk()],
            )
    except GenerationError as exc:
        pytest.skip(f"Ollama unavailable: {exc}")

    assert result.answer
    assert result.refused is False
    assert result.sources
