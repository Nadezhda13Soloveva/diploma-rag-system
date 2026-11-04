from __future__ import annotations

import re
from typing import Any

import httpx

from backend.config import settings
from backend.generation.base import BaseGenerator, GenerationResult
from backend.generation.prompt_builder import build_prompt, list_sources
from backend.observability.logging import get_logger
from backend.stores.base import RetrievalResult

logger = get_logger(__name__)

_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:ответ|answer)\s*:\s*",
    re.IGNORECASE,
)


class GenerationError(RuntimeError):
    """Ошибка вызова LLM"""


def postprocess_answer(text: str) -> str:
    """Убирает обёртки вроде «Ответ:» в начале"""
    cleaned = (text or "").strip()
    cleaned = _ANSWER_PREFIX_RE.sub("", cleaned, count=1).strip()
    return cleaned


class OllamaGenerator(BaseGenerator):
    """Генерация ответа через Ollama /api/chat"""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model or settings.llm.model
        self.base_url = (base_url or settings.llm.base_url).rstrip("/")
        self.temperature = temperature if temperature is not None else settings.llm.temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.llm.max_tokens
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> OllamaGenerator:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def generate(
        self,
        query: str,
        retrieved: list[RetrievalResult],
        *,
        refused: bool = False,
        refusal_reason: str | None = None,
    ) -> GenerationResult:
        sources = list_sources(retrieved)

        if refused:
            answer = (
                "В загруженных документах недостаточно релевантной информации, "
                "чтобы ответить на этот вопрос."
            )
            logger.info(
                "generation_refused",
                reason=refusal_reason,
                query_len=len(query),
            )
            return GenerationResult(
                answer=answer,
                sources=sources,
                refused=True,
                refusal_reason=refusal_reason,
                model=self.model,
            )

        prompt = build_prompt(query, retrieved)
        raw = self._chat(prompt)
        answer = postprocess_answer(raw)

        logger.info(
            "generation_done",
            model=self.model,
            answer_len=len(answer),
            sources=len(sources),
        )
        return GenerationResult(
            answer=answer,
            sources=sources,
            refused=False,
            refusal_reason=None,
            model=self.model,
            raw_answer=raw,
        )

    def _chat(self, prompt: str) -> str:
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            resp = self._get_client().post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as exc:
            logger.error("generation_timeout", model=self.model, error=str(exc))
            raise GenerationError(f"Ollama chat timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            logger.error("generation_http_error", model=self.model, error=str(exc))
            raise GenerationError(f"Ollama chat HTTP error: {exc}") from exc

        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise GenerationError("Ollama chat response missing message.content")
        return content
