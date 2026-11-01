from __future__ import annotations

from backend.stores.base import RetrievalResult

SYSTEM_INSTRUCTIONS = (
    "Ты — помощник по локальной базе знаний.\n"
    "Отвечай только на основе приведённого контекста из загруженных документов.\n"
    "Если в контексте недостаточно данных для ответа — честно скажи, "
    "что в предоставленных материалах этого нет.\n"
    "Не выдумывай факты и не опирайся на знания вне контекста.\n"
    "Отвечай на языке вопроса пользователя.\n"
    "При возможности кратко укажи источники, на которые опираешься."
)

PROMPT_TEMPLATE = """{system}

Контекст:
{context}

Вопрос: {query}

Ответ:"""


def _format_source(chunk: RetrievalResult) -> str:
    filename = chunk.filename or chunk.source or "unknown"
    if chunk.page is not None:
        return f"[Источник: {filename}, стр. {chunk.page}]"
    return f"[Источник: {filename}]"


def _format_chunk(chunk: RetrievalResult) -> str:
    header = _format_source(chunk)
    body = (chunk.content or "").strip()
    return f"{header}\n{body}"


def build_prompt(query: str, chunks: list[RetrievalResult]) -> str:
    ordered = sorted(chunks, key=lambda c: c.score, reverse=True)
    if ordered:
        context = "\n\n".join(_format_chunk(c) for c in ordered)
    else:
        context = "(контекст пуст)"

    return PROMPT_TEMPLATE.format(
        system=SYSTEM_INSTRUCTIONS,
        context=context,
        query=query.strip(),
    )


def list_sources(
    chunks: list[RetrievalResult],
) -> list[dict[str, str | int | float | None]]:
    ordered = sorted(chunks, key=lambda c: c.score, reverse=True)
    sources: list[dict[str, str | int | float | None]] = []
    seen: set[str] = set()
    for c in ordered:
        key = f"{c.filename}|{c.page}|{c.chunk_id}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "filename": c.filename or c.source,
                "page": c.page,
                "chunk_id": c.chunk_id,
                "score": c.score,
            }
        )
    return sources
