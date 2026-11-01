from __future__ import annotations

from backend.generation.prompt_builder import build_prompt, list_sources
from backend.stores.base import RetrievalResult


def _chunk(
    content: str,
    score: float,
    filename: str = "runbook.md",
    page: int | None = 3,
    chunk_id: str = "c1",
) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        score=score,
        rank=1,
        chunk_id=chunk_id,
        filename=filename,
        page=page,
        source=f"docs/{filename}",
    )


def test_build_prompt_structure() -> None:
    chunks = [
        _chunk("Сначала проверь логи.", 0.4, filename="b.md", page=2, chunk_id="b"),
        _chunk("Перезапусти PostgreSQL.", 0.9, filename="a.md", page=1, chunk_id="a"),
    ]
    prompt = build_prompt("Как перезапустить postgres?", chunks)

    assert "Контекст:" in prompt
    assert "Вопрос: Как перезапустить postgres?" in prompt
    assert "Ответ:" in prompt
    assert "Перезапусти PostgreSQL." in prompt
    assert "Сначала проверь логи." in prompt

    # сортировка по score: a раньше b
    pos_a = prompt.index("Перезапусти PostgreSQL.")
    pos_b = prompt.index("Сначала проверь логи.")
    assert pos_a < pos_b


def test_sources_format() -> None:
    chunks = [_chunk("Текст чанка", 0.8, filename="db.md", page=5)]
    prompt = build_prompt("вопрос", chunks)
    assert "[Источник: db.md, стр. 5]" in prompt


def test_source_without_page() -> None:
    chunks = [_chunk("Текст", 0.7, filename="notes.txt", page=None)]
    prompt = build_prompt("вопрос", chunks)
    assert "[Источник: notes.txt]" in prompt
    assert "стр." not in prompt.split("[Источник: notes.txt]")[1].split("\n")[0]


def test_empty_chunks() -> None:
    prompt = build_prompt("Что-то?", [])
    assert "(контекст пуст)" in prompt
    assert "Вопрос: Что-то?" in prompt


def test_list_sources() -> None:
    chunks = [
        _chunk("x", 0.5, filename="b.md", page=2, chunk_id="2"),
        _chunk("y", 0.9, filename="a.md", page=1, chunk_id="1"),
    ]
    sources = list_sources(chunks)
    assert sources[0]["filename"] == "a.md"
    assert sources[0]["page"] == 1
    assert sources[0]["score"] == 0.9
