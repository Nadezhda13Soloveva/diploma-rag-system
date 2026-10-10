# Локальная RAG-система

Локальная база знаний на базе ИИ: хранилище неструктурированных данных + RAG на локальном компьютере

## Структура проекта

```
backend/
├── config/  # конфигурация (Pydantic Settings)
├── ingestion/  # загрузка, парсинг, чанкинг, индексация
├── retrieval/  # Hybrid retrieval (vector + BM25 + RRF)
├── generation/  # промпты, LLM, refusal logic
├── api/  # FastAPI endpoints
├── observability/  # request_id, timings, logging
└── stores/  # абстракции Vector Store, BM25, Document Store

migrations/ # SQL-схемы (pgvector)
docs/  # архитектура и документация
config.yaml # основные параметры системы
docker-compose.yml  # Ollama + PostgreSQL
```

## Быстрый старт

```bash
# 1. Окружение
cp .env.example .env
docker compose up -d

# 2. Python-зависимости
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Проверка конфига
python -c "from backend.config import settings; print(settings.llm.model)"
```

## Конфигурация

Все ключевые параметры находятся в `config.yaml` и доступны через:

```python
from backend.config import settings

print(settings.embedding.model)
print(settings.retrieval.top_k)
print(settings.vector_store.backend)
```
```
