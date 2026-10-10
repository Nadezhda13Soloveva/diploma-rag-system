
## Запуск окружения

### Требования
- Docker + Docker Compose v2
- ~10 ГБ свободного места (модели + данные)

### Быстрый старт

```bash
# 1. Создать .env из примера
cp .env.example .env

# 2. Поднять все сервисы
docker compose up -d

# 3. Проверить статус
docker compose ps
```

После запуска будут доступны:

| Сервис | Адрес | Назначение  |
|---|---|----|
| Ollama | http://localhost:11434 | LLM + Embeddings |
| PostgreSQL | localhost:5432 | pgvector (user/pass/db: rag) |

### Скачивание моделей

```bash
# Зайти в контейнер Ollama
docker exec -it local-rag-ollama ollama pull nomic-embed-text
docker exec -it local-rag-ollama ollama pull qwen2.5:7b-instruct-q4_K_M

# Проверить
docker exec -it local-rag-ollama ollama list
```

Или с хоста (если порт проброшен):

```bash
curl http://localhost:11434/api/tags
```

### Полезные команды

```bash
# Остановить без удаления данных
docker compose down

# Остановить и удалить volumes (полный сброс)
docker compose down -v

# Логи
docker compose logs -f ollama
docker compose logs -f db

# Статус
docker compose ps
```

### Сохранение данных

- Модели Ollama хранятся в volume `ollama_data`
- Данные PostgreSQL — в volume `pgdata`

После `docker compose down` и повторного `docker compose up -d` все данные и модели сохраняются.

### Проверка связности

Контейнеры находятся в одной сети `rag-network`.
Из backend-контейнера (когда появится) Ollama будет доступен по имени `ollama:11434`, а БД — по `db:5432`
```
