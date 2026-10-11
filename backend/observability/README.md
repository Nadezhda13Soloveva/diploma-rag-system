## Observability

Каждый HTTP-запрос получает `request_id` (UUID)

### Заголовки
- Вход: можно передать `X-Request-ID`
- Выход: всегда возвращается `X-Request-ID`

### Формат логов (JSON)

```json
{
  "event": "request_completed",
  "request_id": "a1b2c3d4-...",
  "method": "GET",
  "path": "/health",
  "status_code": 200,
  "duration_ms": 12.34,
  "level": "info",
  "timestamp": "2026-10-11T12:00:00.000000Z"
}
```

### Timings в ответах api (для /ask и т п )

```json
{
  "request_id": "...",
  "answer": "...",
  "timings": {
    "embed_ms": 15.2,
    "retrieve_ms": 8.7,
    "generate_ms": 420.1,
    "total_ms": 448.5
  }
}
```

request_id автоматически попадает во все логи через structlog.contextvars
