from __future__ import annotations

from fastapi import FastAPI

from backend.api.middleware import RequestIdMiddleware
from backend.api.routes import health
from backend.observability.logging import setup_logging

setup_logging(json_logs=True)

app = FastAPI(
    title="Local RAG API",
    version="0.1.0",
    description="Локальная база знаний на базе ИИ",
)

app.add_middleware(RequestIdMiddleware)
app.include_router(health.router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "local-rag-api",
        "docs": "/docs",
        "health": "/health",
    }
