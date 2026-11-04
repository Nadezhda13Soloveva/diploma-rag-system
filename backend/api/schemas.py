from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Вопрос пользователя")
    top_k: int | None = Field(default=None, ge=1, le=50)


class TimingInfo(BaseModel):
    embed_ms: float | None = None
    retrieve_ms: float | None = None
    generate_ms: float | None = None
    total_ms: float | None = None


class SourceInfo(BaseModel):
    filename: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    score: float | None = None


class AskResponse(BaseModel):
    request_id: str
    answer: str
    refused: bool = False
    refusal_reason: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    timings: TimingInfo
    model: str | None = None
