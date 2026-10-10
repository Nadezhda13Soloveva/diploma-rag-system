from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseModel):
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    dimension: int = 768


class LLMConfig(BaseModel):
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    temperature: float = 0.1
    max_tokens: int = 1024
    base_url: str = "http://localhost:11434"


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "recursive", "semantic", "parent_document"] = "recursive"
    chunk_size: int = 800
    chunk_overlap: int = 150


class RetrievalConfig(BaseModel):
    top_k: int = 8
    hybrid: bool = True
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    min_score: float = 0.32
    refusal_threshold: float = 0.25
    min_relevant_chunks: int = 1


class VectorStoreConfig(BaseModel):
    backend: Literal["chroma", "pgvector"] = "chroma"
    persist_directory: str = "./data/chroma"
    collection_name: str = "documents"


class DatabaseConfig(BaseModel):
    url: str = "postgresql://rag:rag@localhost:5432/rag"


class PathsConfig(BaseModel):
    documents: str = "./data/documents"
    index: str = "./data/index"
    evaluation: str = "./data/evaluation"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "Settings":
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)


@lru_cache
def get_settings() -> Settings:
    return Settings.from_yaml("config.yaml")


settings = get_settings()
