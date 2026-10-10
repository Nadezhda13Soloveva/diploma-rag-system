-- Инит схемы для локальной RAG-системы
-- Embedding model: nomic-embed-text (768 измерений)

CREATE EXTENSION IF NOT EXISTS vector;

-- основная таблица чанков
CREATE TABLE IF NOT EXISTS documents (
    id              BIGSERIAL PRIMARY KEY,
    content         TEXT NOT NULL,
    embedding       vector(768),
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE documents IS 'Чанки документов с векторными представлениями';
COMMENT ON COLUMN documents.embedding IS 'Эмбеддинг размерности 768 (nomic-embed-text)';
COMMENT ON COLUMN documents.metadata IS 'Метаданные: source, filename, page, chunk_index, document_id и т.д.';

-- HNSW-индекс для семантического поиска (cosine distance)
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
    ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN-индекс по metadata (для фильтра по source, file_type, ...)
CREATE INDEX IF NOT EXISTS documents_metadata_gin_idx
    ON documents
    USING gin (metadata);

-- индекс по времени создания
CREATE INDEX IF NOT EXISTS documents_created_at_idx
    ON documents (created_at DESC);
