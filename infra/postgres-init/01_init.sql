-- PostgreSQL initialization script for GraphRAG
-- Runs automatically on first container start

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- for fuzzy text matching

-- ───────────────────────────────────────────────────────────
-- Document chunks table — primary store for RAG context
-- ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_uri      TEXT        NOT NULL,
    document_title  TEXT,
    chunk_index     INTEGER     NOT NULL,
    chunk_text      TEXT        NOT NULL,
    token_count     INTEGER,
    embedding       vector(768),            -- Gemini text-embedding-004 (768-dim)
    metadata        JSONB       DEFAULT '{}',
    ingested_at     TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- HNSW index for sub-millisecond ANN cosine similarity
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- Exact-match index on source URI for batch retrieval
CREATE INDEX IF NOT EXISTS document_chunks_source_uri_idx
    ON document_chunks (source_uri);

-- GIN index on metadata JSONB for arbitrary filter pushdown
CREATE INDEX IF NOT EXISTS document_chunks_metadata_gin
    ON document_chunks USING gin (metadata);

-- ───────────────────────────────────────────────────────────
-- Ingestion job tracking table
-- ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_uri      TEXT        NOT NULL,
    status          TEXT        NOT NULL DEFAULT 'pending',  -- pending|running|done|failed
    chunks_created  INTEGER     DEFAULT 0,
    entities_extracted INTEGER  DEFAULT 0,
    error_message   TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- ───────────────────────────────────────────────────────────
-- Entity cache table — mirrors canonical Neo4j node IDs
-- Allows JOIN between Postgres chunks and Neo4j entities
-- ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entity_chunk_links (
    id              BIGSERIAL PRIMARY KEY,
    chunk_id        UUID        REFERENCES document_chunks(chunk_id) ON DELETE CASCADE,
    canonical_entity_name TEXT  NOT NULL,
    entity_type     TEXT,
    neo4j_element_id TEXT,       -- Neo4j internal element ID for direct lookup
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS entity_chunk_links_chunk_id_idx
    ON entity_chunk_links (chunk_id);

CREATE INDEX IF NOT EXISTS entity_chunk_links_entity_name_idx
    ON entity_chunk_links (canonical_entity_name);

-- ───────────────────────────────────────────────────────────
-- Trigger: auto-update updated_at on document_chunks
-- ───────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_document_chunks_updated_at ON document_chunks;
CREATE TRIGGER trg_document_chunks_updated_at
    BEFORE UPDATE ON document_chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
