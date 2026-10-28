"""
Pydantic domain models shared across ingestion, retrieval, and API layers.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ──────────────────────────────────────────────────────────────────────────────
# Document & Chunk models
# ──────────────────────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A single tokenised segment of a source document."""

    chunk_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_uri: str
    document_title: Optional[str] = None
    chunk_index: int
    chunk_text: str
    token_count: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None  # filled after embed step
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class IngestionRequest(BaseModel):
    source_uri: str = Field(..., description="File path or HTTP URL of the document")
    document_title: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    force_reingest: bool = False


class IngestionResult(BaseModel):
    job_id: uuid.UUID
    source_uri: str
    status: str
    chunks_created: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Knowledge Graph extraction models (strict Pydantic schemas)
# ──────────────────────────────────────────────────────────────────────────────

class Entity(BaseModel):
    """A canonical named entity extracted from a document chunk."""

    name: str = Field(..., description="Canonical display name of the entity")
    type: str = Field(
        ...,
        description="Entity type: PERSON | ORG | PRODUCT | LOCATION | CONCEPT | EVENT | METRIC | DEPARTMENT | REGULATION | TECHNOLOGY | SECURITY",
    )
    description: Optional[str] = Field(
        None, description="Short factual description from source text"
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Known alternative names / ticker symbols / abbreviations",
    )

    @model_validator(mode="after")
    def _normalise_name(self) -> "Entity":
        self.name = self.name.strip().title()
        self.type = self.type.upper()
        return self


class Relationship(BaseModel):
    """A directed triple (source) -[type]-> (target) with provenance."""

    source: str = Field(..., description="Canonical source entity name")
    target: str = Field(..., description="Canonical target entity name")
    type: str = Field(
        ...,
        description="Relationship type in SCREAMING_SNAKE_CASE, e.g. ACQUIRED, INVESTED_IN",
    )
    context: Optional[str] = Field(
        None, description="Verbatim supporting sentence from source text"
    )
    context_chunk_id: str = Field(
        ..., description="UUID of the chunk this relationship was extracted from"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _normalise_type(self) -> "Relationship":
        self.type = self.type.upper().replace(" ", "_")
        return self


class KGExtractionResult(BaseModel):
    """Batch extraction output from a single chunk."""

    chunk_id: str
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval models
# ──────────────────────────────────────────────────────────────────────────────

class VectorSearchResult(BaseModel):
    chunk_id: str
    source_uri: str
    chunk_text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    rank: int = 0


class GraphPath(BaseModel):
    """A traversed Neo4j path returned as structured context."""

    path_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    nodes: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    summary: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class HybridContext(BaseModel):
    """Unified retrieval payload fed into the LLM synthesis step."""

    query: str
    vector_chunks: list[VectorSearchResult]
    graph_paths: list[GraphPath]
    merged_context_text: str
    entity_seeds: list[str] = Field(default_factory=list)
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# API request / response models
# ──────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    hop_depth: int = Field(default=2, ge=1, le=3)
    stream: bool = True
    document_uri: Optional[str] = Field(
        default=None,
        description="Filter retrieval to chunks and entities from a specific document URI",
    )
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="Recent conversation turns formatted as [{role: 'user'|'assistant', text: '...'}, ...]",
    )
    enable_hyde: bool = Field(
        default=False,
        description="Enable Hypothetical Document Embeddings for conceptual query expansion",
    )


class DocumentInfo(BaseModel):
    source_uri: str
    document_title: str
    chunk_count: int
    ingested_at: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    cited_chunk_ids: list[str]
    graph_paths_traversed: list[GraphPath]
    retrieval_metadata: dict[str, Any]
    latency_ms: float


class SubgraphResponse(BaseModel):
    entity_id: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
