"""
GraphRAG — Centralised Application Configuration
All settings are loaded from environment variables / .env file.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import AnyUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    secret_key: str = "change-me"

    # ── LLM Provider ─────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "gemini"] = "gemini"

    # ── Google Gemini ────────────────────────────────────────────────────────
    google_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    gemini_chat_model: str = "gemini-flash-lite-latest"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    gemini_embedding_dimensions: int = 768

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI secret key")
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_dimensions: int = 1536

    @property
    def embedding_dimensions(self) -> int:
        return self.gemini_embedding_dimensions if self.llm_provider == "gemini" else self.openai_embedding_dimensions

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "graphrag"
    postgres_user: str = "graphrag"
    postgres_password: str = "graphrag_secret"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_dsn_sync(self) -> str:
        """Synchronous psycopg DSN (no driver prefix)."""
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"dbname={self.postgres_db} user={self.postgres_user} "
            f"password={self.postgres_password}"
        )

    # ── Neo4j ────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_secret"
    neo4j_database: str = "neo4j"

    # ── Retrieval ────────────────────────────────────────────────────────────
    vector_top_k: int = 10
    graph_hop_depth: int = 2
    rrf_k_constant: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Ingestion ────────────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 100
    max_entities_per_chunk: int = 20
    entity_similarity_threshold: float = 0.92

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, v: str) -> str:
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — safe to call from anywhere."""
    return Settings()  # type: ignore[call-arg]
