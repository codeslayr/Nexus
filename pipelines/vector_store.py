"""
PostgreSQL vector store — Section 2, Step 2.

Embeds chunks via OpenAI and upserts into document_chunks with pgvector HNSW.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Optional

import psycopg

from src.config import get_settings
from src.models.schemas import DocumentChunk, VectorSearchResult
from src.utils.db import get_pg_pool
from src.utils.llm_factory import get_embeddings
from src.utils.logging import get_logger

log = get_logger(__name__)


class VectorStore:
    """
    Async wrapper around the document_chunks table.
    Handles batch embedding + upsert + similarity search.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._embeddings = get_embeddings()

    # ── Embedding ─────────────────────────────────────────────────────────────

    async def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Batch-embed all chunks, attaching embedding to each chunk object."""
        texts = [c.chunk_text for c in chunks]
        # LangChain OpenAI embeddings natively batches API calls
        vectors = await self._embeddings.aembed_documents(texts)
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
        log.info("chunks_embedded", count=len(chunks))
        return chunks

    # ── Upsert ────────────────────────────────────────────────────────────────

    async def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        Upsert chunks into PostgreSQL.
        ON CONFLICT (chunk_id) → updates text and embedding.
        Returns the number of rows written.
        """
        if not chunks:
            return 0

        UPSERT_SQL = """
            INSERT INTO document_chunks
                (chunk_id, source_uri, document_title, chunk_index, chunk_text,
                 token_count, embedding, metadata)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s::vector, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                chunk_text    = EXCLUDED.chunk_text,
                embedding     = EXCLUDED.embedding,
                token_count   = EXCLUDED.token_count,
                metadata      = EXCLUDED.metadata,
                updated_at    = now()
        """

        pool = await get_pg_pool()
        written = 0

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                for chunk in chunks:
                    if chunk.embedding is None:
                        log.warning("chunk_missing_embedding", chunk_id=str(chunk.chunk_id))
                        continue

                    await cur.execute(
                        UPSERT_SQL,
                        (
                            str(chunk.chunk_id),
                            chunk.source_uri,
                            chunk.document_title,
                            chunk.chunk_index,
                            chunk.chunk_text,
                            chunk.token_count,
                            # pgvector expects string format "[1.0, 2.0, ...]"
                            "[" + ",".join(map(str, chunk.embedding)) + "]",
                            json.dumps(chunk.metadata),
                        ),
                    )
                    written += 1

            await conn.commit()

        log.info("chunks_upserted", count=written)
        return written

    # ── Vector Similarity Search ───────────────────────────────────────────────

    async def similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        source_uri: Optional[str] = None,
    ) -> list[VectorSearchResult]:
        """
        Cosine similarity ANN search using HNSW index.
        Optionally filter by source_uri or apply metadata JSONB filter.
        """
        k = top_k or self._settings.vector_top_k
        query_vector = await self._embeddings.aembed_query(query)
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"

        conditions: list[str] = []
        where_params: list[Any] = []

        if source_uri:
            conditions.append("source_uri = %s")
            where_params.append(source_uri)

        if metadata_filter:
            for key, val in metadata_filter.items():
                conditions.append(f"metadata->>{json.dumps(key)} = %s")
                where_params.append(str(val))

        WHERE_CLAUSE = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        SEARCH_SQL = f"""
            SELECT
                chunk_id::text,
                source_uri,
                chunk_text,
                metadata,
                1 - (embedding <=> %s::vector) AS cosine_score
            FROM document_chunks
            {WHERE_CLAUSE}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        pool = await get_pg_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                sql_params = [vector_str] + where_params + [vector_str, k]
                await cur.execute(SEARCH_SQL, sql_params)
                rows = await cur.fetchall()

        results = [
            VectorSearchResult(
                chunk_id=row[0],
                source_uri=row[1],
                chunk_text=row[2],
                metadata=row[3] or {},
                score=float(row[4]),
                rank=i + 1,
            )
            for i, row in enumerate(rows)
        ]

        log.info("vector_search_complete", query_len=len(query), results=len(results))
        return results

    # ── Delete by source URI ───────────────────────────────────────────────────

    async def delete_by_source(self, source_uri: str) -> int:
        pool = await get_pg_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM document_chunks WHERE source_uri = %s RETURNING chunk_id",
                    (source_uri,),
                )
                deleted = await cur.fetchall()
            await conn.commit()
        log.info("chunks_deleted", source_uri=source_uri, count=len(deleted))
        return len(deleted)
