"""
Ingestion Orchestrator — coordinates chunking → embedding → PG upsert → KG extraction → Neo4j write.

This is the top-level pipeline entry point, called from the FastAPI /ingest
endpoint via a Celery task for async execution.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from pipelines.chunker import DocumentChunker
from pipelines.entity_resolver import EntityResolver
from pipelines.graph_store import GraphStore
from pipelines.kg_extractor import KGExtractor
from pipelines.vector_store import VectorStore
from src.models.schemas import IngestionResult
from src.utils.db import get_pg_pool
from src.utils.logging import get_logger

log = get_logger(__name__)


class IngestionOrchestrator:
    """
    End-to-end ingestion pipeline.

    Pipeline stages:
        1. Chunk document               → list[DocumentChunk]
        2. Embed chunks                 → filled embeddings
        3. Upsert to PostgreSQL         → chunk_ids persisted
        4. Extract KG triples (async)   → KGExtractionResult per chunk
        5. Resolve entity canonicalization
        6. Upsert entities + edges to Neo4j
        7. Record chunk→entity links in Postgres
    """

    def __init__(self) -> None:
        self._chunker = DocumentChunker()
        self._vector_store = VectorStore()
        self._kg_extractor = KGExtractor(concurrency=5)
        self._graph_store = GraphStore()

    async def ingest(
        self,
        source_uri: str,
        document_title: Optional[str] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        force_reingest: bool = False,
    ) -> IngestionResult:
        job_id = uuid.uuid4()
        log.info("ingestion_started", job_id=str(job_id), source_uri=source_uri)

        # ── Record job as running ──────────────────────────────────────────────
        await self._record_job(job_id, source_uri, "running")

        try:
            # ── Stage 1: Chunk ─────────────────────────────────────────────────
            chunks = self._chunker.chunk_document(
                source_uri=source_uri,
                document_title=document_title,
                extra_metadata=extra_metadata,
            )

            if force_reingest:
                await self._vector_store.delete_by_source(source_uri)

            # ── Stage 2: Embed ─────────────────────────────────────────────────
            chunks = await self._vector_store.embed_chunks(chunks)

            # ── Stage 3: Upsert chunks to PostgreSQL ──────────────────────────
            chunks_written = await self._vector_store.upsert_chunks(chunks)

            # ── Stage 4: Extract KG triples ────────────────────────────────────
            batch_input = [(str(c.chunk_id), c.chunk_text) for c in chunks]
            extraction_results = await self._kg_extractor.extract_batch(batch_input)

            # ── Stage 5 & 6: Resolve & write to Neo4j ─────────────────────────
            resolver = EntityResolver()
            total_entities = 0
            total_rels = 0
            pg_link_rows: list[tuple] = []

            for result in extraction_results:
                # Resolve each entity to canonical name
                resolved_entities = []
                entity_name_map: dict[str, str] = {}  # original → canonical

                for entity in result.entities:
                    canonical_name = await resolver.resolve_with_embedding(entity)
                    entity_name_map[entity.name] = canonical_name
                    # Rewrite entity with canonical name for Neo4j write
                    resolved = entity.model_copy(update={"name": canonical_name})
                    resolved_entities.append(resolved)

                # Rewrite relationships with canonical names
                resolved_rels = []
                for rel in result.relationships:
                    resolved_rel = rel.model_copy(update={
                        "source": entity_name_map.get(rel.source, rel.source),
                        "target": entity_name_map.get(rel.target, rel.target),
                    })
                    resolved_rels.append(resolved_rel)

                result.entities = resolved_entities
                result.relationships = resolved_rels

                counts = await self._graph_store.upsert_extraction_result(result)
                total_entities += counts["entities"]
                total_rels += counts["relationships"]

                # Collect chunk→entity links for Postgres
                for entity in resolved_entities:
                    pg_link_rows.append((result.chunk_id, entity.name, entity.type))

            # ── Stage 7: Write chunk→entity links to Postgres ─────────────────
            await self._write_entity_chunk_links(pg_link_rows)

            # ── Finalise job ──────────────────────────────────────────────────
            result_obj = IngestionResult(
                job_id=job_id,
                source_uri=source_uri,
                status="done",
                chunks_created=chunks_written,
                entities_extracted=total_entities,
                relationships_extracted=total_rels,
            )
            await self._record_job(
                job_id, source_uri, "done",
                chunks_created=chunks_written,
                entities_extracted=total_entities,
            )

            log.info(
                "ingestion_complete",
                job_id=str(job_id),
                chunks=chunks_written,
                entities=total_entities,
                rels=total_rels,
            )
            return result_obj

        except Exception as exc:
            log.error("ingestion_failed", job_id=str(job_id), error=str(exc))
            await self._record_job(job_id, source_uri, "failed", error=str(exc))
            raise

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _record_job(
        self,
        job_id: uuid.UUID,
        source_uri: str,
        status: str,
        chunks_created: int = 0,
        entities_extracted: int = 0,
        error: Optional[str] = None,
    ) -> None:
        SQL = """
            INSERT INTO ingestion_jobs
                (job_id, source_uri, status, chunks_created, entities_extracted,
                 error_message, started_at, completed_at)
            VALUES (%s, %s, %s, %s, %s, %s,
                    CASE WHEN %s = 'running' THEN now() ELSE NULL END,
                    CASE WHEN %s IN ('done','failed') THEN now() ELSE NULL END)
            ON CONFLICT (job_id) DO UPDATE SET
                status             = EXCLUDED.status,
                chunks_created     = EXCLUDED.chunks_created,
                entities_extracted = EXCLUDED.entities_extracted,
                error_message      = EXCLUDED.error_message,
                completed_at       = CASE WHEN EXCLUDED.status IN ('done','failed') THEN now() ELSE NULL END
        """
        pool = await get_pg_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    SQL,
                    (str(job_id), source_uri, status, chunks_created,
                     entities_extracted, error, status, status),
                )
            await conn.commit()

    async def _write_entity_chunk_links(
        self, rows: list[tuple]
    ) -> None:
        """Upsert (chunk_id, entity_name, entity_type) into entity_chunk_links."""
        if not rows:
            return
        SQL = """
            INSERT INTO entity_chunk_links (chunk_id, canonical_entity_name, entity_type)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
        """
        pool = await get_pg_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(SQL, rows)
            await conn.commit()
        log.debug("entity_chunk_links_written", count=len(rows))
