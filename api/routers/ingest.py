"""
POST /ingest — Trigger document ingestion asynchronously via Celery.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from pipelines.orchestrator import IngestionOrchestrator
from src.models.schemas import IngestionRequest, IngestionResult
from src.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


async def _run_ingestion(request: IngestionRequest) -> IngestionResult:
    """Direct async ingestion (no Celery). Swap with .delay() in production."""
    orchestrator = IngestionOrchestrator()
    return await orchestrator.ingest(
        source_uri=request.source_uri,
        document_title=request.document_title,
        extra_metadata=request.metadata,
        force_reingest=request.force_reingest,
    )


@router.post(
    "",
    response_model=IngestionResult,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger document ingestion",
    description=(
        "Ingest a document from a URI into PostgreSQL (embeddings) and "
        "Neo4j (knowledge graph). Runs asynchronously via background task."
    ),
)
async def ingest_document(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Accepts ingestion request and dispatches to background.
    Returns a job_id for status polling.
    """
    job_id = uuid.uuid4()
    log.info("ingest_request_received", source_uri=request.source_uri, job_id=str(job_id))

    # FastAPI BackgroundTasks — swap for Celery in multi-worker deployments
    background_tasks.add_task(_run_ingestion, request)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": str(job_id),
            "source_uri": request.source_uri,
            "status": "accepted",
            "message": "Ingestion started. Poll /ingest/status/{job_id} for progress.",
        },
    )


@router.get(
    "/status/{job_id}",
    summary="Poll ingestion job status",
)
async def get_ingestion_status(job_id: uuid.UUID) -> dict:
    """Look up ingestion job status from PostgreSQL."""
    from src.utils.db import pg_cursor

    async with pg_cursor() as cur:
        await cur.execute(
            """
            SELECT job_id, source_uri, status, chunks_created,
                   entities_extracted, error_message, started_at, completed_at
            FROM ingestion_jobs
            WHERE job_id = %s
            """,
            (str(job_id),),
        )
        row = await cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return {
        "job_id": str(row[0]),
        "source_uri": row[1],
        "status": row[2],
        "chunks_created": row[3],
        "entities_extracted": row[4],
        "error_message": row[5],
        "started_at": row[6].isoformat() if row[6] else None,
        "completed_at": row[7].isoformat() if row[7] else None,
    }


@router.post(
    "/upload",
    response_model=IngestionResult,
    summary="Upload and ingest an enterprise document",
    description="Accepts PDF, DOCX, Markdown, or text files, saves them, and ingests into GraphRAG.",
)
async def upload_document(
    file: UploadFile = File(...),
    document_title: Optional[str] = Form(None),
) -> IngestionResult:
    """Accepts multipart file upload, saves locally, and runs ingestion pipeline."""
    from pathlib import Path
    import shutil

    allowed_exts = {".pdf", ".docx", ".md", ".txt"}
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported: {', '.join(allowed_exts)}",
        )

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    clean_name = Path(file.filename).name.replace(" ", "_")
    target_path = upload_dir / f"{uuid.uuid4().hex[:8]}_{clean_name}"

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    title = document_title or Path(file.filename).stem.replace("_", " ").title()
    log.info("file_uploaded_for_ingestion", file=file.filename, target=str(target_path))

    orchestrator = IngestionOrchestrator()
    result = await orchestrator.ingest(
        source_uri=str(target_path).replace("\\", "/"),
        document_title=title,
        force_reingest=True,
    )
    return result


@router.get(
    "/documents",
    summary="List all ingested documents",
    description="Returns distinct documents present in the vector store with chunk counts and titles.",
)
async def list_documents() -> list[dict]:
    """Retrieve distinct documents and chunk counts from PostgreSQL."""
    from src.utils.db import pg_cursor

    async with pg_cursor() as cur:
        await cur.execute(
            """
            SELECT 
                source_uri,
                COALESCE(document_title, source_uri) as title,
                COUNT(chunk_id)::int as chunk_count,
                MIN(ingested_at)::text as ingested_at
            FROM document_chunks
            GROUP BY source_uri, document_title
            ORDER BY MIN(ingested_at) DESC
            """
        )
        rows = await cur.fetchall()

    return [
        {
            "source_uri": row[0],
            "title": row[1] if row[1] else row[0].split("/")[-1].split("\\")[-1],
            "document_title": row[1] if row[1] else row[0].split("/")[-1].split("\\")[-1],
            "chunk_count": row[2],
            "ingested_at": row[3],
        }
        for row in rows
    ]


@router.delete(
    "/documents",
    summary="Delete a document by source URI and prune knowledge graph",
    description="Removes document chunks from PostgreSQL and prunes orphaned graph nodes in Neo4j.",
)
async def delete_document(source_uri: str = Query(...)) -> dict:
    """Delete chunks for a source_uri and prune associated graph entities."""
    from src.utils.db import pg_cursor
    from pipelines.graph_store import GraphStore

    chunk_ids: list[str] = []
    async with pg_cursor() as cur:
        await cur.execute(
            "SELECT chunk_id::text FROM document_chunks WHERE source_uri = %s",
            (source_uri,),
        )
        rows = await cur.fetchall()
        chunk_ids = [r[0] for r in rows]

        await cur.execute(
            "DELETE FROM document_chunks WHERE source_uri = %s",
            (source_uri,),
        )
        deleted_chunks = cur.rowcount

    graph_store = GraphStore()
    prune_res = await graph_store.prune_chunk_entities(chunk_ids)

    log.info(
        "document_and_graph_pruned",
        source_uri=source_uri,
        chunks_deleted=deleted_chunks,
        **prune_res,
    )
    return {
        "status": "deleted",
        "source_uri": source_uri,
        "chunks_deleted": deleted_chunks,
        "graph_pruned": prune_res,
    }


@router.get(
    "/chunks/{chunk_id}",
    summary="Get single chunk text and metadata for citation drawer",
)
async def get_chunk_detail(chunk_id: uuid.UUID) -> dict:
    """Retrieve full chunk excerpt and source metadata by UUID."""
    from src.utils.db import pg_cursor

    async with pg_cursor() as cur:
        await cur.execute(
            """
            SELECT chunk_id::text, source_uri, document_title, chunk_index, chunk_text, token_count, metadata, ingested_at
            FROM document_chunks
            WHERE chunk_id = %s
            """,
            (str(chunk_id),),
        )
        row = await cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk {chunk_id} not found",
        )

    doc_title = row[2] or row[1].split("/")[-1].split("\\")[-1]
    return {
        "id": row[0],
        "chunk_id": row[0],
        "source_uri": row[1],
        "title": doc_title,
        "document_title": doc_title,
        "chunk_index": row[3],
        "text": row[4],
        "chunk_text": row[4],
        "token_count": row[5],
        "metadata": row[6] or {},
        "ingested_at": row[7].isoformat() if row[7] else None,
    }


