"""
Celery worker configuration for async ingestion jobs.
Start worker: celery -A pipelines.celery_worker worker --loglevel=info --concurrency=4
"""
from __future__ import annotations

import asyncio
from celery import Celery

from src.config import get_settings

settings = get_settings()

celery_app = Celery(
    "graphrag",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_soft_time_limit=600,   # 10 min soft limit
    task_time_limit=900,        # 15 min hard limit
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)


@celery_app.task(
    name="graphrag.ingest_document",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(Exception,),
)
def ingest_document_task(
    self,
    source_uri: str,
    document_title: str | None = None,
    extra_metadata: dict | None = None,
    force_reingest: bool = False,
) -> dict:
    """
    Celery task wrapping the async IngestionOrchestrator.
    """
    from pipelines.orchestrator import IngestionOrchestrator

    orchestrator = IngestionOrchestrator()

    result = asyncio.get_event_loop().run_until_complete(
        orchestrator.ingest(
            source_uri=source_uri,
            document_title=document_title,
            extra_metadata=extra_metadata,
            force_reingest=force_reingest,
        )
    )
    return result.model_dump(mode="json")
