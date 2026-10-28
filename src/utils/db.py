"""
PostgreSQL connection pool management using psycopg3 + SQLAlchemy 2 async engine.

Usage:
    from src.utils.db import get_pg_pool, close_pg_pool
    pool = await get_pg_pool()
    async with pool.connection() as conn:
        ...
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import psycopg
import psycopg_pool
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)

_pool: psycopg_pool.AsyncConnectionPool | None = None
_engine: AsyncEngine | None = None


async def get_pg_pool() -> psycopg_pool.AsyncConnectionPool:
    """Return (or lazily create) the shared async connection pool."""
    global _pool
    if _pool is None or _pool.closed:
        settings = get_settings()
        _pool = psycopg_pool.AsyncConnectionPool(
            conninfo=settings.postgres_dsn_sync,
            min_size=2,
            max_size=20,
            max_waiting=50,
            max_idle=300,
            reconnect_timeout=30,
            open=False,
        )
        await _pool.open()
        log.info("pg_pool_opened", min=2, max=20)
    return _pool


async def close_pg_pool() -> None:
    global _pool
    if _pool and not _pool.closed:
        await _pool.close()
        log.info("pg_pool_closed")
    _pool = None


def get_sqlalchemy_engine() -> AsyncEngine:
    """SQLAlchemy async engine for ORM-style queries (used by pgvector helpers)."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.postgres_dsn,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=settings.app_env == "development",
        )
    return _engine


@asynccontextmanager
async def pg_cursor() -> AsyncGenerator[psycopg.AsyncCursor, None]:
    """Convenience context manager yielding a ready cursor."""
    pool = await get_pg_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            yield cur
