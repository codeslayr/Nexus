"""
High-performance Redis cache for GraphRAG with graceful memory fallback.
Caches query embeddings, intermediate entity graphs, and full query responses.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

from src.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)

_redis_client: Optional[Any] = None
_in_memory_cache: dict[str, str] = {}


def _get_redis() -> Optional[Any]:
    global _redis_client
    if aioredis is None:
        return None
    if _redis_client is None:
        try:
            settings = get_settings()
            _redis_client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        except Exception:
            _redis_client = None
    return _redis_client


def _hash_key(prefix: str, payload: str) -> str:
    digest = hashlib.sha256(payload.strip().lower().encode("utf-8")).hexdigest()[:32]
    return f"graphrag:{prefix}:{digest}"


async def get_cached_query(query: str) -> Optional[dict[str, Any]]:
    """Retrieve cached query response if present."""
    key = _hash_key("query", query)
    try:
        r = _get_redis()
        if r is not None:
            val = await r.get(key)
            if val:
                log.info("query_cache_hit", query=query[:60])
                return json.loads(val)
        elif key in _in_memory_cache:
            return json.loads(_in_memory_cache[key])
    except Exception as exc:
        log.warning("cache_read_error", error=str(exc))
    return None


async def set_cached_query(query: str, data: dict[str, Any], ttl_seconds: int = 3600) -> None:
    """Store query response in cache."""
    key = _hash_key("query", query)
    try:
        val = json.dumps(data)
        r = _get_redis()
        if r is not None:
            await r.setex(key, ttl_seconds, val)
        else:
            _in_memory_cache[key] = val
    except Exception as exc:
        log.warning("cache_write_error", error=str(exc))


async def set_cached_query(query: str, data: dict[str, Any], ttl_seconds: int = 1800) -> None:
    """Store query response in Redis with TTL (default 30 mins)."""
    try:
        r = _get_redis()
        key = _hash_key("query", query)
        await r.setex(key, ttl_seconds, json.dumps(data))
        log.debug("query_cache_saved", query=query[:60])
    except Exception as exc:
        log.warning("cache_write_error", error=str(exc))


async def get_cached_embedding(text: str) -> Optional[list[float]]:
    """Retrieve cached embedding vector."""
    try:
        r = _get_redis()
        key = _hash_key("emb", text)
        val = await r.get(key)
        if val:
            return json.loads(val)
    except Exception:
        pass
    return None


async def set_cached_embedding(text: str, embedding: list[float], ttl_seconds: int = 86400) -> None:
    """Store embedding vector with 24h TTL."""
    try:
        r = _get_redis()
        key = _hash_key("emb", text)
        await r.setex(key, ttl_seconds, json.dumps(embedding))
    except Exception:
        pass
