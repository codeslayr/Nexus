"""
POST /query — Hybrid GraphRAG query endpoint with SSE streaming.

Returns synthesized answer + cited chunk IDs + traversed graph paths.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sse_starlette.sse import EventSourceResponse

from src.config import get_settings
from src.core.retriever import HybridGraphRAGRetriever
from src.models.schemas import HybridContext, QueryRequest, QueryResponse
from src.utils.cache import get_cached_query, set_cached_query
from src.utils.llm_factory import get_chat_llm
from src.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()

# ──────────────────────────────────────────────────────────────────────────────
# LLM synthesis prompt
# ──────────────────────────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = """You are an enterprise research assistant powered by Hybrid GraphRAG.

## Instructions:
1. Provide a clear, brief, and well-structured answer (2-4 concise paragraphs or bullet points).
2. Answer the question using ONLY the provided context — never fabricate facts.
3. Cite sources inline using [ChunkID: <id>] notation for document excerpts.
4. For knowledge graph connections, cite inline as [GraphPath: <path_summary>].
5. If the context is insufficient to answer the question, state that clearly and specify what is known.
6. Be direct, professional, and crisp.
"""

_SYNTHESIS_HUMAN = """Question: {query}

## Retrieved Context:
{context}

Provide a concise, factual, cited answer:
"""


def _build_synthesis_chain():
    llm = get_chat_llm(
        temperature=0.1,
        streaming=True,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYNTHESIS_SYSTEM),
        ("human", _SYNTHESIS_HUMAN),
    ])
    return prompt | llm | StrOutputParser()


# ──────────────────────────────────────────────────────────────────────────────
# Streaming SSE helper
# ──────────────────────────────────────────────────────────────────────────────

async def _stream_response(
    query: str,
    hybrid_context: HybridContext,
    start_time: float,
) -> AsyncIterator[dict]:
    """
    Async generator that:
    1. Streams LLM synthesis tokens as SSE data events.
    2. Emits a final 'metadata' event with cited IDs and graph paths.
    """
    chain = _build_synthesis_chain()
    cited_chunk_ids = [c.chunk_id for c in hybrid_context.vector_chunks]
    full_answer_parts: list[str] = []

    try:
        async for token in chain.astream({
            "query": query,
            "context": hybrid_context.merged_context_text,
        }):
            full_answer_parts.append(token)
            yield {"event": "token", "data": json.dumps({"token": token})}

        full_answer = "".join(full_answer_parts)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Final metadata event — guarantee single-line JSON string
        response = QueryResponse(
            answer=full_answer,
            cited_chunk_ids=cited_chunk_ids,
            graph_paths_traversed=hybrid_context.graph_paths,
            retrieval_metadata=hybrid_context.retrieval_metadata,
            latency_ms=round(latency_ms, 2),
        )
        yield {
            "event": "metadata",
            "data": json.dumps(response.model_dump()),
        }
        yield {"event": "done", "data": "{}"}

    except Exception as exc:
        log.error("streaming_synthesis_failed", error=str(exc))
        yield {"event": "error", "data": json.dumps({"error": str(exc)})}


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "",
    summary="Hybrid GraphRAG query (streaming SSE)",
    description=(
        "Runs semantic vector search + graph traversal + Text-to-Cypher, "
        "fuses results via RRF, and streams synthesized answer via SSE. "
        "Set `stream: false` for a synchronous JSON response."
    ),
)
async def query_endpoint(request: QueryRequest):
    start_time = time.perf_counter()
    log.info(
        "query_received",
        query=request.query[:100],
        stream=request.stream,
        document_uri=request.document_uri,
    )

    cache_key = f"{request.document_uri or 'all'}:{request.query}"

    # 1. Check Redis cache for instant response (<10ms)
    if not request.stream:
        cached = await get_cached_query(cache_key)
        if cached:
            cached["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
            meta = cached.get("retrieval_metadata", {})
            meta["cache_hit"] = True
            cached["retrieval_metadata"] = meta
            return QueryResponse(**cached)

    retriever = HybridGraphRAGRetriever(
        top_k=request.top_k,
        hop_depth=request.hop_depth,
    )

    hybrid_context = await retriever.retrieve(
        query=request.query,
        document_uri=request.document_uri,
        conversation_history=request.conversation_history,
        enable_hyde=request.enable_hyde,
    )

    if request.stream:
        return EventSourceResponse(
            _stream_response(request.query, hybrid_context, start_time),
            media_type="text/event-stream",
        )

    # ── Synchronous path ──────────────────────────────────────────────────────
    chain = _build_synthesis_chain()
    answer = await chain.ainvoke({
        "query": request.query,
        "context": hybrid_context.merged_context_text,
    })

    latency_ms = (time.perf_counter() - start_time) * 1000
    response = QueryResponse(
        answer=answer,
        cited_chunk_ids=[c.chunk_id for c in hybrid_context.vector_chunks],
        graph_paths_traversed=hybrid_context.graph_paths,
        retrieval_metadata=hybrid_context.retrieval_metadata,
        latency_ms=round(latency_ms, 2),
    )

    # Store in Redis cache for 30 minutes
    await set_cached_query(cache_key, response.model_dump())
    return response
