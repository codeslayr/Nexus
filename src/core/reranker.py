"""
Reciprocal Rank Fusion (RRF) Re-ranker — Section 3, Step 6.

Merges ranked lists from vector search and graph traversal into a single
unified, noise-pruned context payload.

Also wraps a cross-encoder re-ranker (sentence-transformers) for the final
precision pass when a GPU / CPU is available.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from src.models.schemas import GraphPath, HybridContext, VectorSearchResult
from src.utils.logging import get_logger

log = get_logger(__name__)

# Lazy import — sentence-transformers is heavy; only loaded when reranker is used
_cross_encoder = None


def _get_cross_encoder(model_name: str):
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(model_name, max_length=512)
        log.info("cross_encoder_loaded", model=model_name)
    return _cross_encoder


# ──────────────────────────────────────────────────────────────────────────────
# RRF fusion
# ──────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],  # each list: ordered chunk/path IDs
    k: int = 60,
) -> dict[str, float]:
    """
    Compute RRF scores across multiple ranked lists.

    RRF(d) = Σ_r 1 / (k + rank_r(d))

    Returns: {id: rrf_score} sorted desc (higher = better).
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# ──────────────────────────────────────────────────────────────────────────────
# Context merger & re-ranker
# ──────────────────────────────────────────────────────────────────────────────

class HybridReranker:
    """
    Merges vector chunks + graph paths and applies RRF + optional cross-encoder.

    Usage:
        reranker = HybridReranker()
        hybrid_context = reranker.merge_and_rerank(
            query=query,
            vector_results=vector_results,
            graph_paths=graph_paths,
            top_n=10,
        )
    """

    def __init__(
        self,
        rrf_k: int = 60,
        reranker_model: Optional[str] = None,
        use_cross_encoder: bool = False,
    ) -> None:
        from src.config import get_settings
        settings = get_settings()
        self._rrf_k = rrf_k or settings.rrf_k_constant
        self._reranker_model = reranker_model or settings.reranker_model
        self._use_cross_encoder = use_cross_encoder

    def merge_and_rerank(
        self,
        query: str,
        vector_results: list[VectorSearchResult],
        graph_paths: list[GraphPath],
        top_n: int = 10,
    ) -> HybridContext:
        """
        1. Build ranked ID lists from vector and graph sources.
        2. Apply RRF to get unified scores.
        3. Optionally re-rank with cross-encoder.
        4. Build merged_context_text for LLM synthesis.
        """

        # ── Convert graph paths to text passages for scoring ──────────────────
        graph_passages: dict[str, str] = {
            path.path_id: path.summary for path in graph_paths
        }

        # ── RRF across vector and graph ranked lists ───────────────────────────
        vector_ranked = [r.chunk_id for r in vector_results]
        graph_ranked = [p.path_id for p in graph_paths]

        rrf_scores = reciprocal_rank_fusion(
            [vector_ranked, graph_ranked], k=self._rrf_k
        )

        # ── Build candidate pool: chunk texts + graph summaries ───────────────
        chunk_map: dict[str, VectorSearchResult] = {r.chunk_id: r for r in vector_results}
        path_map: dict[str, GraphPath] = {p.path_id: p for p in graph_paths}

        candidate_texts: list[tuple[str, str]] = []  # (id, text)
        for item_id in rrf_scores:
            if item_id in chunk_map:
                candidate_texts.append((item_id, chunk_map[item_id].chunk_text))
            elif item_id in path_map:
                candidate_texts.append((item_id, path_map[item_id].summary))

        # ── Optional cross-encoder re-ranking ─────────────────────────────────
        if self._use_cross_encoder and candidate_texts:
            candidate_texts = self._cross_encode_rerank(query, candidate_texts)

        # ── Trim to top_n ─────────────────────────────────────────────────────
        top_candidates = candidate_texts[:top_n]
        top_ids = {cid for cid, _ in top_candidates}

        # ── Filter retained results ───────────────────────────────────────────
        retained_chunks = [r for r in vector_results if r.chunk_id in top_ids]
        retained_paths = [p for p in graph_paths if p.path_id in top_ids]

        # ── Build merged context text ─────────────────────────────────────────
        context_parts: list[str] = []

        if retained_chunks:
            context_parts.append("### Retrieved Document Passages ###")
            for i, chunk in enumerate(retained_chunks, 1):
                context_parts.append(
                    f"[Chunk {i} | ID: {chunk.chunk_id} | Source: {chunk.source_uri}]\n"
                    f"{chunk.chunk_text}"
                )

        if retained_paths:
            context_parts.append("\n### Knowledge Graph Context ###")
            for path in retained_paths:
                context_parts.append(f"[Graph Path | IDs: {path.source_chunk_ids}]\n{path.summary}")

        merged_text = "\n\n---\n\n".join(context_parts)

        log.info(
            "context_merged",
            vector_chunks=len(retained_chunks),
            graph_paths=len(retained_paths),
            total_candidates=len(top_candidates),
        )

        return HybridContext(
            query=query,
            vector_chunks=retained_chunks,
            graph_paths=retained_paths,
            merged_context_text=merged_text,
            retrieval_metadata={
                "rrf_scores": {k: round(v, 6) for k, v in list(rrf_scores.items())[:top_n]},
                "cross_encoder_used": self._use_cross_encoder,
                "total_vector_candidates": len(vector_results),
                "total_graph_paths": len(graph_paths),
            },
        )

    def _cross_encode_rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """
        Use sentence-transformers CrossEncoder to rerank candidates.
        Returns candidates sorted by CE score descending.
        """
        encoder = _get_cross_encoder(self._reranker_model)
        pairs = [[query, text] for _, text in candidates]
        scores = encoder.predict(pairs)
        ranked = sorted(
            zip([cid for cid, _ in candidates], [t for _, t in candidates], scores),
            key=lambda x: x[2],
            reverse=True,
        )
        return [(cid, text) for cid, text, _ in ranked]
