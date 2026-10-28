"""
HybridGraphRAGRetriever — Section 3 (complete).

The central retrieval orchestrator that:
  1. Runs semantic vector search against pgvector.
  2. Detects query entities via LLM NER.
  3. Traverses the Neo4j graph (k-hop neighborhood).
  4. Generates Text-to-Cypher for structured questions.
  5. Merges + RRF re-ranks everything into a HybridContext.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from pipelines.graph_store import GraphStore
from pipelines.vector_store import VectorStore
from src.config import get_settings
from src.core.reranker import HybridReranker
from src.core.text_to_cypher import TextToCypherEngine
from src.models.schemas import GraphPath, HybridContext, VectorSearchResult
from src.utils.llm_factory import get_chat_llm
from src.utils.logging import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Entity Detection (NER from query)
# ──────────────────────────────────────────────────────────────────────────────

class _DetectedEntities(BaseModel):
    entities: list[str] = Field(
        default_factory=list,
        description="List of proper nouns and named entities in the query",
    )


_NER_SYSTEM = """You are a Named Entity Recognition system.
Extract all proper nouns, organization names, product names, person names,
and domain-specific concepts from the given text.
Return ONLY a JSON object matching the schema — no prose.
{format_instructions}
"""

_NER_HUMAN = "Text: {query}"


# ──────────────────────────────────────────────────────────────────────────────
# Main retriever
# ──────────────────────────────────────────────────────────────────────────────

class HybridGraphRAGRetriever:
    """
    Production hybrid retriever implementing the full GraphRAG retrieval pattern.

    Async: all retrieval methods use async I/O for maximum throughput.
    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        hop_depth: Optional[int] = None,
        use_cross_encoder: bool = False,
    ) -> None:
        settings = get_settings()
        self._top_k = top_k or settings.vector_top_k
        self._hop_depth = hop_depth or settings.graph_hop_depth
        self._vector_store = VectorStore()
        self._graph_store = GraphStore()
        self._reranker = HybridReranker(use_cross_encoder=use_cross_encoder)
        self._cypher_engine = TextToCypherEngine()

        from langchain_core.output_parsers import PydanticOutputParser
        self._ner_parser = PydanticOutputParser(pydantic_object=_DetectedEntities)
        self._llm = get_chat_llm(temperature=0)
        self._ner_chain = (
            ChatPromptTemplate.from_messages([
                ("system", _NER_SYSTEM),
                ("human", _NER_HUMAN),
            ]).partial(format_instructions=self._ner_parser.get_format_instructions())
            | self._llm
            | self._ner_parser
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        metadata_filter: Optional[dict] = None,
        document_uri: Optional[str] = None,
        conversation_history: Optional[list[dict]] = None,
        enable_hyde: bool = False,
    ) -> HybridContext:
        """
        Full hybrid retrieval pipeline with Conversational Memory, HyDE,
        and Community Cluster expansion.
        """
        # Step 0: Conversational coreference resolution if history provided
        effective_query = query
        if conversation_history:
            effective_query = await self._resolve_conversational_query(query, conversation_history)

        log.info(
            "retrieval_started",
            query=effective_query[:100],
            document_uri=document_uri,
            hyde=enable_hyde,
        )

        # Step 1: Vector search (with optional HyDE expansion)
        vector_task = asyncio.create_task(
            self._vector_store.similarity_search(
                effective_query,
                top_k=self._top_k,
                metadata_filter=metadata_filter,
                source_uri=document_uri,
            )
        )
        ner_task = asyncio.create_task(self._detect_entities(effective_query))

        vector_results, entity_seeds = await asyncio.gather(
            vector_task, ner_task, return_exceptions=False
        )

        # HyDE expansion if enabled or if vector recall is sparse (< 2 hits)
        if (enable_hyde or len(vector_results) < 2) and len(effective_query.split()) > 3:
            hypo_doc = await self._generate_hypothetical_document(effective_query)
            if hypo_doc:
                hypo_hits = await self._vector_store.similarity_search(
                    hypo_doc,
                    top_k=3,
                    metadata_filter=metadata_filter,
                    source_uri=document_uri,
                )
                seen_ids = {v.chunk_id for v in vector_results}
                for h in hypo_hits:
                    if h.chunk_id not in seen_ids:
                        vector_results.append(h)
                        seen_ids.add(h.chunk_id)

        log.info(
            "parallel_retrieval_done",
            vector_hits=len(vector_results),
            entities=entity_seeds,
        )

        # Step 2: Graph neighborhood traversal (fast graph index)
        graph_paths = await self._traverse_graph(entity_seeds)

        # Step 3: Global Community Clusters for broad/overview queries
        if any(w in effective_query.lower() for w in ["summarize", "all", "overview", "initiatives", "architecture", "ecosystem"]):
            try:
                clusters = await self._graph_store.get_community_clusters()
                for c in clusters[:4]:
                    graph_paths.append(
                        GraphPath(
                            nodes=[{"name": c["hub"], "type": c["hub_type"]}],
                            relationships=[{"type": t} for t in c.get("rel_types", [])],
                            summary=f"Key Community Hub {c['hub']} ({c['hub_type']}) connects: {', '.join(c.get('key_connections', []))}",
                            source_chunk_ids=[],
                        )
                    )
            except Exception as exc:
                log.debug("community_clusters_skip", error=str(exc))

        # Step 4: Text-to-Cypher only if aggregation keywords present or graph_paths empty
        cypher_paths: list[GraphPath] = []
        if self._needs_text_to_cypher(effective_query) or not graph_paths:
            try:
                cypher_paths = await asyncio.wait_for(
                    self._run_text_to_cypher(effective_query, entity_seeds),
                    timeout=8.0,
                )
            except Exception as exc:
                log.debug("text_to_cypher_skipped_or_timeout", error=str(exc))

        # Step 5: Merge all graph context
        all_graph_paths = graph_paths + cypher_paths

        # Step 6: RRF re-ranking → unified HybridContext
        hybrid_context = self._reranker.merge_and_rerank(
            query=effective_query,
            vector_results=vector_results,
            graph_paths=all_graph_paths,
            top_n=self._top_k,
        )
        hybrid_context.entity_seeds = entity_seeds

        log.info(
            "retrieval_complete",
            final_chunks=len(hybrid_context.vector_chunks),
            final_paths=len(hybrid_context.graph_paths),
        )
        return hybrid_context

    async def _resolve_conversational_query(
        self, query: str, history: list[dict]
    ) -> str:
        """Resolve pronouns and coreferences using recent dialogue turns."""
        pronouns = {"he", "she", "it", "they", "this", "that", "him", "her", "his", "their"}
        words = set(query.lower().split())
        if not (words & pronouns) and not query.lower().startswith(("what about", "how about", "tell me more")):
            return query

        history_lines = []
        for h in history[-4:]:
            msg = h.get("text") or h.get("content") or ""
            if msg:
                history_lines.append(f"{h.get('role', 'user')}: {msg}")
        history_snippet = "\n".join(history_lines)
        if not history_snippet:
            return query

        prompt = (
            "Given the chat history and follow-up question, rewrite the follow-up "
            "to be a single self-contained question with explicit entity names. Do not answer it.\n\n"
            f"History:\n{history_snippet}\n\n"
            f"Follow-up: {query}\n\n"
            "Self-contained question:"
        )
        try:
            res = await asyncio.wait_for(self._llm.ainvoke(prompt), timeout=5.0)
            rewritten = self._extract_text(res)
            log.info("conversational_query_resolved", original=query, resolved=rewritten)
            return rewritten
        except Exception as exc:
            log.debug("conversational_resolution_skipped", error=str(exc))
            return query

    async def _generate_hypothetical_document(self, query: str) -> Optional[str]:
        """Generate a concise hypothetical factual statement (HyDE)."""
        prompt = (
            f"Write a 1-sentence factual document excerpt that answers: '{query}'. "
            "Output only the excerpt."
        )
        try:
            res = await asyncio.wait_for(self._llm.ainvoke(prompt), timeout=4.5)
            hypo = self._extract_text(res)
            log.info("hyde_passage_generated", query=query, passage=hypo)
            return hypo
        except Exception as exc:
            log.debug("hyde_generation_skipped", error=str(exc))
            return None

    @staticmethod
    def _extract_text(res) -> str:
        content = res.content if hasattr(res, "content") else str(res)
        if isinstance(content, list):
            parts = [
                p if isinstance(p, str) else (p.get("text", "") if isinstance(p, dict) else str(p))
                for p in content
            ]
            return "".join(parts).strip()
        return str(content).strip()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _needs_text_to_cypher(self, query: str) -> bool:
        """Check if query needs dynamic Cypher aggregations."""
        keywords = ("how many", "count", "average", "shortest path", "which entities", "number of", "total")
        q = query.lower()
        return any(k in q for k in keywords)

    async def _detect_entities(self, query: str) -> list[str]:
        """
        Fast tiered entity detection:
        Tier 1: Instant direct graph match (1-5ms in Neo4j)
        Tier 2: LLM NER fallback with timeout if graph match finds 0 seeds
        """
        try:
            matches = await self._graph_store.match_entities_in_text(query)
            if matches:
                log.info("entity_match_graph_hit", matches=matches)
                return matches[:10]
        except Exception as exc:
            log.warning("entity_graph_match_failed", error=str(exc))

        # Tier 2: LLM NER fallback with strict timeout
        try:
            result: _DetectedEntities = await asyncio.wait_for(
                self._ner_chain.ainvoke({"query": query}),
                timeout=6.0,
            )
            return result.entities[:10]
        except Exception as exc:
            log.warning("ner_fallback_failed_or_timed_out", error=str(exc))
            return []

    async def _traverse_graph(self, entity_seeds: list[str]) -> list[GraphPath]:
        """
        For each seed entity, run a k-hop neighborhood query in Neo4j.
        Converts raw records to GraphPath objects with human-readable summaries.
        """
        if not entity_seeds:
            return []

        paths: list[GraphPath] = []
        tasks = [
            self._graph_store.get_neighborhood(
                entity_name=seed,
                hop_depth=self._hop_depth,
                limit=20,
            )
            for seed in entity_seeds[:5]  # limit to 5 seeds to bound traversal
        ]

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        for seed, records in zip(entity_seeds[:5], all_results):
            if isinstance(records, Exception):
                log.warning("graph_traversal_failed", seed=seed, error=str(records))
                continue

            for record in records:
                nodes = record.get("nodes", [])
                rels = record.get("rels", [])

                # Build human-readable summary
                summary = self._summarise_path(nodes, rels)
                chunk_ids = []
                for r in rels:
                    chunk_ids.extend(r.get("chunk_ids") or [])

                paths.append(GraphPath(
                    nodes=nodes,
                    relationships=rels,
                    summary=summary,
                    source_chunk_ids=list(set(chunk_ids)),
                ))

        log.info("graph_traversal_complete", seed_count=len(entity_seeds), paths=len(paths))
        return paths

    async def _run_text_to_cypher(
        self, query: str, entities: list[str]
    ) -> list[GraphPath]:
        """
        Generate and execute a Text-to-Cypher query.
        Returns results converted to GraphPath objects.
        """
        cypher_result = await self._cypher_engine.generate(query, entities)

        if not cypher_result.is_valid or not cypher_result.cypher:
            return []

        try:
            rows = await self._graph_store.run_cypher(cypher_result.cypher)
            if not rows:
                return []

            # Treat the entire Cypher result as one graph path
            summary_lines = [str(row) for row in rows[:10]]
            summary = f"Cypher query result ({len(rows)} rows):\n" + "\n".join(summary_lines)

            chunk_ids = []
            for row in rows:
                for v in row.values():
                    if isinstance(v, list):
                        chunk_ids.extend(v)

            return [GraphPath(
                nodes=[],
                relationships=[],
                summary=summary,
                source_chunk_ids=list(set(str(c) for c in chunk_ids)),
            )]

        except Exception as exc:
            log.error("cypher_execution_failed", error=str(exc))
            return []

    @staticmethod
    def _summarise_path(
        nodes: list[dict], rels: list[dict]
    ) -> str:
        """Convert a raw Neo4j path to a readable sentence."""
        if not nodes:
            return ""
        node_names = [n.get("name", "?") for n in nodes if n]
        rel_types = [r.get("type", "RELATES") for r in rels if r]
        if not rel_types:
            return " → ".join(node_names)
        parts = [node_names[0]] if node_names else []
        for i, rel_type in enumerate(rel_types):
            parts.append(f"--[{rel_type}]-->")
            if i + 1 < len(node_names):
                parts.append(node_names[i + 1])
        return " ".join(parts)
