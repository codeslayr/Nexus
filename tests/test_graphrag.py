"""
Integration and unit tests for the GraphRAG system.

Run: pytest tests/ -v --asyncio-mode=auto
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from pipelines.chunker import DocumentChunker
from pipelines.entity_resolver import EntityResolver
from src.core.reranker import HybridReranker, reciprocal_rank_fusion
from src.core.text_to_cypher import CypherValidationError, validate_cypher
from src.models.schemas import (
    DocumentChunk,
    Entity,
    GraphPath,
    Relationship,
    VectorSearchResult,
)


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests: Document Chunker
# ──────────────────────────────────────────────────────────────────────────────

class TestDocumentChunker:
    def test_chunk_text_file(self, tmp_path):
        doc = tmp_path / "test.txt"
        doc.write_text("Hello world. " * 200)
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_document(str(doc))
        assert len(chunks) > 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.source_uri == str(doc) for c in chunks)

    def test_chunk_ids_are_deterministic(self, tmp_path):
        """Same source + index → same UUID."""
        doc = tmp_path / "test.txt"
        doc.write_text("A" * 2000)
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks1 = chunker.chunk_document(str(doc))
        chunks2 = chunker.chunk_document(str(doc))
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        assert ids1 == ids2

    def test_chunk_ids_differ_by_source(self, tmp_path):
        doc1 = tmp_path / "a.txt"
        doc2 = tmp_path / "b.txt"
        doc1.write_text("Same content")
        doc2.write_text("Same content")
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=0)
        c1 = chunker.chunk_document(str(doc1))
        c2 = chunker.chunk_document(str(doc2))
        # Different URIs → different IDs even for same text
        assert c1[0].chunk_id != c2[0].chunk_id

    def test_missing_file_raises(self):
        chunker = DocumentChunker()
        with pytest.raises(FileNotFoundError):
            chunker.chunk_document("/nonexistent/path/doc.txt")


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests: Entity Resolver
# ──────────────────────────────────────────────────────────────────────────────

class TestEntityResolver:
    def test_exact_name_deduplication(self):
        resolver = EntityResolver.__new__(EntityResolver)
        resolver._threshold = 0.92
        resolver._registry = {}
        resolver._alias_map = {}
        # Mock embeddings to avoid API calls
        resolver._embeddings = MagicMock()

        e1 = Entity(name="Apple Inc.", type="ORG")
        e2 = Entity(name="Apple Inc.", type="ORG")

        # Manually register first (bypass embedding check)
        from pipelines.entity_resolver import CanonicalEntity
        resolver._registry["apple"] = CanonicalEntity(name="Apple Inc.", type="ORG")

        c1 = resolver.resolve(e1)
        c2 = resolver.resolve(e2)
        assert c1 == c2 == "Apple Inc."
        assert len(resolver._registry) == 1

    def test_alias_resolution(self):
        resolver = EntityResolver.__new__(EntityResolver)
        resolver._threshold = 0.92
        resolver._registry = {}
        resolver._alias_map = {}
        resolver._embeddings = MagicMock()

        from pipelines.entity_resolver import CanonicalEntity
        resolver._registry["apple inc"] = CanonicalEntity(
            name="Apple Inc.", type="ORG", aliases=["AAPL"]
        )
        resolver._alias_map["aapl"] = "Apple Inc."

        e = Entity(name="AAPL", type="ORG")
        canonical = resolver.resolve(e)
        assert canonical == "Apple Inc."

    def test_legal_suffix_stripping(self):
        from pipelines.entity_resolver import _normalise
        assert _normalise("Microsoft Corporation") == "microsoft corporation"
        # After legal suffix stripping + whitespace collapse:
        assert "inc" not in _normalise("Apple Inc.")
        assert "corp" not in _normalise("Tesla Corp.")


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests: Cypher Validation
# ──────────────────────────────────────────────────────────────────────────────

class TestCypherValidation:
    def test_valid_match_query(self):
        cypher = "MATCH (e:Entity {name: 'Apple'}) RETURN e.name LIMIT 10"
        result = validate_cypher(cypher)
        assert result == cypher.strip()

    def test_rejects_delete(self):
        with pytest.raises(CypherValidationError):
            validate_cypher("MATCH (n) DELETE n")

    def test_rejects_create(self):
        with pytest.raises(CypherValidationError):
            validate_cypher("CREATE (n:Entity {name: 'Bad'})")

    def test_rejects_non_match_start(self):
        with pytest.raises(CypherValidationError):
            validate_cypher("RETURN 1")

    def test_rejects_too_long(self):
        with pytest.raises(CypherValidationError):
            validate_cypher("MATCH (n) RETURN n " + "x" * 3000)

    def test_optional_match_allowed(self):
        cypher = "OPTIONAL MATCH (e:Entity) RETURN e.name LIMIT 5"
        result = validate_cypher(cypher)
        assert "OPTIONAL MATCH" in result


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests: RRF Reranker
# ──────────────────────────────────────────────────────────────────────────────

class TestRRF:
    def test_rrf_single_list(self):
        ranked = [["a", "b", "c"]]
        scores = reciprocal_rank_fusion(ranked, k=60)
        assert scores["a"] > scores["b"] > scores["c"]

    def test_rrf_fusion_boosts_overlap(self):
        list1 = ["a", "b", "c", "d"]
        list2 = ["c", "a", "e", "f"]
        scores = reciprocal_rank_fusion([list1, list2], k=60)
        # "a" appears in both and should score higher than "b" (only in list1)
        # "c" appears first in list2 and second in list1 — should score well
        assert scores["a"] > scores["d"]  # d only in list1 at rank 4
        assert scores["c"] > scores["b"]  # c appears in both vs b only once

    def test_rrf_empty_lists(self):
        scores = reciprocal_rank_fusion([], k=60)
        assert scores == {}

    def test_reranker_merge_returns_hybrid_context(self):
        reranker = HybridReranker(use_cross_encoder=False)
        chunks = [
            VectorSearchResult(
                chunk_id=f"chunk-{i}",
                source_uri="test.pdf",
                chunk_text=f"Text {i}",
                score=1.0 - i * 0.1,
                rank=i + 1,
            )
            for i in range(5)
        ]
        paths = [
            GraphPath(
                path_id=f"path-{i}",
                nodes=[{"name": f"Entity{i}", "type": "ORG"}],
                relationships=[],
                summary=f"Entity{i} --[RELATES]--> Entity{i+1}",
                source_chunk_ids=[f"chunk-{i}"],
            )
            for i in range(3)
        ]
        result = reranker.merge_and_rerank(
            query="test query",
            vector_results=chunks,
            graph_paths=paths,
            top_n=5,
        )
        assert result.query == "test query"
        assert len(result.merged_context_text) > 0
        assert "rrf_scores" in result.retrieval_metadata


# ──────────────────────────────────────────────────────────────────────────────
# Integration tests: FastAPI endpoints (mocked dependencies)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAPIEndpoints:
    async def test_health_endpoint(self):
        from httpx import AsyncClient, ASGITransport
        from api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Mock lifespan by patching startup functions
            with patch("api.main.get_pg_pool", AsyncMock(return_value=None)), \
                 patch("api.main.GraphStore") as mock_gs:
                mock_gs.return_value.ensure_schema = AsyncMock()
                mock_gs.return_value.close = AsyncMock()
                response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    async def test_ingest_accepts_request(self):
        from httpx import AsyncClient, ASGITransport
        from api.main import app

        payload = {
            "source_uri": "/tmp/test.pdf",
            "document_title": "Test Doc",
            "metadata": {},
            "force_reingest": False,
        }

        with patch("api.routers.ingest._run_ingestion", AsyncMock()), \
             patch("api.main.get_pg_pool", AsyncMock(return_value=None)), \
             patch("api.main.GraphStore") as mock_gs:
            mock_gs.return_value.ensure_schema = AsyncMock()
            mock_gs.return_value.close = AsyncMock()

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/ingest", json=payload)

        assert response.status_code == 202
        assert "job_id" in response.json()
