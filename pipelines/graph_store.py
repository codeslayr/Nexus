"""
Neo4j Graph Store — Section 2, Step 5.

Writes canonical entities and directed edges into Neo4j via Cypher.
Nodes carry provenance back to PostgreSQL chunk_ids.
"""
from __future__ import annotations

from typing import Any, Optional

from neo4j import AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import Neo4jError

from src.config import get_settings
from src.models.schemas import Entity, KGExtractionResult, Relationship, GraphPath
from src.utils.logging import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Cypher queries
# ──────────────────────────────────────────────────────────────────────────────

# MERGE on (name, type) — prevents duplicates; SET refreshes properties
_MERGE_ENTITY_CYPHER = """
MERGE (e:Entity {name: $name, type: $type})
ON CREATE SET
    e.description = $description,
    e.aliases     = $aliases,
    e.created_at  = datetime()
ON MATCH SET
    e.description = COALESCE($description, e.description),
    e.aliases     = apoc.coll.toSet(e.aliases + $aliases),
    e.updated_at  = datetime()
RETURN elementId(e) AS element_id
"""

_MERGE_RELATIONSHIP_CYPHER = """
MATCH (src:Entity {name: $source})
MATCH (tgt:Entity {name: $target})
MERGE (src)-[r:RELATES {type: $rel_type}]->(tgt)
ON CREATE SET
    r.context        = $context,
    r.chunk_ids      = [$chunk_id],
    r.confidence     = $confidence,
    r.created_at     = datetime()
ON MATCH SET
    r.chunk_ids      = apoc.coll.toSet(r.chunk_ids + [$chunk_id]),
    r.confidence     = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
    r.updated_at     = datetime()
RETURN type(r) AS rel_type
"""

_GET_NEIGHBORHOOD_CYPHER = """
MATCH path = (seed:Entity {name: $entity_name})-[*1..$hop_depth]-(neighbor:Entity)
RETURN
    [n IN nodes(path) | {name: n.name, type: n.type, description: n.description}]       AS nodes,
    [r IN relationships(path) | {type: r.type, chunk_ids: r.chunk_ids, context: r.context}] AS rels
LIMIT $limit
"""

_GET_SUBGRAPH_CYPHER = """
MATCH (e:Entity)
WHERE elementId(e) = $element_id OR e.name = $entity_id
OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
RETURN
    collect(DISTINCT {
        id: elementId(e), name: e.name, type: e.type, description: e.description
    }) + collect(DISTINCT {
        id: elementId(neighbor), name: neighbor.name,
        type: neighbor.type, description: neighbor.description
    }) AS nodes,
    collect(DISTINCT {
        source: startNode(r).name,
        target: endNode(r).name,
        type: r.type,
        chunk_ids: r.chunk_ids
    }) AS edges
"""


# ──────────────────────────────────────────────────────────────────────────────
# Graph store
# ──────────────────────────────────────────────────────────────────────────────

class GraphStore:
    """
    Async Neo4j client for entity/relationship upsert and graph traversal.

    Lifecycle: create once per application, call `close()` on shutdown.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._driver = AsyncGraphDatabase.driver(
            self._settings.neo4j_uri,
            auth=(self._settings.neo4j_user, self._settings.neo4j_password),
            max_connection_pool_size=50,
            connection_timeout=30,
        )
        self._database = self._settings.neo4j_database

    async def close(self) -> None:
        await self._driver.close()
        log.info("neo4j_driver_closed")

    # ── Constraint / Index setup (run once on startup) ────────────────────────

    async def ensure_schema(self) -> None:
        """Create uniqueness constraints and indexes if they don't exist."""
        constraints = [
            "CREATE CONSTRAINT entity_name_type IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        ]
        async with self._driver.session(database=self._database) as session:
            for stmt in constraints + indexes:
                try:
                    await session.run(stmt)
                    log.info("neo4j_schema_applied", statement=stmt[:60])
                except Neo4jError as exc:
                    log.warning("neo4j_schema_skip", error=str(exc)[:120])
        log.info("neo4j_schema_ready")

    # ── Write ─────────────────────────────────────────────────────────────────

    async def upsert_extraction_result(
        self, result: KGExtractionResult
    ) -> dict[str, int]:
        """
        Write a full KGExtractionResult to Neo4j within a single session.
        Returns counts of entities and relationships written.
        """
        entity_count = 0
        rel_count = 0

        async with self._driver.session(database=self._database) as session:
            # 1. Upsert entities
            for entity in result.entities:
                try:
                    await session.run(
                        _MERGE_ENTITY_CYPHER,
                        name=entity.name,
                        type=entity.type,
                        description=entity.description or "",
                        aliases=entity.aliases,
                    )
                    entity_count += 1
                except Neo4jError as exc:
                    log.error("entity_upsert_failed", name=entity.name, error=str(exc))

            # 2. Upsert relationships
            for rel in result.relationships:
                try:
                    await session.run(
                        _MERGE_RELATIONSHIP_CYPHER,
                        source=rel.source,
                        target=rel.target,
                        rel_type=rel.type,
                        context=rel.context or "",
                        chunk_id=rel.context_chunk_id,
                        confidence=rel.confidence,
                    )
                    rel_count += 1
                except Neo4jError as exc:
                    log.error(
                        "relationship_upsert_failed",
                        source=rel.source,
                        target=rel.target,
                        error=str(exc),
                    )

        log.info(
            "graph_upserted",
            chunk_id=result.chunk_id,
            entities=entity_count,
            rels=rel_count,
        )
        return {"entities": entity_count, "relationships": rel_count}

    # ── Read / Traversal ──────────────────────────────────────────────────────

    async def get_neighborhood(
        self,
        entity_name: str,
        hop_depth: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return k-hop neighborhood paths as raw dicts."""
        safe_hops = min(max(int(hop_depth), 1), 5)
        cypher = f"""
        MATCH path = (seed:Entity)-[*1..{safe_hops}]-(neighbor:Entity)
        WHERE toLower(seed.name) = toLower($entity_name)
        RETURN
            [n IN nodes(path) | {{name: n.name, type: n.type, description: n.description}}] AS nodes,
            [r IN relationships(path) | {{type: r.type, chunk_ids: r.chunk_ids, context: r.context}}] AS rels
        LIMIT $limit
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                entity_name=entity_name,
                limit=limit,
            )
            records = await result.data()
        return records

    async def match_entities_in_text(self, text: str) -> list[str]:
        """
        Fast Neo4j substring matching: find existing knowledge graph entity names
        contained in the query text. Completes in ~1-5ms without any LLM call.
        """
        cypher = """
        MATCH (e:Entity)
        WHERE toLower($text) CONTAINS toLower(e.name)
        RETURN e.name AS name
        LIMIT 10
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, text=text)
            records = await result.data()
            return [r["name"] for r in records if r.get("name")]

    async def get_subgraph(
        self, entity_id: str
    ) -> tuple[list[dict], list[dict]]:
        """Return (nodes, edges) for the local 1-hop subgraph of an entity."""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                _GET_SUBGRAPH_CYPHER,
                entity_id=entity_id,
                element_id=entity_id,
            )
            record = await result.single()

        if record is None:
            return [], []

        nodes = record["nodes"]
        edges = record["edges"]
        # Deduplicate nodes by id field
        seen: set = set()
        unique_nodes = []
        for n in nodes:
            if n and n.get("id") not in seen:
                seen.add(n.get("id"))
                unique_nodes.append(n)

        return unique_nodes, edges

    async def run_cypher(
        self, cypher: str, params: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """
        Execute an arbitrary Cypher query (used by Text-to-Cypher engine).
        Results are returned as list[dict] for safe JSON serialisation.
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, parameters=params or {})
            return await result.data()

    async def prune_chunk_entities(self, chunk_ids: list[str]) -> dict[str, int]:
        """
        Garbage collection: removes references to deleted chunks from relationships.
        Deletes relationships that have no remaining chunk provenance, and prunes
        orphaned entities with degree 0.
        """
        if not chunk_ids:
            return {"pruned_relationships": 0, "pruned_entities": 0}

        cypher_prune_rels = """
        MATCH ()-[r:RELATES]->()
        WHERE any(cid IN $chunk_ids WHERE cid IN r.chunk_ids)
        SET r.chunk_ids = [cid IN r.chunk_ids WHERE NOT cid IN $chunk_ids]
        WITH r
        WHERE size(r.chunk_ids) = 0
        DELETE r
        RETURN count(r) AS deleted_rels
        """

        cypher_prune_orphans = """
        MATCH (e:Entity)
        WHERE NOT (e)--()
        DELETE e
        RETURN count(e) AS deleted_entities
        """

        async with self._driver.session(database=self._database) as session:
            res_rels = await session.run(cypher_prune_rels, chunk_ids=chunk_ids)
            data_rels = await res_rels.single()
            deleted_rels = data_rels["deleted_rels"] if data_rels else 0

            res_orphans = await session.run(cypher_prune_orphans)
            data_orphans = await res_orphans.single()
            deleted_entities = data_orphans["deleted_entities"] if data_orphans else 0

        log.info(
            "graph_pruning_complete",
            deleted_relationships=deleted_rels,
            deleted_entities=deleted_entities,
        )
        return {"pruned_relationships": deleted_rels, "pruned_entities": deleted_entities}

    async def get_community_clusters(self) -> list[dict[str, Any]]:
        """
        Identify high-connectivity communities/clusters in the graph.
        Returns clusters with core hubs, members, and connected topics.
        """
        cypher = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r:RELATES]-(neighbor:Entity)
        WITH e, count(r) AS degree, collect(DISTINCT neighbor.name)[..6] AS key_connections, collect(DISTINCT r.type)[..6] AS rel_types
        WHERE degree >= 2
        RETURN e.name AS hub, e.type AS hub_type, degree, key_connections, rel_types
        ORDER BY degree DESC
        LIMIT 10
        """
        async with self._driver.session(database=self._database) as session:
            res = await session.run(cypher)
            return await res.data()

