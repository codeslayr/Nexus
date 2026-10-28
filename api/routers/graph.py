"""
GET /graph/subgraph/{entity_id} — Entity neighborhood visualization endpoint.
Used for explainability and debugging of graph traversal.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from src.models.schemas import SubgraphResponse
from src.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


@router.get(
    "/subgraph/{entity_id}",
    response_model=SubgraphResponse,
    summary="Get entity subgraph for visualization",
    description=(
        "Returns the 1-hop local neighborhood (nodes + edges) of a given entity. "
        "Used to visualize why an entity was retrieved and which document chunks it links to."
    ),
)
async def get_subgraph(entity_id: str, request: Request) -> SubgraphResponse:
    """
    entity_id can be either:
      - A canonical entity name (e.g., "Apple Inc.")
      - A Neo4j element ID (e.g., "4:abc123:0")
    """
    log.info("subgraph_requested", entity_id=entity_id)

    graph_store = request.app.state.graph_store
    nodes, edges = await graph_store.get_subgraph(entity_id)

    if not nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity '{entity_id}' not found in knowledge graph.",
        )

    return SubgraphResponse(
        entity_id=entity_id,
        nodes=nodes,
        edges=edges,
    )


@router.get(
    "/entities",
    summary="List all entities in the knowledge graph",
)
async def list_entities(
    request: Request,
    entity_type: str | None = None,
    limit: int = 100,
) -> dict:
    """Return a paginated list of canonical entities, optionally filtered by type."""
    graph_store = request.app.state.graph_store

    cypher = """
        MATCH (e:Entity)
        {where}
        RETURN e.name AS name, e.type AS type, e.description AS description,
               size(e.aliases) AS alias_count
        ORDER BY e.name
        LIMIT $limit
    """.format(
        where=f"WHERE e.type = '{entity_type}'" if entity_type else ""
    )

    rows = await graph_store.run_cypher(cypher, {"limit": limit})
    return {"entities": rows, "count": len(rows)}


@router.get(
    "/stats",
    summary="Knowledge graph statistics",
)
async def graph_stats(
    request: Request,
    document_uri: str | None = None,
) -> dict:
    """Return node/relationship counts, optionally filtered by document_uri."""
    graph_store = request.app.state.graph_store

    if document_uri and document_uri != "all":
        from src.utils.db import pg_cursor
        async with pg_cursor() as cur:
            await cur.execute(
                "SELECT chunk_id::text FROM document_chunks WHERE source_uri = %s",
                (document_uri,),
            )
            rows = await cur.fetchall()
            chunk_ids = [r[0] for r in rows]

        if not chunk_ids:
            return {"entity_count": 0, "rel_count": 0}

        cypher = """
        MATCH (n:Entity)-[r:RELATES]->(m:Entity)
        WHERE any(cid IN r.chunk_ids WHERE cid IN $chunk_ids)
        RETURN count(DISTINCT r) AS rel_count
        """
        rel_rows = await graph_store.run_cypher(cypher, {"chunk_ids": chunk_ids})
        rel_count = rel_rows[0].get("rel_count", 0) if rel_rows else 0

        async with pg_cursor() as cur:
            await cur.execute(
                """
                SELECT count(DISTINCT canonical_entity_name)
                FROM entity_chunk_links
                WHERE chunk_id = ANY(%s::uuid[])
                """,
                (chunk_ids,),
            )
            entity_count = (await cur.fetchone())[0]

        return {"entity_count": entity_count, "rel_count": rel_count}

    stats_cypher = """
        MATCH (e:Entity)
        WITH count(e) AS entity_count
        MATCH ()-[r:RELATES]->()
        RETURN entity_count, count(r) AS rel_count,
               count(DISTINCT r.type) AS rel_type_count
    """
    rows = await graph_store.run_cypher(stats_cypher)
    if not rows:
        return {"entity_count": 0, "rel_count": 0}
    return rows[0]


@router.get(
    "/entire",
    summary="Get entire graph network for interactive visualizer",
)
async def get_entire_graph(
    request: Request,
    document_uri: str | None = None,
    limit: int = 150,
) -> dict:
    """Return nodes and edges formatted for visual network graphs, optionally filtered by document_uri."""
    graph_store = request.app.state.graph_store

    if document_uri and document_uri != "all":
        from src.utils.db import pg_cursor
        async with pg_cursor() as cur:
            await cur.execute(
                "SELECT chunk_id::text FROM document_chunks WHERE source_uri = %s",
                (document_uri,),
            )
            rows = await cur.fetchall()
            chunk_ids = [r[0] for r in rows]

            if not chunk_ids:
                return {"nodes": [], "edges": []}

            await cur.execute(
                """
                SELECT DISTINCT canonical_entity_name, entity_type
                FROM entity_chunk_links
                WHERE chunk_id = ANY(%s::uuid[])
                """,
                (chunk_ids,),
            )
            entity_rows = await cur.fetchall()

        cypher = """
        MATCH (n:Entity)-[r:RELATES]->(m:Entity)
        WHERE any(cid IN r.chunk_ids WHERE cid IN $chunk_ids)
        RETURN
            collect(DISTINCT {id: n.name, label: n.name, type: n.type}) +
            collect(DISTINCT {id: m.name, label: m.name, type: m.type}) AS rel_nodes,
            collect(DISTINCT {source: n.name, target: m.name, label: r.type, chunk_ids: r.chunk_ids}) AS edges
        """
        rows = await graph_store.run_cypher(cypher, {"chunk_ids": chunk_ids})
        rel_nodes = rows[0].get("rel_nodes", []) if rows else []
        edges = rows[0].get("edges", []) if rows else []

        nodes_map = {}
        for n in rel_nodes:
            if n and n.get("id"):
                nodes_map[n["id"]] = n

        for name, etype in entity_rows:
            if name not in nodes_map:
                nodes_map[name] = {"id": name, "label": name, "type": etype}

        filtered_edges = [e for e in edges if e is not None and e.get("source") and e.get("target")]
        return {"nodes": list(nodes_map.values()), "edges": filtered_edges}

    cypher = """
    MATCH (n:Entity)
    OPTIONAL MATCH (n)-[r:RELATES]->(m:Entity)
    RETURN
        collect(DISTINCT {id: n.name, label: n.name, type: n.type}) AS nodes,
        collect(DISTINCT CASE WHEN r IS NOT NULL THEN {source: n.name, target: m.name, label: r.type, chunk_ids: r.chunk_ids} END) AS edges
    """
    rows = await graph_store.run_cypher(cypher, {"limit": limit})
    if not rows:
        return {"nodes": [], "edges": []}
    nodes = rows[0].get("nodes", [])
    raw_edges = rows[0].get("edges", [])
    edges = [e for e in raw_edges if e is not None and e.get("source") and e.get("target")]
    return {"nodes": nodes, "edges": edges}
