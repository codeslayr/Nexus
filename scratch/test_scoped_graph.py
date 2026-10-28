import asyncio
import sys
sys.path.append(".")
from pipelines.graph_store import GraphStore
from src.utils.db import pg_cursor

async def test_scoped_graph(source_uri: str):
    gs = GraphStore()
    
    # 1. Fetch chunk_ids for source_uri
    async with pg_cursor() as cur:
        await cur.execute(
            "SELECT chunk_id::text FROM document_chunks WHERE source_uri = %s",
            (source_uri,)
        )
        chunk_rows = await cur.fetchall()
        chunk_ids = [r[0] for r in chunk_rows]
        
        # Also fetch linked entities
        await cur.execute(
            """
            SELECT DISTINCT canonical_entity_name, entity_type
            FROM entity_chunk_links
            WHERE chunk_id = ANY(%s::uuid[])
            """,
            (chunk_ids,)
        )
        entity_rows = await cur.fetchall()

    print(f"Chunks for {source_uri}: {len(chunk_ids)}")
    print(f"Entities in entity_chunk_links: {len(entity_rows)}")

    # 2. Query Neo4j for relationships matching these chunks
    cypher = """
    MATCH (n:Entity)-[r:RELATES]->(m:Entity)
    WHERE any(cid IN r.chunk_ids WHERE cid IN $chunk_ids)
    RETURN
        collect(DISTINCT {id: n.name, label: n.name, type: n.type}) +
        collect(DISTINCT {id: m.name, label: m.name, type: m.type}) AS rel_nodes,
        collect(DISTINCT {source: n.name, target: m.name, label: r.type, chunk_ids: r.chunk_ids}) AS edges
    """
    rows = await gs.run_cypher(cypher, {"chunk_ids": chunk_ids})
    
    rel_nodes = rows[0].get("rel_nodes", []) if rows else []
    edges = rows[0].get("edges", []) if rows else []
    
    # Merge rel_nodes with all entity_rows from this doc
    nodes_map = {}
    for n in rel_nodes:
        if n and n.get("id"):
            nodes_map[n["id"]] = n
            
    for name, etype in entity_rows:
        if name not in nodes_map:
            nodes_map[name] = {"id": name, "label": name, "type": etype}
            
    print(f"Final scoped graph has {len(nodes_map)} nodes and {len(edges)} edges.")
    print("Sample nodes:", list(nodes_map.keys())[:10])
    print("Edges:", edges)
    
    await gs.close()

if __name__ == "__main__":
    asyncio.run(test_scoped_graph("data/uploads/b10707f5_Healthcare_App_on_Medical_Adherence.pdf"))
