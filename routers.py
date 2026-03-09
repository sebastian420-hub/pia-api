import os
import asyncpg
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from openai import AsyncOpenAI

router = APIRouter()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "pia")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "pia")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

llm_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

class SemanticSearchRequest(BaseModel):
    query: str
    target: str = 'uir' # 'uir' or 'entities'
    limit: int = 20

@router.post("/search/semantic")
async def semantic_search(request: SemanticSearchRequest):
    """
    Performs a semantic vector search across either the UIRs or Entities.
    """
    try:
        # 1. Generate embedding for the search query
        response = await llm_client.embeddings.create(
            model="text-embedding-3-small", # Make sure this matches the model used for insertion
            input=request.query
        )
        query_embedding = response.data[0].embedding
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        conn = await asyncpg.connect(DATABASE_URL)
        
        results = []
        if request.target == 'uir':
            # Vector cosine similarity search on Intelligence Records using diskann
            sql = """
                SELECT 
                    uid, created_at, source_type, priority, domain, 
                    content_headline, content_summary, entities,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM intelligence_records
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $2;
            """
            rows = await conn.fetch(sql, embedding_str, request.limit)
            results = [dict(r) for r in rows]
            
        elif request.target == 'entities':
            # Vector cosine similarity search on Entities
            sql = """
                SELECT 
                    entity_id as uid, entity_type as domain, name as headline,
                    description as content_summary, confidence, threat_score, watch_status,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM entities
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $2;
            """
            rows = await conn.fetch(sql, embedding_str, request.limit)
            
            for r in rows:
                priority = 'NORMAL'
                if r['threat_score'] >= 0.8: priority = 'CRITICAL'
                elif r['threat_score'] >= 0.5: priority = 'HIGH'
                
                results.append({
                    "uid": str(r['uid']),
                    "created_at": None,
                    "source_type": "KNOWLEDGE",
                    "priority": priority,
                    "domain": r['domain'],
                    "content_headline": r['headline'],
                    "content_summary": r['content_summary'],
                    "similarity": r['similarity']
                })
        else:
             return {"status": "error", "message": "Invalid target specified."}

        await conn.close()
        return {"status": "success", "data": results}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/entities/bbox")
async def get_entities_by_bbox(
    minLat: float = Query(...), 
    minLon: float = Query(...), 
    maxLat: float = Query(...), 
    maxLon: float = Query(...)
):
    """
    Spatial query to fetch strategic entities within a specific map bounding box.
    Uses PostGIS ST_MakeEnvelope.
    """
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        # Construct polygon from bounding box coordinates: (minLon, minLat), (maxLon, maxLat)
        # EPSG:4326 is the standard WGS 84 spatial reference system used by GPS/Cesium
        query = """
            SELECT 
                entity_id as uid, 
                name as headline, 
                entity_type as domain, 
                'KNOWLEDGE' as source_type,
                threat_score,
                ST_Y(primary_geo) as lat, 
                ST_X(primary_geo) as lon
            FROM entities
            WHERE primary_geo IS NOT NULL
            AND watch_status != 'PASSIVE'
            ORDER BY threat_score DESC
            LIMIT 500;
        """
        records = await conn.fetch(query, minLon, minLat, maxLon, maxLat)
        await conn.close()
        
        formatted = []
        for r in records:
            priority = 'NORMAL'
            if r['threat_score'] >= 0.8: priority = 'CRITICAL'
            elif r['threat_score'] >= 0.5: priority = 'HIGH'
            
            formatted.append({
                "uid": str(r['uid']),
                "headline": r['headline'],
                "domain": r['domain'],
                "source_type": r['source_type'],
                "priority": priority,
                "geo": {"lat": r['lat'], "lon": r['lon']}
            })
            
        return {"status": "success", "data": formatted}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/entities")
async def get_entities_directory(page: int = 1, limit: int = 50):
    """Fetches paginated entities for the Entity Directory."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        offset = (page - 1) * limit
        query = """
            SELECT 
                entity_id as uid, 
                entity_type as domain, 
                name as headline,
                description as content_summary, 
                confidence, 
                threat_score, 
                watch_status
            FROM entities
            WHERE name IS NOT NULL AND description IS NOT NULL
            ORDER BY watch_status DESC, threat_score DESC, mention_count DESC
            LIMIT $1 OFFSET $2;
        """
        records = await conn.fetch(query, limit, offset)
        
        # Get total count for pagination
        count_query = "SELECT count(*) FROM entities WHERE name IS NOT NULL AND description IS NOT NULL;"
        total = await conn.fetchval(count_query)
        
        await conn.close()
        
        results = []
        for r in records:
            priority = 'NORMAL'
            if r['threat_score'] >= 0.8: priority = 'CRITICAL'
            elif r['threat_score'] >= 0.5: priority = 'HIGH'
            
            results.append({
                "uid": str(r['uid']),
                "created_at": None,
                "source_type": "KNOWLEDGE",
                "priority": priority,
                "domain": r['domain'],
                "content_headline": r['headline'],
                "content_summary": r['content_summary']
            })
            
        return {
            "status": "success",
            "data": results,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "total_pages": (total // limit) + (1 if total % limit > 0 else 0)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/graph/network/{entity_name}")
async def get_entity_network(entity_name: str, hops: int = 3):
    """Fetches the relational network for a specific entity with direction-agnostic aggregation."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)

        # 1. Find the root entity
        root = await conn.fetchrow("SELECT entity_id, name, entity_type, description FROM entities WHERE name ILIKE $1 LIMIT 1", entity_name)
        if not root:
            return {"status": "error", "message": f"Entity '{entity_name}' not found in the Knowledge Graph."}

        root_id = root['entity_id']

        # 2. Direction-agnostic multi-hop query
        query = """
            WITH RECURSIVE raw_network AS (
                -- Base case: find all individual relationships
                SELECT 
                    r.entity_a_id,
                    r.entity_b_id,
                    r.relationship_type,
                    r.confidence,
                    r.metadata,
                    1 as hop_level,
                    ARRAY[r.entity_a_id, r.entity_b_id] as path
                FROM entity_relationships r
                WHERE r.entity_a_id = $1 OR r.entity_b_id = $1
                
                UNION
                
                -- Recursive step: find more relationships
                SELECT 
                    r.entity_a_id,
                    r.entity_b_id,
                    r.relationship_type,
                    r.confidence,
                    r.metadata,
                    rn.hop_level + 1 as hop_level,
                    rn.path || CASE WHEN r.entity_a_id = ANY(rn.path) THEN r.entity_b_id ELSE r.entity_a_id END as path
                FROM entity_relationships r
                JOIN raw_network rn ON (r.entity_a_id = rn.entity_a_id OR r.entity_a_id = rn.entity_b_id OR r.entity_b_id = rn.entity_a_id OR r.entity_b_id = rn.entity_b_id)
                WHERE rn.hop_level < $2
                AND NOT (r.entity_a_id = ANY(rn.path) AND r.entity_b_id = ANY(rn.path))
            )
            -- Aggregation to strictly unify all relationships between two nodes into ONE edge
            SELECT 
                LEAST(entity_a_id, entity_b_id) as node_1,
                GREATEST(entity_a_id, entity_b_id) as node_2,
                string_agg(DISTINCT relationship_type, ', ') as label,
                MAX(confidence) as confidence,
                array_agg(DISTINCT metadata->>'reasoning') FILTER (WHERE metadata->>'reasoning' IS NOT NULL) as reasonings,
                MIN(hop_level) as min_hop
            FROM raw_network
            GROUP BY node_1, node_2
            ORDER BY min_hop
            LIMIT 500;
        """
        edges = await conn.fetch(query, root_id, hops)

        nodes_dict = {}
        links = []

        # Always add the root node
        nodes_dict[str(root_id)] = {
            "id": str(root_id),
            "name": root['name'],
            "group": root['entity_type'],
            "description": root['description'],
            "val": 20
        }

        # Collect unique node IDs to fetch their details
        other_node_ids = set()
        for edge in edges:
            other_node_ids.add(edge['node_1'])
            other_node_ids.add(edge['node_2'])
        
        if root_id in other_node_ids:
            other_node_ids.remove(root_id)

        # Fetch details for all other nodes in one batch
        if other_node_ids:
            node_records = await conn.fetch("SELECT entity_id, name, entity_type, description FROM entities WHERE entity_id = ANY($1)", list(other_node_ids))
            for nr in node_records:
                nodes_dict[str(nr['entity_id'])] = {
                    "id": str(nr['entity_id']),
                    "name": nr['name'],
                    "group": nr['entity_type'],
                    "description": nr['description'],
                    "val": 5 # Basic size for non-root
                }

        for edge in edges:
            raw_reasons = edge['reasonings'] or []
            clean_reasons = list(set([r.strip() for r in raw_reasons if r and r.strip()]))
            combined_reasoning = " | ".join(clean_reasons) if clean_reasons else None

            links.append({
                "source": str(edge['node_1']),
                "target": str(edge['node_2']),
                "label": edge['label'],
                "confidence": edge['confidence'],
                "reasoning": combined_reasoning
            })

        await conn.close()
        return {
            "status": "success", 
            "data": {
                "nodes": list(nodes_dict.values()), 
                "links": links
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
