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
