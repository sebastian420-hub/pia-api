import logging
import os
import re
import uuid
from typing import List, Optional

import asyncpg
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field

from auth import require_token
from config import DOC_DIR, EMBEDDING_MODEL, LLM_MODEL, MAX_UPLOAD_BYTES, llm_client

logger = logging.getLogger("pia-api")

# Every route in this router requires the bearer token.
router = APIRouter(dependencies=[Depends(require_token)])

DEFAULT_CLIENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")
ALLOWED_UPLOAD_EXT = {".pdf", ".txt"}


def get_pool(request: Request) -> asyncpg.Pool:
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database not connected")
    return pool


def priority_from_threat(score: Optional[float]) -> str:
    score = score or 0.0
    if score >= 0.8:
        return 'CRITICAL'
    if score >= 0.5:
        return 'HIGH'
    return 'NORMAL'


def paginate(total: int, page: int, limit: int) -> dict:
    return {
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": max(1, (total + limit - 1) // limit),
    }


# ═══════════════════════════════════════════════════════════
# CHAT / CO-PILOT
# ═══════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: List[ChatMessage] = Field(default_factory=list, max_length=30)


@router.post("/chat")
async def chat_endpoint(body: ChatRequest, pool: asyncpg.Pool = Depends(get_pool)):
    """AI Co-Pilot endpoint for tactical interrogation of the Knowledge Graph."""
    async with pool.acquire() as conn:
        recent_intel = await conn.fetch("""
            SELECT content_headline, content_summary, entities, priority
            FROM intelligence_records
            WHERE created_at > NOW() - INTERVAL '7 days'
            ORDER BY created_at DESC
            LIMIT 10;
        """)

    intel_context = "\n".join(
        f"- [{r['priority']}] {r['content_headline']}: {r['content_summary'] or ''}" for r in recent_intel
    )

    system_prompt = f"""
    You are the Tactical AI Co-Pilot of the Personal Intelligence Agency (PIA).
    The user (Director) is interrogating you via the Live Dashboard.

    Provide concise, tactical intelligence summaries based strictly on the provided context.
    Do not invent facts. If the answer is not in the context, say the data is unavailable.
    Keep responses under 3 paragraphs. Use bullet points for readability.

    CURRENT RECENT INTELLIGENCE CONTEXT:
    {intel_context}
    """

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(m.model_dump() for m in body.history)
    messages.append({"role": "user", "content": body.message})

    try:
        response = await llm_client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.2)
    except Exception as e:
        logger.error("LLM chat failed: %s", e)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Language model unavailable")

    return {"status": "success", "reply": response.choices[0].message.content}


# ═══════════════════════════════════════════════════════════
# CLUSTERS / EVENTS / LOGS / ARCHIVE
# ═══════════════════════════════════════════════════════════

@router.get("/clusters/active")
async def get_active_clusters(pool: asyncpg.Pool = Depends(get_pool)):
    """Fetches currently active intelligence clusters for the map."""
    async with pool.acquire() as conn:
        try:
            records = await conn.fetch("""
                SELECT cluster_id, title as name, status, confidence, priority, domain,
                       ST_Y(geo_centroid) as lat, ST_X(geo_centroid) as lon
                FROM intelligence_clusters
                WHERE status = 'ACTIVE'
                ORDER BY updated_at DESC
                LIMIT 50;
            """)
        except asyncpg.exceptions.UndefinedTableError:
            return {"status": "success", "data": [], "message": "No clusters table yet."}
    return {"status": "success", "data": [dict(r) for r in records]}


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Receives a PDF/TXT and saves it where the document_agent will pick it up."""
    original = os.path.basename(file.filename or "")
    ext = os.path.splitext(original)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Only .pdf and .txt files are accepted")

    # Never trust the client's filename as a path: sanitise and prefix a random id.
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.splitext(original)[0])[:100] or "document"
    safe_name = f"{uuid.uuid4().hex}_{stem}{ext}"
    os.makedirs(DOC_DIR, exist_ok=True)
    destination = os.path.abspath(os.path.join(DOC_DIR, safe_name))
    if os.path.commonpath([destination, DOC_DIR]) != DOC_DIR:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid filename")

    written = 0
    try:
        with open(destination, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                        f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                out.write(chunk)
    except HTTPException:
        if os.path.exists(destination):
            os.remove(destination)
        raise
    finally:
        await file.close()

    logger.info("Document upload stored: %s (%d bytes) from %r", safe_name, written, original)
    return {"status": "success", "message": f"File '{original}' queued for ingestion.", "stored_as": safe_name}


@router.get("/event/{uid}")
async def get_event_details(uid: uuid.UUID, pool: asyncpg.Pool = Depends(get_pool)):
    """Fetches the AI summary and extracted entities for one intelligence record."""
    async with pool.acquire() as conn:
        record = await conn.fetchrow(
            "SELECT content_summary, entities FROM intelligence_records WHERE uid = $1", uid
        )
    if not record:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Record not found")
    return {
        "status": "success",
        "data": {
            "summary": record['content_summary'] or "No AI summary available.",
            "entities": record['entities'] or [],
        },
    }


@router.get("/logs")
async def get_system_logs(pool: asyncpg.Pool = Depends(get_pool)):
    """Latest agent activity across the queue and the ingestors, for the UI terminal."""
    async with pool.acquire() as conn:
        records = await conn.fetch("""
            SELECT created_at, agent, action, message, status
            FROM (
                SELECT created_at,
                       COALESCE(assigned_agent, 'SYSTEM') as agent,
                       trigger_type as action,
                       COALESCE(error_message, 'Job ID: ' || queue_id::text) as message,
                       status
                FROM analysis_queue
                UNION ALL
                SELECT created_at,
                       source_agent as agent,
                       'INGEST_' || source_type as action,
                       content_headline as message,
                       'DONE' as status
                FROM intelligence_records
            ) combined_logs
            ORDER BY created_at DESC
            LIMIT 30;
        """)

    logs = []
    for r in records:
        time_str = r['created_at'].strftime("%H:%M:%S")
        if r['status'] == 'FAILED':
            logs.append(f"[{time_str}] [{r['agent']}] ERROR: {r['message']}")
        elif r['status'] == 'PROCESSING':
            logs.append(f"[{time_str}] [{r['agent']}] PROCESSING: {r['action']}")
        else:
            logs.append(f"[{time_str}] [{r['agent']}] {r['action']}: {r['message']}")
    return {"status": "success", "data": logs}


@router.get("/archive")
async def get_intelligence_archive(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    pool: asyncpg.Pool = Depends(get_pool),
):
    """Paginated historical records. Includes lat/lon so the globe can show history on load."""
    offset = (page - 1) * limit
    async with pool.acquire() as conn:
        records = await conn.fetch("""
            SELECT uid, created_at, source_type, priority, domain, content_headline, content_summary, entities,
                   ST_Y(geo) as lat, ST_X(geo) as lon
            FROM intelligence_records
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2;
        """, limit, offset)
        total = await conn.fetchval("SELECT count(*) FROM intelligence_records;")

    data = []
    for r in records:
        row = dict(r)
        lat, lon = row.pop('lat'), row.pop('lon')
        row['geo'] = {"lat": lat, "lon": lon} if lat is not None and lon is not None else None
        data.append(row)
    return {"status": "success", "data": data, "pagination": paginate(total, page, limit)}


# ═══════════════════════════════════════════════════════════
# ENTITIES
# ═══════════════════════════════════════════════════════════

def _format_geo_entities(records) -> list:
    return [{
        "uid": str(r['uid']),
        "headline": r['headline'],
        "domain": r['domain'],
        "source_type": r['source_type'],
        "priority": priority_from_threat(r['threat_score']),
        "geo": {"lat": r['lat'], "lon": r['lon']},
    } for r in records]


@router.get("/entities/strategic")
async def get_strategic_entities(pool: asyncpg.Pool = Depends(get_pool)):
    """Watched entities with coordinates (the 'Knowledge Underlay'), globally."""
    async with pool.acquire() as conn:
        records = await conn.fetch("""
            SELECT entity_id as uid, name as headline, entity_type as domain, 'KNOWLEDGE' as source_type,
                   threat_score, ST_Y(primary_geo) as lat, ST_X(primary_geo) as lon
            FROM entities
            WHERE primary_geo IS NOT NULL AND watch_status != 'PASSIVE'
            ORDER BY threat_score DESC
            LIMIT 500;
        """)
    return {"status": "success", "data": _format_geo_entities(records)}


@router.get("/entities/bbox")
async def get_entities_by_bbox(
    minLat: float = Query(..., ge=-90, le=90),
    minLon: float = Query(..., ge=-180, le=180),
    maxLat: float = Query(..., ge=-90, le=90),
    maxLon: float = Query(..., ge=-180, le=540),
    pool: asyncpg.Pool = Depends(get_pool),
):
    """
    Watched entities inside the viewport. The UI sends maxLon > 180 when the view
    crosses the antimeridian; that case is split into two envelopes.
    """
    if maxLat < minLat:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "maxLat must be >= minLat")

    if maxLon > 180:
        envelopes = [(minLon, minLat, 180.0, maxLat), (-180.0, minLat, maxLon - 360.0, maxLat)]
    elif maxLon < minLon:
        envelopes = [(minLon, minLat, 180.0, maxLat), (-180.0, minLat, maxLon, maxLat)]
    else:
        envelopes = [(minLon, minLat, maxLon, maxLat)]

    where = " OR ".join(
        f"primary_geo && ST_MakeEnvelope(${i*4+1}, ${i*4+2}, ${i*4+3}, ${i*4+4}, 4326)" for i in range(len(envelopes))
    )
    args = [v for env in envelopes for v in env]
    async with pool.acquire() as conn:
        records = await conn.fetch(f"""
            SELECT entity_id as uid, name as headline, entity_type as domain, 'KNOWLEDGE' as source_type,
                   threat_score, ST_Y(primary_geo) as lat, ST_X(primary_geo) as lon
            FROM entities
            WHERE primary_geo IS NOT NULL AND watch_status != 'PASSIVE' AND ({where})
            ORDER BY threat_score DESC
            LIMIT 500;
        """, *args)
    return {"status": "success", "data": _format_geo_entities(records)}


@router.get("/entities")
async def get_entities_directory(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    pool: asyncpg.Pool = Depends(get_pool),
):
    """Paginated entities for the Entity Directory."""
    offset = (page - 1) * limit
    async with pool.acquire() as conn:
        records = await conn.fetch("""
            SELECT entity_id as uid, entity_type as domain, name as headline,
                   description as content_summary, confidence, threat_score, watch_status
            FROM entities
            WHERE name IS NOT NULL AND description IS NOT NULL
            ORDER BY watch_status DESC, threat_score DESC, mention_count DESC
            LIMIT $1 OFFSET $2;
        """, limit, offset)
        total = await conn.fetchval(
            "SELECT count(*) FROM entities WHERE name IS NOT NULL AND description IS NOT NULL;"
        )

    results = [{
        "uid": str(r['uid']),
        "created_at": None,
        "source_type": "KNOWLEDGE",
        "priority": priority_from_threat(r['threat_score']),
        "domain": r['domain'],
        "content_headline": r['headline'],
        "content_summary": r['content_summary'],
    } for r in records]
    return {"status": "success", "data": results, "pagination": paginate(total, page, limit)}


# ═══════════════════════════════════════════════════════════
# SEMANTIC SEARCH
# ═══════════════════════════════════════════════════════════

class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    target: str = Field('uir', pattern="^(uir|entities)$")
    limit: int = Field(20, ge=1, le=100)


@router.post("/search/semantic")
async def semantic_search(body: SemanticSearchRequest, pool: asyncpg.Pool = Depends(get_pool)):
    """Vector search across intelligence records or entities."""
    try:
        response = await llm_client.embeddings.create(model=EMBEDDING_MODEL, input=body.query)
    except Exception as e:
        logger.error("Embedding request failed (%s): %s", EMBEDDING_MODEL, e)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Embedding model unavailable")
    embedding_str = "[" + ",".join(map(str, response.data[0].embedding)) + "]"

    async with pool.acquire() as conn:
        if body.target == 'uir':
            rows = await conn.fetch("""
                SELECT uid, created_at, source_type, priority, domain,
                       content_headline, content_summary, entities,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM intelligence_records
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $2;
            """, embedding_str, body.limit)
            return {"status": "success", "data": [dict(r) for r in rows]}

        rows = await conn.fetch("""
            SELECT entity_id as uid, entity_type as domain, name as headline,
                   description as content_summary, confidence, threat_score, watch_status,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM entities
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $2;
        """, embedding_str, body.limit)

    results = [{
        "uid": str(r['uid']),
        "created_at": None,
        "source_type": "KNOWLEDGE",
        "priority": priority_from_threat(r['threat_score']),
        "domain": r['domain'],
        "content_headline": r['headline'],
        "content_summary": r['content_summary'],
        "similarity": r['similarity'],
    } for r in rows]
    return {"status": "success", "data": results}


# ═══════════════════════════════════════════════════════════
# GRAPH
# ═══════════════════════════════════════════════════════════

@router.get("/graph/network/{entity_name}")
async def get_entity_network(
    entity_name: str,
    hops: int = Query(2, ge=1, le=3),
    pool: asyncpg.Pool = Depends(get_pool),
):
    """Relational network for an entity, direction-agnostic, one edge per node pair."""
    async with pool.acquire() as conn:
        root = await conn.fetchrow(
            "SELECT entity_id, name, entity_type, description FROM entities WHERE name ILIKE $1 "
            "ORDER BY mention_count DESC LIMIT 1",
            entity_name,
        )
        if not root:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Entity '{entity_name}' not found in the Knowledge Graph.")
        root_id = root['entity_id']

        edges = await conn.fetch("""
            WITH RECURSIVE raw_network AS (
                SELECT r.entity_a_id, r.entity_b_id, r.relationship_type, r.confidence, r.metadata,
                       1 as hop_level, ARRAY[r.entity_a_id, r.entity_b_id] as path
                FROM entity_relationships r
                WHERE (r.entity_a_id = $1 OR r.entity_b_id = $1) AND r.still_valid

                UNION

                SELECT r.entity_a_id, r.entity_b_id, r.relationship_type, r.confidence, r.metadata,
                       rn.hop_level + 1,
                       rn.path || CASE WHEN r.entity_a_id = ANY(rn.path) THEN r.entity_b_id ELSE r.entity_a_id END
                FROM entity_relationships r
                JOIN raw_network rn ON (r.entity_a_id = rn.entity_a_id OR r.entity_a_id = rn.entity_b_id
                                     OR r.entity_b_id = rn.entity_a_id OR r.entity_b_id = rn.entity_b_id)
                WHERE rn.hop_level < $2 AND r.still_valid
                  AND NOT (r.entity_a_id = ANY(rn.path) AND r.entity_b_id = ANY(rn.path))
            )
            SELECT LEAST(entity_a_id, entity_b_id) as node_1,
                   GREATEST(entity_a_id, entity_b_id) as node_2,
                   string_agg(DISTINCT relationship_type, ', ') as label,
                   MAX(confidence) as confidence,
                   array_agg(DISTINCT metadata->>'reasoning') FILTER (WHERE metadata->>'reasoning' IS NOT NULL) as reasonings,
                   MIN(hop_level) as min_hop
            FROM raw_network
            GROUP BY node_1, node_2
            ORDER BY min_hop
            LIMIT 500;
        """, root_id, hops)

        nodes_dict = {str(root_id): {
            "id": str(root_id), "name": root['name'], "group": root['entity_type'],
            "description": root['description'], "val": 20,
        }}
        other_ids = {e['node_1'] for e in edges} | {e['node_2'] for e in edges}
        other_ids.discard(root_id)
        if other_ids:
            for nr in await conn.fetch(
                "SELECT entity_id, name, entity_type, description FROM entities WHERE entity_id = ANY($1)",
                list(other_ids),
            ):
                nodes_dict[str(nr['entity_id'])] = {
                    "id": str(nr['entity_id']), "name": nr['name'], "group": nr['entity_type'],
                    "description": nr['description'], "val": 5,
                }

        # relationship ids let the UI send feedback on a specific edge
        rel_rows = await conn.fetch("""
            SELECT relationship_id, LEAST(entity_a_id, entity_b_id) as node_1,
                   GREATEST(entity_a_id, entity_b_id) as node_2, relationship_type
            FROM entity_relationships
            WHERE entity_a_id = ANY($1) AND entity_b_id = ANY($1) AND still_valid
        """, list(other_ids | {root_id}))
    rel_index = {}
    for rr in rel_rows:
        rel_index.setdefault((rr['node_1'], rr['node_2']), []).append(
            {"relationship_id": str(rr['relationship_id']), "type": rr['relationship_type']}
        )

    links = []
    for edge in edges:
        clean = sorted({r.strip() for r in (edge['reasonings'] or []) if r and r.strip()})
        links.append({
            "source": str(edge['node_1']),
            "target": str(edge['node_2']),
            "label": edge['label'],
            "confidence": edge['confidence'],
            "reasoning": " | ".join(clean) if clean else None,
            "relationships": rel_index.get((edge['node_1'], edge['node_2']), []),
        })

    return {"status": "success", "data": {"nodes": list(nodes_dict.values()), "links": links}}


# ═══════════════════════════════════════════════════════════
# HUMAN FEEDBACK (the ai_feedback table had no writer)
# ═══════════════════════════════════════════════════════════

class FeedbackRequest(BaseModel):
    relationship_id: uuid.UUID
    feedback_type: str = Field(pattern="^(CONFIRMED|REJECTED_HALLUCINATION|REJECTED_WRONG_PREDICATE)$")
    human_correction: Optional[str] = Field(None, max_length=2000)


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
async def submit_feedback(body: FeedbackRequest, pool: asyncpg.Pool = Depends(get_pool)):
    """
    Records a human verdict on an inferred relationship. Rejections feed the analyst's
    negative few-shot prompt and mark the edge invalid; confirmations raise its confidence.
    """
    async with pool.acquire() as conn:
        rel = await conn.fetchrow("""
            SELECT r.relationship_id, r.relationship_type, r.client_id, a.name as subject, b.name as object
            FROM entity_relationships r
            JOIN entities a ON a.entity_id = r.entity_a_id
            JOIN entities b ON b.entity_id = r.entity_b_id
            WHERE r.relationship_id = $1
        """, body.relationship_id)
        if not rel:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Relationship not found")

        async with conn.transaction():
            feedback_id = await conn.fetchval("""
                INSERT INTO ai_feedback (client_id, relationship_id, original_subject, original_predicate,
                                         original_object, feedback_type, human_correction)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING feedback_id
            """, rel['client_id'] or DEFAULT_CLIENT_ID, rel['relationship_id'], rel['subject'],
                rel['relationship_type'], rel['object'], body.feedback_type, body.human_correction)

            if body.feedback_type == 'CONFIRMED':
                await conn.execute(
                    "UPDATE entity_relationships SET confidence = LEAST(confidence + 0.2, 0.99), updated_at = NOW() WHERE relationship_id = $1",
                    rel['relationship_id'])
            else:
                await conn.execute(
                    "UPDATE entity_relationships SET still_valid = FALSE, updated_at = NOW() WHERE relationship_id = $1",
                    rel['relationship_id'])

    return {"status": "success", "data": {"feedback_id": str(feedback_id)}}
