import logging
import os
import re
import json
import uuid
from typing import List, Optional

import asyncpg
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field

from auth import require_admin, require_analyst, require_token
from config import DOC_DIR, EMBEDDING_MODEL, LLM_MODEL, MAX_UPLOAD_BYTES, llm_client

logger = logging.getLogger("pia-api")

# Every route in this router requires the bearer token.
router = APIRouter(dependencies=[Depends(require_token)])

DEFAULT_CLIENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")
ALLOWED_UPLOAD_EXT = {".pdf", ".txt", ".md", ".json"}


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


@router.post("/documents/upload", dependencies=[Depends(require_analyst)])
async def upload_document(file: UploadFile = File(...)):
    """Receives a PDF/TXT and saves it where the document_agent will pick it up."""
    original = os.path.basename(file.filename or "")
    ext = os.path.splitext(original)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Only .pdf, .txt, .md and .json files are accepted")

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


@router.get("/reports/{uid}")
async def get_report(uid: uuid.UUID, pool: asyncpg.Pool = Depends(get_pool)):
    """One report with the fields the UI needs to open it as a selection."""
    async with pool.acquire() as conn:
        r = await conn.fetchrow("""
            SELECT uid, created_at, published_at, source_type, source_id, source_url, priority, domain,
                   content_headline, content_summary, entities, body_status, ST_Y(geo) AS lat, ST_X(geo) AS lon
            FROM intelligence_records WHERE uid = $1
        """, uid)
    if not r:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
    d = dict(r)
    d["uid"] = str(d["uid"])
    lat, lon = d.pop("lat"), d.pop("lon")
    d["geo"] = {"lat": lat, "lon": lon} if lat is not None and lon is not None else None
    return {"status": "success", "data": d}


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


@router.get("/logs", dependencies=[Depends(require_admin)])
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
                WHERE COALESCE(metadata->>'skip_analysis', 'false') <> 'true'
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
    mission_id: Optional[uuid.UUID] = None,
    pool: asyncpg.Pool = Depends(get_pool),
):
    """Paginated historical records. Includes lat/lon so the globe can show history on load.
    With `mission_id`, only what that mission scored relevant (alerts always)."""
    from missions_router import mission_filter
    offset = (page - 1) * limit
    where = f"""WHERE (COALESCE(metadata->>'skip_analysis', 'false') <> 'true' OR COALESCE(metadata->>'alert', 'false') = 'true')
                  AND (source_agent <> 'mission_alerts' OR $3::uuid IS NULL OR mission_id = $3)
                  {mission_filter(mission_id, 'report', 'uid', '$3')}"""
    async with pool.acquire() as conn:
        records = await conn.fetch(f"""
            SELECT uid, created_at, source_type, priority, domain, content_headline, content_summary, entities,
                   ST_Y(geo) as lat, ST_X(geo) as lon, mission_id, COALESCE(metadata->>'alert', 'false') = 'true' AS alert
            FROM intelligence_records
            {where}
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2;
        """, limit, offset, mission_id)
        total = await conn.fetchval(f"SELECT count(*) FROM intelligence_records {where.replace('$3', '$1')}", mission_id)

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
            SELECT entity_id as uid, name as headline, kind as domain, 'KNOWLEDGE' as source_type,
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
    mission_id: Optional[uuid.UUID] = None,
    pool: asyncpg.Pool = Depends(get_pool),
):
    """
    Watched entities inside the viewport (only the mission's, when one is given). The UI sends maxLon > 180 when the view
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
    from missions_router import mission_filter
    args = [v for env in envelopes for v in env]
    async with pool.acquire() as conn:
        records = await conn.fetch(f"""
            SELECT entity_id as uid, name as headline, kind as domain, 'KNOWLEDGE' as source_type,
                   threat_score, ST_Y(primary_geo) as lat, ST_X(primary_geo) as lon
            FROM entities
            WHERE primary_geo IS NOT NULL AND watch_status != 'PASSIVE' AND ({where})
              {mission_filter(mission_id, 'entity', 'entities.entity_id', f'${len(args) + 1}')}
            ORDER BY threat_score DESC
            LIMIT 500;
        """, *args, mission_id)
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
            SELECT entity_id as uid, qid, kind as domain, name as headline,
                   description as content_summary, threat_score, watch_status, mention_count
            FROM entities
            WHERE resolution = 'RESOLVED' AND origin <> 'geonames' AND mention_count > 0
            ORDER BY mention_count DESC, sitelinks DESC
            LIMIT $1 OFFSET $2;
        """, limit, offset)
        total = await conn.fetchval(
            "SELECT count(*) FROM entities WHERE resolution = 'RESOLVED' AND origin <> 'geonames' AND mention_count > 0;"
        )

    results = [{
        "uid": str(r['uid']),
        "qid": r['qid'],
        "created_at": None,
        "source_type": "KNOWLEDGE",
        "priority": priority_from_threat(r['threat_score']),
        "domain": r['domain'],
        "content_headline": r['headline'],
        "content_summary": r['content_summary'],
        "mentions": r['mention_count'],
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
            SELECT entity_id as uid, kind as domain, name as headline,
                   description as content_summary, threat_score, watch_status,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM entities
            WHERE embedding IS NOT NULL AND resolution = 'RESOLVED'
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

def _top_topics(raw, top: int = 3):
    """relations.topics jsonb → [{"topic", "count"}], strongest first."""
    if not raw:
        return []
    d = json.loads(raw) if isinstance(raw, str) else dict(raw)
    return [{"topic": k, "count": v} for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:top]]


@router.get("/graph/network/{entity_name}")
async def get_entity_network(
    entity_name: str,
    hops: int = Query(1, ge=1, le=2),
    kinds: Optional[str] = Query(None, description="csv of HOSTILE,COOPERATIVE,ROLE,OWNERSHIP,MEMBERSHIP,LOCATED,MENTIONED_WITH"),
    sources: Optional[str] = Query(None, description="csv of events,wikidata,cooccurrence"),
    min_events: int = Query(0, ge=0),
    min_weight: float = Query(0.0, ge=0),
    limit: int = Query(60, ge=1, le=400),
    pool: asyncpg.Pool = Depends(get_pool),
):
    """
    Relations around an entity for the web view. Event-based edges rank above Wikidata facts,
    facts above co-mentions; within a rank by weight. Each link carries its source, event count
    and the outlets behind it, so the UI can draw facts dashed and write "27 attacks (bbc, gdelt)".
    """
    kind_list = [k.strip().upper() for k in kinds.split(",")] if kinds else None
    source_list = [x.strip().lower() for x in sources.split(",")] if sources else None
    async with pool.acquire() as conn:
        root = await conn.fetchrow("""
            SELECT e.entity_id, e.qid, e.name, e.kind, e.description, e.mention_count
            FROM entities e
            LEFT JOIN entity_aliases a ON a.entity_id = e.entity_id
            WHERE (a.alias_norm = lower($1) OR e.qid = $1 OR e.entity_id::text = $1) AND e.resolution = 'RESOLVED'
            ORDER BY e.mention_count DESC, e.sitelinks DESC LIMIT 1
        """, entity_name)
        if not root:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"'{entity_name}' is not in the knowledge web.")
        frontier = {root['entity_id']}
        seen = set(frontier)
        edges = []
        for _ in range(hops):
            rows = await conn.fetch("""
                SELECT a_id, b_id, kind, source, label, event_count, weight, first_seen, last_seen, topics,
                       verified_count, wire_count, verified_topics, via_source
                FROM relations
                WHERE (a_id = ANY($1) OR b_id = ANY($1))
                  AND ($2::text[] IS NULL OR kind = ANY($2))
                  AND ($3::text[] IS NULL OR source = ANY($3))
                  AND event_count >= $4 AND weight >= $5
                ORDER BY CASE source WHEN 'events' THEN 0 WHEN 'connector' THEN 1 WHEN 'wikidata' THEN 2 ELSE 3 END, weight DESC
                LIMIT $6
            """, list(frontier), kind_list, source_list, min_events, min_weight, limit)
            edges.extend(rows)
            nxt = {r['a_id'] for r in rows} | {r['b_id'] for r in rows}
            frontier = nxt - seen
            seen |= nxt
        nodes = await conn.fetch(
            "SELECT entity_id, qid, name, kind, description, mention_count FROM entities WHERE entity_id = ANY($1)", list(seen))
        # outlets behind each event-based pair, and the words of the strongest verified event per (pair, kind)
        pairs = [(e['a_id'], e['b_id']) for e in edges if e['source'] == 'events']
        outlets = {}
        whys = {}
        if pairs:
            wrows = await conn.fetch("""
                SELECT DISTINCT ON (a, b, kind) a, b, kind, predicate, action, quote, modality, verifier_verdict, actor_id
                FROM (SELECT LEAST(actor_id, target_id) AS a, GREATEST(actor_id, target_id) AS b, * FROM events
                      WHERE origin = 'llm' AND actor_id IS NOT NULL AND target_id IS NOT NULL AND kind IS NOT NULL
                        AND LEAST(actor_id, target_id) = ANY($1) AND GREATEST(actor_id, target_id) = ANY($2)) x
                ORDER BY a, b, kind, (verifier_verdict = 'yes') DESC, (modality = 'asserted') DESC, confidence DESC, event_time DESC
            """, [p[0] for p in pairs], [p[1] for p in pairs])
            whys = {(r['a'], r['b'], r['kind']): {"predicate": r['predicate'] or (r['action'] or '').lower().replace('_', ' '),
                                                  "quote": r['quote'], "modality": r['modality'], "verdict": r['verifier_verdict'],
                                                  "actor_id": str(r['actor_id'])} for r in wrows}
            rows = await conn.fetch("""
                SELECT LEAST(actor_id, target_id) AS a, GREATEST(actor_id, target_id) AS b,
                       array_agg(DISTINCT COALESCE(source_id, origin)) AS srcs
                FROM events
                WHERE actor_id IS NOT NULL AND target_id IS NOT NULL
                  AND LEAST(actor_id, target_id) = ANY($1) AND GREATEST(actor_id, target_id) = ANY($2)
                GROUP BY 1, 2
            """, [p[0] for p in pairs], [p[1] for p in pairs])
            outlets = {(r['a'], r['b']): list(r['srcs']) for r in rows}

    node_map = {str(n['entity_id']): {"id": str(n['entity_id']), "qid": n['qid'], "name": n['name'], "group": n['kind'],
                                      "description": n['description'], "mentions": n['mention_count'] or 0,
                                      "val": max(3, min(24, 3 + (n['mention_count'] or 0) ** 0.5)), "is_root": False}
                for n in nodes}
    node_map[str(root['entity_id'])].update({"is_root": True, "val": 26})
    dedup = {}
    for e in edges:
        key = (str(e['a_id']), str(e['b_id']), e['kind'], e['source'])
        if key in dedup:
            continue
        dedup[key] = {
            "source": str(e['a_id']), "target": str(e['b_id']), "kind": e['kind'], "origin": e['source'],
            "label": e['label'] or e['kind'].lower(), "confidence": round(min(1.0, float(e['weight'])), 3),
            "weight": round(float(e['weight']), 3), "event_count": e['event_count'],
            "outlets": outlets.get((e['a_id'], e['b_id']), []),
            "topics": _top_topics(e['topics']), "verified_topics": _top_topics(e['verified_topics']),
            "verified_count": e['verified_count'], "wire_count": e['wire_count'],
            "why": whys.get((e['a_id'], e['b_id'], e['kind'])),
            "via_source": e['via_source'],
            "first_seen": e['first_seen'], "last_seen": e['last_seen'], "reasoning": None,
        }
    return {"status": "success", "data": {"root": str(root['entity_id']), "nodes": list(node_map.values()), "links": list(dedup.values())}}


# ═══════════════════════════════════════════════════════════
# HUMAN FEEDBACK (the ai_feedback table had no writer)
# ═══════════════════════════════════════════════════════════

class FeedbackRequest(BaseModel):
    event_id: uuid.UUID
    feedback_type: str = Field(pattern="^(CONFIRMED|REJECTED_HALLUCINATION|REJECTED_WRONG_ACTION)$")
    human_correction: Optional[str] = Field(None, max_length=2000)


@router.post("/feedback", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_analyst)])
async def submit_feedback(body: FeedbackRequest, pool: asyncpg.Pool = Depends(get_pool)):
    """Human verdict on one extracted event (a claim). Rejections remove it from the web."""
    async with pool.acquire() as conn:
        ev = await conn.fetchrow("SELECT event_id, event_time FROM events WHERE event_id = $1", body.event_id)
        if not ev:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Event not found")
        async with conn.transaction():
            fid = await conn.fetchval("""
                INSERT INTO ai_feedback (event_id, feedback_type, human_correction) VALUES ($1, $2, $3) RETURNING feedback_id
            """, body.event_id, body.feedback_type, body.human_correction)
            if body.feedback_type == 'CONFIRMED':
                await conn.execute("UPDATE events SET confidence = LEAST(confidence + 0.2, 0.99) WHERE event_id = $1", body.event_id)
            else:
                await conn.execute("DELETE FROM events WHERE event_id = $1", body.event_id)
    return {"status": "success", "data": {"feedback_id": str(fid)}}
