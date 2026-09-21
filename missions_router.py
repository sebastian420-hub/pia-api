"""
Missions: collect broadly, look narrowly.

A mission is a collection plan (countries, area, feeds, watchlist, topics, alert rules, default view).
The agents keep collecting everything; the enrichment agent scores relevance per mission; the UI asks
the feed / globe / web endpoints for one mission and gets only what matters to it. `mission_filter()`
is the one place the filter is written.
"""
import json
import uuid
from typing import List, Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from auth import require_admin, require_analyst, require_token
from routers import get_pool

router = APIRouter(dependencies=[Depends(require_token)])

RELEVANT = 0.4   # "matters to the mission": topic match or better

MISSION_COLS = """mission_id, name, description, is_active, countries, languages, feeds, sources, watchlist::text[] AS watchlist,
                  topics, alert_rules, default_view, model, created_at, updated_at,
                  ST_AsGeoJSON(area)::jsonb AS area"""


def mission_filter(mission_id: Optional[uuid.UUID], kind: str, ref: str, arg: str) -> str:
    """SQL fragment restricting `ref` (a report uid / event id / entity id column) to what the mission
    scored ≥ RELEVANT. `arg` is the $n placeholder holding mission_id; NULL means no filter."""
    # a mission with no scores yet (General, or one created a minute ago) shows everything rather than nothing
    return f"""AND ({arg}::uuid IS NULL
                 OR NOT EXISTS (SELECT 1 FROM mission_relevance mr0 WHERE mr0.mission_id = {arg})
                 OR EXISTS (SELECT 1 FROM mission_relevance mr WHERE mr.mission_id = {arg} AND mr.kind = '{kind}'
                            AND mr.ref_id = {ref} AND mr.score >= {RELEVANT}))"""


def _row(r) -> dict:
    d = dict(r)
    d["mission_id"] = str(d["mission_id"])
    for k in ("alert_rules", "default_view", "area"):
        if isinstance(d.get(k), str):
            d[k] = json.loads(d[k])
    d["is_general"] = not (d["watchlist"] or d["countries"] or d["area"] or d["topics"])
    return d


class MissionIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: Optional[str] = None
    countries: List[str] = []            # Q-ids
    languages: List[str] = ["en"]
    feeds: List[str] = []                # RSS urls read first and in full
    sources: List[str] = []              # connector source_ids
    watchlist: List[str] = []            # entity_ids
    topics: List[str] = []
    alert_rules: dict = {"watchlist_hostile": True, "watchlist_pair": True, "new_entity_in_area": 3}
    default_view: dict = {}
    model: Optional[str] = None
    bbox: Optional[List[float]] = None   # [minLon, minLat, maxLon, maxLat] → area


def _area_sql(bbox: Optional[List[float]]):
    if not bbox:
        return "NULL"
    if len(bbox) != 4:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "bbox is [minLon, minLat, maxLon, maxLat]")
    a, b, c, d = (float(x) for x in bbox)
    return f"ST_Multi(ST_MakeEnvelope({a}, {b}, {c}, {d}, 4326))"


@router.get("/missions")
async def list_missions(pool: asyncpg.Pool = Depends(get_pool)):
    """Every mission with its size (relevant items) and its last alert."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT {MISSION_COLS},
                   (SELECT COUNT(*) FROM mission_relevance r WHERE r.mission_id = m.mission_id AND r.kind = 'report' AND r.score >= {RELEVANT}) AS reports,
                   (SELECT COUNT(*) FROM mission_relevance r WHERE r.mission_id = m.mission_id AND r.kind = 'event' AND r.score >= {RELEVANT}) AS events,
                   (SELECT COUNT(*) FROM mission_relevance r WHERE r.mission_id = m.mission_id AND r.kind = 'entity' AND r.score >= {RELEVANT}) AS entities,
                   (SELECT COUNT(*) FROM mission_memory mm WHERE mm.mission_id = m.mission_id) AS alerts,
                   (SELECT MAX(created_at) FROM mission_memory mm WHERE mm.mission_id = m.mission_id) AS last_alert,
                   (SELECT ARRAY_AGG(e.name ORDER BY array_position(m.watchlist, e.entity_id)) FROM entities e WHERE e.entity_id = ANY(m.watchlist)) AS watchlist_names,
                   (SELECT ARRAY_AGG(e.name ORDER BY array_position(m.countries, e.qid)) FROM entities e WHERE e.qid = ANY(m.countries) AND e.kind = 'COUNTRY') AS country_names
            FROM missions m ORDER BY is_active DESC, name
        """)
    return {"status": "success", "data": [_row(r) for r in rows]}


@router.get("/missions/options")
async def mission_options(pool: asyncpg.Pool = Depends(get_pool)):
    """What a mission can be made of: the topics the events actually use, the connector sources, the broad feeds."""
    async with pool.acquire() as conn:
        topics = await conn.fetch("""
            SELECT topic, COUNT(*) AS n FROM events WHERE topic IS NOT NULL AND event_time > NOW() - INTERVAL '90 days'
            GROUP BY topic ORDER BY n DESC
        """)
        sources = await conn.fetch("SELECT source_id, label, kind, trust FROM sources WHERE kind NOT IN ('NEWS', 'SYSTEM') ORDER BY label")
    return {"status": "success", "data": {"topics": [dict(t) for t in topics], "sources": [dict(s) for s in sources]}}


@router.get("/missions/{mission_id}")
async def get_mission(mission_id: uuid.UUID, pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(f"SELECT {MISSION_COLS} FROM missions m WHERE mission_id = $1", mission_id)
        if not row:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such mission")
        alerts = await conn.fetch("""
            SELECT u.uid, u.created_at, u.priority, u.content_headline, u.content_summary, ST_Y(u.geo) AS lat, ST_X(u.geo) AS lon
            FROM intelligence_records u WHERE u.source_agent = 'mission_alerts' AND u.mission_id = $1
            ORDER BY u.created_at DESC LIMIT 50
        """, mission_id)
    d = _row(row)
    d["recent_alerts"] = [dict(a, uid=str(a["uid"])) for a in alerts]
    return {"status": "success", "data": d}


@router.post("/missions", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_analyst)])
async def create_mission(body: MissionIn, pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(f"""
                INSERT INTO missions (name, description, countries, languages, feeds, sources, watchlist, topics, alert_rules, default_view, model, area)
                VALUES ($1, $2, $3, $4, $5, $6, $7::uuid[], $8, $9::jsonb, $10::jsonb, $11, {_area_sql(body.bbox)})
                RETURNING {MISSION_COLS}
            """, body.name, body.description, body.countries, body.languages, body.feeds, body.sources,
                 [uuid.UUID(w) for w in body.watchlist], body.topics, json.dumps(body.alert_rules), json.dumps(body.default_view), body.model)
        except asyncpg.exceptions.UniqueViolationError:
            raise HTTPException(status.HTTP_409_CONFLICT, "a mission with this name exists")
    return {"status": "success", "data": _row(row)}


@router.put("/missions/{mission_id}", dependencies=[Depends(require_analyst)])
async def update_mission(mission_id: uuid.UUID, body: MissionIn, pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        area = _area_sql(body.bbox) if body.bbox is not None else "area"
        row = await conn.fetchrow(f"""
            UPDATE missions SET name = $2, description = $3, countries = $4, languages = $5, feeds = $6, sources = $7,
                   watchlist = $8::uuid[], topics = $9, alert_rules = $10::jsonb, default_view = $11::jsonb, model = $12,
                   area = {area}, updated_at = NOW()
            WHERE mission_id = $1 RETURNING {MISSION_COLS}
        """, mission_id, body.name, body.description, body.countries, body.languages, body.feeds, body.sources,
             [uuid.UUID(w) for w in body.watchlist], body.topics, json.dumps(body.alert_rules), json.dumps(body.default_view), body.model)
        if not row:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such mission")
    return {"status": "success", "data": _row(row)}


@router.post("/missions/{mission_id}/activate", dependencies=[Depends(require_analyst)])
async def activate_mission(mission_id: uuid.UUID, pool: asyncpg.Pool = Depends(get_pool)):
    """One mission is active at a time: it is what the agents read first and what the screens show by default."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("UPDATE missions SET is_active = FALSE WHERE is_active")
            row = await conn.fetchrow(f"UPDATE missions SET is_active = TRUE, updated_at = NOW() WHERE mission_id = $1 RETURNING {MISSION_COLS}", mission_id)
            if not row:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "no such mission")
    return {"status": "success", "data": _row(row)}


@router.delete("/missions/{mission_id}", dependencies=[Depends(require_admin)])
async def delete_mission(mission_id: uuid.UUID, pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT name, is_active FROM missions WHERE mission_id = $1", mission_id)
        if not row:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such mission")
        if row["name"] == "General":
            raise HTTPException(status.HTTP_409_CONFLICT, "General cannot be deleted")
        async with conn.transaction():
            await conn.execute("DELETE FROM missions WHERE mission_id = $1", mission_id)
            if row["is_active"]:
                await conn.execute("UPDATE missions SET is_active = TRUE WHERE name = 'General'")
    return {"status": "success"}


@router.get("/missions/{mission_id}/alerts")
async def mission_alerts(mission_id: uuid.UUID, limit: int = Query(50, ge=1, le=200), pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT u.uid, u.created_at, u.priority, u.content_headline, u.content_summary, u.metadata->>'source_report' AS source_report,
                   ST_Y(u.geo) AS lat, ST_X(u.geo) AS lon
            FROM intelligence_records u WHERE u.source_agent = 'mission_alerts' AND u.mission_id = $1
            ORDER BY u.created_at DESC LIMIT $2
        """, mission_id, limit)
    return {"status": "success", "data": [dict(r, uid=str(r["uid"])) for r in rows]}
