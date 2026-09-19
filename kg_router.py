"""Knowledge web endpoints: entity cards, events, neighbours, evidence, search, review queue."""
import json
import uuid
from datetime import datetime
from typing import Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from auth import require_token
from routers import get_pool

router = APIRouter(dependencies=[Depends(require_token)])

ENTITY_COLS = "entity_id, qid, kind, name, description, resolution, origin, country_qid, sitelinks, mention_count, first_seen, last_seen, watch_status, threat_score, ST_Y(primary_geo) AS lat, ST_X(primary_geo) AS lon"


def _entity(r) -> dict:
    d = dict(r)
    d["entity_id"] = str(d["entity_id"])
    lat, lon = d.pop("lat", None), d.pop("lon", None)
    d["geo"] = {"lat": lat, "lon": lon} if lat is not None and lon is not None else None
    return d


async def _find(conn, key: str):
    """Accepts an entity uuid, a Q-id, or a name/alias."""
    try:
        uid = uuid.UUID(key)
        return await conn.fetchrow(f"SELECT {ENTITY_COLS} FROM entities WHERE entity_id = $1", uid)
    except ValueError:
        pass
    if key.upper().startswith("Q") and key[1:].isdigit():
        row = await conn.fetchrow(f"SELECT {ENTITY_COLS} FROM entities WHERE qid = $1", key.upper())
        if row:
            return row
    return await conn.fetchrow(f"""
        SELECT {ENTITY_COLS} FROM entities e
        WHERE e.resolution = 'RESOLVED' AND e.entity_id IN (SELECT entity_id FROM entity_aliases WHERE alias_norm = lower($1))
        ORDER BY e.mention_count DESC, e.sitelinks DESC LIMIT 1
    """, key)


@router.get("/kg/entities/{key}")
async def entity_card(key: str, pool: asyncpg.Pool = Depends(get_pool)):
    """The 'who is who' card: identity, trend, relations grouped by kind, recent reports."""
    async with pool.acquire() as conn:
        e = await _find(conn, key)
        if not e:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Entity not found")
        eid = e['entity_id']
        aliases = await conn.fetch("SELECT alias, source FROM entity_aliases WHERE entity_id = $1 ORDER BY alias LIMIT 40", eid)
        trend = await conn.fetchrow("""
            SELECT count(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') AS last_7d,
                   count(*) FILTER (WHERE created_at BETWEEN NOW() - INTERVAL '14 days' AND NOW() - INTERVAL '7 days') AS prev_7d
            FROM mentions WHERE entity_id = $1
        """, eid)
        rels = await conn.fetch("""
            SELECT r.kind, r.source, r.label, r.event_count, r.weight, r.first_seen, r.last_seen, r.directed,
                   (r.a_id = $1) AS outgoing,
                   o.entity_id AS other_id, o.qid AS other_qid, o.name AS other_name, o.kind AS other_kind
            FROM relations r JOIN entities o ON o.entity_id = CASE WHEN r.a_id = $1 THEN r.b_id ELSE r.a_id END
            WHERE (r.a_id = $1 OR r.b_id = $1) ORDER BY r.weight DESC LIMIT 200
        """, eid)
        reports = await conn.fetch("""
            SELECT m.report_uid, m.role, m.surface, u.content_headline, u.created_at, u.source_id, u.priority
            FROM mentions m JOIN intelligence_records u ON u.uid = m.report_uid
            WHERE m.entity_id = $1 ORDER BY u.created_at DESC LIMIT 20
        """, eid)
        event_counts = await conn.fetch("""
            SELECT action, count(*) AS n FROM events WHERE (actor_id = $1 OR target_id = $1)
              AND event_time > NOW() - INTERVAL '90 days' GROUP BY action ORDER BY n DESC
        """, eid)
        pair_sources = await conn.fetch("""
            SELECT CASE WHEN actor_id = $1 THEN target_id ELSE actor_id END AS other_id,
                   CASE action
                     WHEN 'ATTACK' THEN 'HOSTILE' WHEN 'THREATEN' THEN 'HOSTILE' WHEN 'SANCTION' THEN 'HOSTILE'
                     WHEN 'COERCE' THEN 'HOSTILE' WHEN 'ACCUSE' THEN 'HOSTILE' WHEN 'ARREST' THEN 'HOSTILE'
                     WHEN 'PROTEST' THEN 'HOSTILE' WHEN 'REJECT' THEN 'HOSTILE'
                     WHEN 'COOPERATE' THEN 'COOPERATIVE' WHEN 'AID' THEN 'COOPERATIVE' WHEN 'AGREE' THEN 'COOPERATIVE'
                     WHEN 'MEET' THEN 'COOPERATIVE' WHEN 'VISIT' THEN 'COOPERATIVE' WHEN 'APPEAL' THEN 'COOPERATIVE'
                     WHEN 'APPOINT' THEN 'ROLE' WHEN 'RESIGN' THEN 'ROLE' WHEN 'ELECT' THEN 'ROLE'
                     WHEN 'ACQUIRE' THEN 'OWNERSHIP' WHEN 'INVEST' THEN 'OWNERSHIP' ELSE NULL END AS kind,
                   array_agg(DISTINCT COALESCE(source_id, origin)) AS srcs,
                   array_agg(DISTINCT action) AS actions
            FROM events WHERE (actor_id = $1 OR target_id = $1) AND actor_id IS NOT NULL AND target_id IS NOT NULL
            GROUP BY 1, 2
        """, eid)
    srcs_by_pair = {(r['other_id'], r['kind']): {"sources": list(r['srcs']), "actions": list(r['actions'])} for r in pair_sources}
    grouped: dict = {}
    for r in rels:
        extra = srcs_by_pair.get((r['other_id'], r['kind']), {}) if r['source'] == 'events' else {}
        grouped.setdefault(r['kind'], []).append({
            "sources": extra.get("sources", []), "actions": extra.get("actions", []),
            "entity_id": str(r['other_id']), "qid": r['other_qid'], "name": r['other_name'], "kind": r['other_kind'],
            "label": r['label'], "source": r['source'], "event_count": r['event_count'], "weight": round(float(r['weight']), 3),
            "first_seen": r['first_seen'], "last_seen": r['last_seen'], "direction": ("out" if r['outgoing'] else "in") if r['directed'] else None,
        })
    return {"status": "success", "data": {
        **_entity(e),
        "aliases": [dict(a) for a in aliases],
        "trend": dict(trend) if trend else None,
        "relations": grouped,
        "event_counts": {r['action']: r['n'] for r in event_counts},
        "recent_reports": [dict(r, report_uid=str(r['report_uid'])) for r in reports],
        "wikidata_url": f"https://www.wikidata.org/wiki/{e['qid']}" if e['qid'] else None,
    }}


@router.get("/kg/entities/{key}/events")
async def entity_events(key: str, from_: Optional[datetime] = Query(None, alias="from"), to: Optional[datetime] = None,
                        action: Optional[str] = None, limit: int = Query(100, ge=1, le=500), pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        e = await _find(conn, key)
        if not e:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Entity not found")
        rows = await conn.fetch("""
            SELECT ev.event_id, ev.event_time, ev.time_precision, ev.action, ev.confidence, ev.tone, ev.quote, ev.origin, ev.source_id,
                   ev.report_uid, u.content_headline,
                   a.name AS actor, a.qid AS actor_qid, t.name AS target, t.qid AS target_qid, l.name AS location,
                   ST_Y(ev.geo) AS lat, ST_X(ev.geo) AS lon
            FROM events ev
            LEFT JOIN entities a ON a.entity_id = ev.actor_id
            LEFT JOIN entities t ON t.entity_id = ev.target_id
            LEFT JOIN entities l ON l.entity_id = ev.location_id
            LEFT JOIN intelligence_records u ON u.uid = ev.report_uid
            WHERE (ev.actor_id = $1 OR ev.target_id = $1 OR ev.location_id = $1)
              AND ($2::timestamptz IS NULL OR ev.event_time >= $2) AND ($3::timestamptz IS NULL OR ev.event_time <= $3)
              AND ($4::text IS NULL OR ev.action = $4)
            ORDER BY ev.event_time DESC LIMIT $5
        """, e['entity_id'], from_, to, action, limit)
    return {"status": "success", "data": [_event(r) for r in rows]}


def _event(r) -> dict:
    d = dict(r)
    d["event_id"] = str(d["event_id"])
    if d.get("report_uid"):
        d["report_uid"] = str(d["report_uid"])
    lat, lon = d.pop("lat", None), d.pop("lon", None)
    d["geo"] = {"lat": lat, "lon": lon} if lat is not None and lon is not None else None
    return d


@router.get("/kg/relations/{a}/{b}")
async def relation_evidence(a: str, b: str, limit: int = Query(50, ge=1, le=200), pool: asyncpg.Pool = Depends(get_pool)):
    """Why are these two connected: the events (with quotes) and the Wikidata facts between them."""
    async with pool.acquire() as conn:
        ea, eb = await _find(conn, a), await _find(conn, b)
        if not ea or not eb:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Entity not found")
        rels = await conn.fetch("SELECT kind, source, label, event_count, weight, first_seen, last_seen FROM relations WHERE (a_id = $1 AND b_id = $2) OR (a_id = $2 AND b_id = $1)",
                                ea['entity_id'], eb['entity_id'])
        evs = await conn.fetch("""
            SELECT ev.event_id, ev.event_time, ev.action, ev.confidence, ev.quote, ev.source_id, ev.origin, ev.report_uid, u.content_headline, u.source_url,
                   a.name AS actor, t.name AS target
            FROM events ev LEFT JOIN intelligence_records u ON u.uid = ev.report_uid
            LEFT JOIN entities a ON a.entity_id = ev.actor_id LEFT JOIN entities t ON t.entity_id = ev.target_id
            WHERE (ev.actor_id = $1 AND ev.target_id = $2) OR (ev.actor_id = $2 AND ev.target_id = $1)
            ORDER BY ev.event_time DESC LIMIT $3
        """, ea['entity_id'], eb['entity_id'], limit)
        shared = await conn.fetch("""
            SELECT u.uid, u.content_headline, u.created_at, u.source_id FROM intelligence_records u
            WHERE u.uid IN (SELECT report_uid FROM mentions WHERE entity_id = $1)
              AND u.uid IN (SELECT report_uid FROM mentions WHERE entity_id = $2)
            ORDER BY u.created_at DESC LIMIT 10
        """, ea['entity_id'], eb['entity_id'])
    return {"status": "success", "data": {
        "a": _entity(ea), "b": _entity(eb),
        "relations": [dict(r) for r in rels],
        "events": [dict(e, event_id=str(e['event_id']), report_uid=str(e['report_uid']) if e['report_uid'] else None) for e in evs],
        "shared_reports": [dict(s, uid=str(s['uid'])) for s in shared],
    }}


@router.get("/kg/events")
async def list_events(from_: Optional[datetime] = Query(None, alias="from"), to: Optional[datetime] = None,
                      action: Optional[str] = None, min_confidence: float = Query(0.5, ge=0, le=1),
                      minLat: Optional[float] = None, minLon: Optional[float] = None, maxLat: Optional[float] = None, maxLon: Optional[float] = None,
                      limit: int = Query(500, ge=1, le=5000), pool: asyncpg.Pool = Depends(get_pool)):
    """Events for the globe / timeline."""
    bbox = None not in (minLat, minLon, maxLat, maxLon)
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT ev.event_id, ev.event_time, ev.action, ev.confidence, ev.tone, ev.origin, ev.source_id, ev.report_uid, ev.quote,
                   a.name AS actor, a.qid AS actor_qid, t.name AS target, t.qid AS target_qid, l.name AS location,
                   ST_Y(ev.geo) AS lat, ST_X(ev.geo) AS lon, u.content_headline
            FROM events ev
            LEFT JOIN entities a ON a.entity_id = ev.actor_id LEFT JOIN entities t ON t.entity_id = ev.target_id
            LEFT JOIN entities l ON l.entity_id = ev.location_id LEFT JOIN intelligence_records u ON u.uid = ev.report_uid
            WHERE ev.confidence >= $1
              AND ($2::timestamptz IS NULL OR ev.event_time >= $2) AND ($3::timestamptz IS NULL OR ev.event_time <= $3)
              AND ($4::text IS NULL OR ev.action = $4)
              {"AND ev.geo && ST_MakeEnvelope($6, $7, $8, $9, 4326)" if bbox else ""}
            ORDER BY ev.event_time DESC LIMIT $5
        """, min_confidence, from_, to, action, limit, *([minLon, minLat, maxLon, maxLat] if bbox else []))
    return {"status": "success", "data": [_event(r) for r in rows]}


@router.get("/kg/search")
async def search_entities(q: str = Query(..., min_length=1, max_length=200), limit: int = Query(10, ge=1, le=50), pool: asyncpg.Pool = Depends(get_pool)):
    """Alias search over the web (exact first, then fuzzy)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT DISTINCT ON (e.entity_id) {ENTITY_COLS.replace('entity_id,', 'e.entity_id,')},
                   similarity(a.alias_norm, lower($1)) AS score
            FROM entities e JOIN entity_aliases a ON a.entity_id = e.entity_id
            WHERE e.resolution = 'RESOLVED' AND e.origin <> 'geonames' AND (a.alias_norm = lower($1) OR a.alias_norm % lower($1))
            ORDER BY e.entity_id, score DESC
        """, q)
    ranked = sorted(rows, key=lambda r: (r['score'], r['mention_count'] or 0, r['sitelinks'] or 0), reverse=True)[:limit]
    return {"status": "success", "data": [dict(_entity(r), score=round(float(r['score']), 3)) for r in ranked]}


# ── review queue ────────────────────────────────────────────────────────────

import re as _re
import unicodedata as _ud


def _norm(name: str) -> str:
    """Mirror of pia.kg.normalize.normalize (kept in sync by hand; used only for cache lookups)."""
    s = _ud.normalize("NFKD", name or "")
    s = "".join(ch for ch in s if not _ud.combining(ch)).strip().lower()
    s = _re.sub(r"^(the|a|an)\s+", "", s)
    s = _re.sub(r"[’']s$", "", s)
    s = _re.sub(r"[^\w\s\-]", " ", s)
    return _re.sub(r"\s+", " ", s).strip()

_KIND_BY_CLASS = {"Q5": "PERSON", "Q6256": "COUNTRY", "Q3624078": "COUNTRY", "Q515": "PLACE", "Q43229": "ORG",
                  "Q4830453": "ORG", "Q7278": "ORG", "Q484652": "ORG", "Q17149090": "ORG", "Q11446": "VESSEL", "Q11436": "AIRCRAFT"}


async def _load_qid_minimal(conn, qid: str):
    """Creates a RESOLVED entity for a Q-id straight from Wikidata; the maintenance agent refines it later."""
    import httpx
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "PIA-api/1.0 (https://github.com/sebastian420-hub/pia)"}) as client:
        r = await client.get("https://www.wikidata.org/w/api.php", params={
            "action": "wbgetentities", "ids": qid, "props": "labels|descriptions|aliases|claims|sitelinks", "languages": "en", "format": "json"})
    ent = (r.json().get("entities") or {}).get(qid)
    if not ent or "missing" in ent:
        return None
    label = (ent.get("labels") or {}).get("en", {}).get("value") or qid
    desc = (ent.get("descriptions") or {}).get("en", {}).get("value")
    aliases = [a["value"] for a in (ent.get("aliases") or {}).get("en", [])]
    p31 = [c["mainsnak"]["datavalue"]["value"]["id"] for c in (ent.get("claims") or {}).get("P31", [])
           if c.get("mainsnak", {}).get("snaktype") == "value"]
    kind = next((_KIND_BY_CLASS[c] for c in p31 if c in _KIND_BY_CLASS), "UNKNOWN")
    eid = await conn.fetchval("""
        INSERT INTO entities (qid, kind, name, description, resolution, origin, sitelinks, wikidata_synced_at)
        VALUES ($1, $2, $3, $4, 'RESOLVED', 'wikidata', $5, '1970-01-01') RETURNING entity_id
    """, qid, kind, label, desc, len(ent.get("sitelinks") or {}))
    for a in {label, *aliases}:
        await conn.execute("INSERT INTO entity_aliases (entity_id, alias, alias_norm, source) VALUES ($1, $2, lower($2), 'wikidata') ON CONFLICT DO NOTHING", eid, a)
    return eid


class ReviewDecision(BaseModel):
    decision: str = Field(pattern="^(merge|keep|reject)$")
    qid: Optional[str] = Field(None, pattern="^Q[0-9]+$")
    note: Optional[str] = Field(None, max_length=500)


@router.get("/kg/review")
async def review_queue(limit: int = Query(50, ge=1, le=200), pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT e.entity_id, e.name, e.kind, e.mention_count, e.created_at, e.metadata,
                   (SELECT json_agg(json_build_object('report_uid', m.report_uid, 'surface', m.surface, 'headline', u.content_headline))
                    FROM (SELECT * FROM mentions WHERE entity_id = e.entity_id ORDER BY created_at DESC LIMIT 3) m
                    JOIN intelligence_records u ON u.uid = m.report_uid) AS examples
            FROM entities e WHERE e.resolution = 'NEEDS_REVIEW'
            ORDER BY e.mention_count DESC, e.created_at DESC LIMIT $1
        """, limit)
        cands = {}
        metas = {str(r['entity_id']): (json.loads(r['metadata']) if isinstance(r['metadata'], str) else (r['metadata'] or {})) for r in rows}
        for r in rows:
            for q in (metas[str(r['entity_id'])].get('candidates') or [])[:5]:
                cands.setdefault(q, None)
        if cands:
            for c in await conn.fetch("SELECT qid, name, kind, description FROM entities WHERE qid = ANY($1)", list(cands)):
                cands[c['qid']] = dict(c)
            # labels/descriptions for candidates that were never loaded live in the search cache
            cache_rows = await conn.fetch("SELECT candidates FROM resolution_cache WHERE query_norm = ANY($1)",
                                          [_norm(r['name']) for r in rows])
            for cr in cache_rows:
                for c in (json.loads(cr['candidates']) if isinstance(cr['candidates'], str) else cr['candidates']) or []:
                    q = c.get('qid')
                    if q in cands and cands[q] is None:
                        cands[q] = {"qid": q, "name": c.get('label'), "description": c.get('description'), "kind": None}
    out = []
    for r in rows:
        meta = metas[str(r['entity_id'])]
        out.append({
            "entity_id": str(r['entity_id']), "name": r['name'], "kind": r['kind'], "mentions": r['mention_count'],
            "created_at": r['created_at'], "note": meta.get('note'),
            "candidates": [cands.get(q) or {"qid": q} for q in (meta.get('candidates') or [])[:5]],
            "examples": (json.loads(r['examples']) if isinstance(r['examples'], str) else r['examples']) or [],
        })
    return {"status": "success", "data": out}


@router.post("/kg/review/{entity_id}")
async def review_decide(entity_id: uuid.UUID, body: ReviewDecision, pool: asyncpg.Pool = Depends(get_pool)):
    """merge → into the Wikidata item `qid` (loaded on demand); keep → LOCAL; reject → REJECTED."""
    async with pool.acquire() as conn:
        loser = await conn.fetchrow("SELECT entity_id, name FROM entities WHERE entity_id = $1", entity_id)
        if not loser:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Entity not found")
        if body.decision == "keep":
            await conn.execute("UPDATE entities SET resolution = 'LOCAL', updated_at = NOW() WHERE entity_id = $1", entity_id)
        elif body.decision == "reject":
            await conn.execute("UPDATE entities SET resolution = 'REJECTED', updated_at = NOW() WHERE entity_id = $1", entity_id)
            await conn.execute("DELETE FROM mentions WHERE entity_id = $1", entity_id)
        else:
            if not body.qid:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "merge needs a qid")
            keeper = await conn.fetchrow("SELECT entity_id FROM entities WHERE qid = $1", body.qid)
            if not keeper:
                keeper_id = await _load_qid_minimal(conn, body.qid)
                if not keeper_id:
                    raise HTTPException(status.HTTP_404_NOT_FOUND, f"{body.qid} not found on Wikidata")
                keeper = {"entity_id": keeper_id}
            kid = keeper['entity_id']
            async with conn.transaction():
                await conn.execute("UPDATE mentions SET entity_id = $1 WHERE entity_id = $2", kid, entity_id)
                for col in ("actor_id", "target_id", "location_id"):
                    await conn.execute(f"UPDATE events SET {col} = $1 WHERE {col} = $2", kid, entity_id)
                await conn.execute("INSERT INTO entity_aliases (entity_id, alias, alias_norm, source) VALUES ($1, $2, lower($2), 'human') ON CONFLICT DO NOTHING", kid, loser['name'])
                await conn.execute("UPDATE entities k SET mention_count = k.mention_count + l.mention_count FROM entities l WHERE k.entity_id = $1 AND l.entity_id = $2", kid, entity_id)
                await conn.execute("DELETE FROM entities WHERE entity_id = $1", entity_id)
                await conn.execute("INSERT INTO ai_feedback (entity_id, feedback_type, human_correction) VALUES ($1, 'MERGED', $2)", kid, body.note or loser['name'])
    return {"status": "success", "data": {"decision": body.decision}}
