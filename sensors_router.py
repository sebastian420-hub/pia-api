"""
Sensor layers (cameras), snapshot proxy, agent health, and live sessions.

The snapshot proxy is the only way the browser gets camera images: it hides upstream
URLs, avoids CORS problems, caches for the camera's refresh interval, and caps size.
"""
import asyncio
import logging
import os
import shlex
import subprocess
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Optional

import asyncpg
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from auth import require_token
from routers import get_pool

logger = logging.getLogger("pia-api.sensors")
router = APIRouter(dependencies=[Depends(require_token)])

RELAY_URL = os.getenv("RELAY_URL", "").rstrip("/")
RELAY_TOKEN = os.getenv("RELAY_TOKEN", "")
RELAY_START_CMD = os.getenv("RELAY_START_CMD", "")
RELAY_STOP_CMD = os.getenv("RELAY_STOP_CMD", "")
LIVE_IDLE_MINUTES = int(os.getenv("LIVE_IDLE_MINUTES", "15"))
LIVE_COST_PER_HOUR = float(os.getenv("LIVE_COST_PER_HOUR_USD", "0.01"))
SNAPSHOT_MAX_BYTES = int(os.getenv("SNAPSHOT_MAX_BYTES", str(2 * 1024 * 1024)))
SNAPSHOT_CACHE_ITEMS = int(os.getenv("SNAPSHOT_CACHE_ITEMS", "500"))
UPSTREAM_CONCURRENCY = int(os.getenv("SNAPSHOT_UPSTREAM_CONCURRENCY", "8"))

_http = httpx.AsyncClient(timeout=15, follow_redirects=True, headers={"User-Agent": "PIA-snapshot-proxy/1.0"})
_sem = asyncio.Semaphore(UPSTREAM_CONCURRENCY)
_cache: "OrderedDict[str, tuple[float, bytes, str]]" = OrderedDict()   # sensor_id -> (expires, body, content_type)
_inflight: dict = {}


# ═══════════════════════════════════════════════════════════
# LAYERS
# ═══════════════════════════════════════════════════════════

@router.get("/layers")
async def get_layers(pool: asyncpg.Pool = Depends(get_pool)):
    """All layers with counts. Reports/entities/situations are the existing tables; cameras come from sensors."""
    async with pool.acquire() as conn:
        live = await _active_session(conn)
        rows = await conn.fetch("""
            SELECT 'reports'    AS layer_id, 'Reports'    AS label, count(*) AS count FROM intelligence_records WHERE created_at > NOW() - INTERVAL '24 hours' AND COALESCE(metadata->>'skip_analysis','false') <> 'true'
            UNION ALL SELECT 'entities',   'Entities',   count(*) FROM entities WHERE resolution = 'RESOLVED' AND origin <> 'geonames' AND mention_count > 0
            UNION ALL SELECT 'events',     'Events',     count(*) FROM events WHERE event_time > NOW() - INTERVAL '24 hours'
            UNION ALL SELECT 'situations', 'Situations', count(*) FROM intelligence_clusters WHERE status = 'ACTIVE'
            UNION ALL SELECT 'cameras',    'Cameras',    count(*) FROM sensors
                      WHERE layer_id = 'cameras' AND status <> 'OFFLINE' AND (NOT requires_relay OR $1)
        """, live is not None)
    return {"status": "success", "data": [dict(r) for r in rows], "live_session": _session_payload(live)}


# ═══════════════════════════════════════════════════════════
# SENSORS
# ═══════════════════════════════════════════════════════════

def _envelopes(minLon, minLat, maxLon, maxLat):
    if maxLon > 180:
        return [(minLon, minLat, 180.0, maxLat), (-180.0, minLat, maxLon - 360.0, maxLat)]
    if maxLon < minLon:
        return [(minLon, minLat, 180.0, maxLat), (-180.0, minLat, maxLon, maxLat)]
    return [(minLon, minLat, maxLon, maxLat)]


@router.get("/sensors")
async def list_sensors(
    layer: str = Query("cameras", pattern="^[a-z_]+$"),
    minLat: Optional[float] = Query(None, ge=-90, le=90),
    minLon: Optional[float] = Query(None, ge=-180, le=180),
    maxLat: Optional[float] = Query(None, ge=-90, le=90),
    maxLon: Optional[float] = Query(None, ge=-180, le=540),
    near_lat: Optional[float] = Query(None, ge=-90, le=90),
    near_lon: Optional[float] = Query(None, ge=-180, le=180),
    radius_km: float = Query(5, gt=0, le=200),
    limit: int = Query(2000, ge=1, le=5000),
    pool: asyncpg.Pool = Depends(get_pool),
):
    """Sensors in a bbox or near a point. ON_DEMAND (relay) sensors appear only during a live session."""
    where = ["layer_id = $1", "status <> 'OFFLINE'"]
    args: list = [layer]
    async with pool.acquire() as conn:
        live = await _active_session(conn)
        if live is None:
            where.append("NOT requires_relay")

        if near_lat is not None and near_lon is not None:
            args += [near_lon, near_lat, radius_km * 1000]
            where.append(f"ST_DWithin(geo::geography, ST_SetSRID(ST_MakePoint(${len(args)-2}, ${len(args)-1}), 4326)::geography, ${len(args)})")
            order = f"ORDER BY geo <-> ST_SetSRID(ST_MakePoint(${len(args)-2}, ${len(args)-1}), 4326)"
        elif None not in (minLat, minLon, maxLat, maxLon):
            parts = []
            for env in _envelopes(minLon, minLat, maxLon, maxLat):
                args += list(env)
                n = len(args)
                parts.append(f"geo && ST_MakeEnvelope(${n-3}, ${n-2}, ${n-1}, ${n}, 4326)")
            where.append("(" + " OR ".join(parts) + ")")
            order = "ORDER BY status = 'ONLINE' DESC"
        else:
            order = "ORDER BY status = 'ONLINE' DESC"
        args.append(limit)

        rows = await conn.fetch(f"""
            SELECT sensor_id, provider, name, city, country_code, media_kind, video_url IS NOT NULL AS has_video,
                   refresh_seconds, cost_class, requires_relay, status, last_ok,
                   ST_Y(geo) AS lat, ST_X(geo) AS lon
            FROM sensors
            WHERE {' AND '.join(where)}
            {order}
            LIMIT ${len(args)}
        """, *args)
    return {"status": "success", "data": [_sensor_row(r) for r in rows]}


def _sensor_row(r) -> dict:
    d = dict(r)
    d["sensor_id"] = str(d["sensor_id"])
    d["geo"] = {"lat": d.pop("lat"), "lon": d.pop("lon")}
    return d


@router.get("/sensors/{sensor_id}")
async def get_sensor(sensor_id: uuid.UUID, pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        r = await conn.fetchrow("""
            SELECT sensor_id, layer_id, provider, external_id, name, city, country_code, media_kind,
                   video_url IS NOT NULL AS has_video, refresh_seconds, attribution, cost_class, requires_relay,
                   status, first_seen, last_seen, last_ok, metadata, ST_Y(geo) AS lat, ST_X(geo) AS lon
            FROM sensors WHERE sensor_id = $1
        """, sensor_id)
    if not r:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sensor not found")
    d = _sensor_row(r)
    d["snapshot_url"] = f"/api/v1/sensors/{sensor_id}/snapshot"
    d["video_proxy_url"] = f"/api/v1/sensors/{sensor_id}/video" if d["has_video"] else None
    return {"status": "success", "data": d}


# ═══════════════════════════════════════════════════════════
# SNAPSHOT / VIDEO PROXY
# ═══════════════════════════════════════════════════════════

async def _resolve_upstream(conn, sensor_id: uuid.UUID, want_video: bool):
    r = await conn.fetchrow(
        "SELECT provider, external_id, media_url, media_kind, video_url, refresh_seconds, requires_relay FROM sensors WHERE sensor_id = $1",
        sensor_id)
    if not r:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sensor not found")
    if r["requires_relay"]:
        live = await _active_session(conn)
        if live is None:
            raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, "This source needs an active live session (US relay)")
        if not RELAY_URL:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "RELAY_URL is not configured")
        await conn.execute("UPDATE live_sessions SET last_used_at = NOW() WHERE session_id = $1", live["session_id"])

    url = r["video_url"] if want_video else r["media_url"]
    if not url:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such media for this sensor")
    # Singapore rotates image URLs per refresh; resolve the current one by camera id.
    if r["provider"] == "sg_lta" and not want_video:
        from pia_sg import resolve_singapore_image  # small helper, keeps this module free of provider code
        url = await resolve_singapore_image(_http, r["external_id"])
        if not url:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Upstream did not list this camera")
    return url, r["refresh_seconds"], r["requires_relay"]


async def _fetch_upstream(url: str, via_relay: bool) -> tuple[bytes, str]:
    headers = {}
    if via_relay:
        headers["Authorization"] = f"Bearer {RELAY_TOKEN}"
        req_url, params = f"{RELAY_URL}/fetch", {"url": url}
    else:
        req_url, params = url, None
    async with _sem:
        async with _http.stream("GET", req_url, params=params, headers=headers) as resp:
            if resp.status_code != 200:
                raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Upstream returned {resp.status_code}")
            ctype = resp.headers.get("content-type", "application/octet-stream").split(";")[0]
            chunks, total = [], 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > SNAPSHOT_MAX_BYTES:
                    raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Upstream image too large")
                chunks.append(chunk)
    return b"".join(chunks), ctype


@router.get("/sensors/{sensor_id}/snapshot")
async def get_snapshot(sensor_id: uuid.UUID, request: Request, pool: asyncpg.Pool = Depends(get_pool)):
    """Latest still image, cached for the sensor's refresh interval."""
    key = str(sensor_id)
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and hit[0] > now:
        _cache.move_to_end(key)
        return Response(hit[1], media_type=hit[2], headers={"Cache-Control": f"max-age={int(hit[0]-now)}", "X-Cache": "HIT"})

    # Collapse concurrent requests for the same camera into one upstream fetch.
    fut = _inflight.get(key)
    if fut is None:
        fut = asyncio.get_running_loop().create_future()
        _inflight[key] = fut
        try:
            async with pool.acquire() as conn:
                url, refresh, via_relay = await _resolve_upstream(conn, sensor_id, want_video=False)
            body, ctype = await _fetch_upstream(url, via_relay)
            if not ctype.startswith("image/"):
                raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Upstream returned {ctype}, not an image")
            ttl = max(2, min(int(refresh), 600))
            _cache[key] = (now + ttl, body, ctype)
            _cache.move_to_end(key)
            while len(_cache) > SNAPSHOT_CACHE_ITEMS:
                _cache.popitem(last=False)
            fut.set_result((body, ctype, ttl))
        except BaseException as e:
            fut.set_exception(e)
            raise
        finally:
            _inflight.pop(key, None)
    body, ctype, ttl = await fut
    return Response(body, media_type=ctype, headers={"Cache-Control": f"max-age={ttl}", "X-Cache": "MISS"})


@router.get("/sensors/{sensor_id}/video")
async def get_video(sensor_id: uuid.UUID, pool: asyncpg.Pool = Depends(get_pool)):
    """Video clip / stream pass-through (MP4 clips such as TfL; HLS playlists are returned as-is)."""
    async with pool.acquire() as conn:
        url, _, via_relay = await _resolve_upstream(conn, sensor_id, want_video=True)
    body, ctype = await _fetch_upstream(url, via_relay)
    return Response(body, media_type=ctype, headers={"Cache-Control": "max-age=30"})


# ═══════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════

@router.get("/health")
async def get_health(pool: asyncpg.Pool = Depends(get_pool)):
    """Agents (from heartbeats), queue depth, freshness — for the status bar."""
    async with pool.acquire() as conn:
        agents = await conn.fetch("""
            SELECT agent_name, agent_kind, status, last_beat,
                   EXTRACT(EPOCH FROM (NOW() - last_beat))::int AS age_seconds,
                   detail
            FROM agent_heartbeats
            WHERE last_beat > NOW() - INTERVAL '1 hour'   -- replaced containers leave old rows behind
            ORDER BY agent_kind, agent_name
        """)
        queue = await conn.fetch("SELECT status, count(*) AS n FROM analysis_queue GROUP BY status")
        last_record = await conn.fetchval("SELECT EXTRACT(EPOCH FROM (NOW() - max(created_at)))::int FROM intelligence_records")
        cams = await conn.fetchrow("SELECT count(*) FILTER (WHERE status = 'ONLINE') AS online, count(*) FILTER (WHERE status = 'OFFLINE') AS offline, count(*) AS total FROM sensors")
        live = await _active_session(conn)
    agent_rows = []
    for a in agents:
        d = dict(a)
        # an agent is 'stale' when it missed ~3 polls
        interval = ((a["detail"] or {}).get("interval_sec") if isinstance(a["detail"], dict) else None) or 60
        d["alive"] = a["age_seconds"] < max(90, interval * 3)
        d["detail"] = None
        agent_rows.append(d)
    return {"status": "success", "data": {
        "agents": agent_rows,
        "agents_alive": sum(1 for a in agent_rows if a["alive"]),
        "agents_total": len(agent_rows),
        "queue": {r["status"]: r["n"] for r in queue},
        "last_record_age_seconds": last_record,
        "cameras": dict(cams) if cams else None,
        "live_session": _session_payload(live),
        "server_time": datetime.now(timezone.utc).isoformat(),
    }}


# ═══════════════════════════════════════════════════════════
# LIVE SESSIONS (on-demand sources)
# ═══════════════════════════════════════════════════════════

class LiveStartRequest(BaseModel):
    minutes: int = Field(30, ge=5, le=240)
    note: Optional[str] = Field(None, max_length=200)


async def _active_session(conn):
    return await conn.fetchrow("""
        SELECT session_id, started_at, expires_at, last_used_at, note
        FROM live_sessions WHERE stopped_at IS NULL AND expires_at > NOW()
        ORDER BY started_at DESC LIMIT 1
    """)


def _session_payload(row):
    if not row:
        return None
    return {"session_id": str(row["session_id"]), "started_at": row["started_at"].isoformat(),
            "expires_at": row["expires_at"].isoformat(), "note": row["note"]}


def _run_hook(cmd: str, what: str):
    if not cmd:
        return
    try:
        subprocess.run(shlex.split(cmd), check=True, timeout=120, capture_output=True)
    except Exception as e:
        logger.error("relay %s hook failed: %s", what, e)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Relay {what} failed")


async def _relay_healthy() -> bool:
    if not RELAY_URL:
        return False
    try:
        r = await _http.get(f"{RELAY_URL}/healthz", timeout=8)
        return r.status_code == 200
    except Exception:
        return False


@router.get("/live")
async def live_status(pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        live = await _active_session(conn)
        used_today = await conn.fetchval("""
            SELECT COALESCE(SUM(EXTRACT(EPOCH FROM (LEAST(COALESCE(stopped_at, NOW()), expires_at, NOW()) - started_at))), 0) / 60
            FROM live_sessions WHERE started_at > date_trunc('day', NOW())
        """)
    minutes = float(used_today or 0)
    return {"status": "success", "data": {
        "active": live is not None, "session": _session_payload(live),
        "relay_configured": bool(RELAY_URL),
        "minutes_used_today": round(minutes, 1),
        "estimated_cost_today_usd": round(minutes / 60 * LIVE_COST_PER_HOUR, 4),
    }}


@router.post("/live/start", status_code=status.HTTP_201_CREATED)
async def live_start(body: LiveStartRequest, pool: asyncpg.Pool = Depends(get_pool)):
    """Starts the relay (if a start hook is configured) and opens a time-boxed session."""
    if not RELAY_URL:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No relay configured (RELAY_URL); nothing to switch on")
    async with pool.acquire() as conn:
        if await _active_session(conn):
            raise HTTPException(status.HTTP_409_CONFLICT, "A live session is already active")
    await asyncio.get_running_loop().run_in_executor(None, _run_hook, RELAY_START_CMD, "start")
    for _ in range(20):
        if await _relay_healthy():
            break
        await asyncio.sleep(3)
    else:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Relay did not become healthy")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO live_sessions (expires_at, started_by, note) VALUES (NOW() + make_interval(mins => $1), 'api', $2) RETURNING session_id, started_at, expires_at, last_used_at, note",
            body.minutes, body.note)
    logger.info("live session started for %d min", body.minutes)
    return {"status": "success", "data": _session_payload(row)}


@router.post("/live/stop")
async def live_stop(pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        live = await _active_session(conn)
        if not live:
            return {"status": "success", "data": None, "message": "No active session"}
        await conn.execute("UPDATE live_sessions SET stopped_at = NOW() WHERE session_id = $1", live["session_id"])
    await asyncio.get_running_loop().run_in_executor(None, _run_hook, RELAY_STOP_CMD, "stop")
    return {"status": "success", "data": None}


async def live_session_reaper(pool: asyncpg.Pool):
    """Background task: stops the relay when a session expires or goes idle."""
    while True:
        try:
            async with pool.acquire() as conn:
                live = await _active_session(conn)
                if live:
                    idle_cutoff = datetime.now(timezone.utc) - timedelta(minutes=LIVE_IDLE_MINUTES)
                    last_used = live["last_used_at"] or live["started_at"]
                    if last_used < idle_cutoff:
                        await conn.execute("UPDATE live_sessions SET stopped_at = NOW(), note = COALESCE(note,'') || ' [idle stop]' WHERE session_id = $1", live["session_id"])
                        _run_hook(RELAY_STOP_CMD, "stop")
                        logger.info("live session stopped: idle")
                else:
                    # expired but relay hook never ran → make sure it is off
                    expired = await conn.fetchrow("SELECT session_id FROM live_sessions WHERE stopped_at IS NULL AND expires_at <= NOW() ORDER BY expires_at DESC LIMIT 1")
                    if expired:
                        await conn.execute("UPDATE live_sessions SET stopped_at = NOW() WHERE session_id = $1", expired["session_id"])
                        _run_hook(RELAY_STOP_CMD, "stop")
                        logger.info("live session stopped: expired")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("live session reaper: %s", e)
        await asyncio.sleep(30)
