"""
Who is calling, what they may do, what they may see.

Tokens are per user (`api_tokens`, sha256-hashed). `PIA_API_TOKEN` from the environment is the
bootstrap admin token: on first use it creates the admin user and registers itself, so a deployment
that only ever had the one token keeps working. A missing PIA_API_TOKEN *and* no tokens in the
database fails closed (503) — a misconfigured deployment can never be open by accident.

Roles: viewer (read) < analyst (+ review, missions, uploads, verbs, feedback) < admin (+ users, tokens,
source visibility, grants, deletion, logs).

Visibility: a row is as visible as its source. `visible_sources(user)` is every public source, every
org source (any signed-in user), and the restricted sources the user holds a grant for (admins: all).
"""
import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import asyncpg
from fastapi import Depends, HTTPException, Request, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import PIA_API_TOKEN

_bearer = HTTPBearer(auto_error=False)

# <img>/<video> tags cannot send headers; these GET paths may carry ?token= instead.
_QUERY_TOKEN_SUFFIXES = ("/snapshot", "/video")

ROLE_RANK = {"viewer": 0, "analyst": 1, "admin": 2}
CACHE_TTL = 60.0


@dataclass
class User:
    user_id: str
    name: str
    role: str
    token_hash: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    return "pia_" + secrets.token_urlsafe(32)


# token hash → (User, fetched_at); a revoked token is forgotten within CACHE_TTL
_users: Dict[str, Tuple[User, float]] = {}
# user id → (source list, fetched_at)
_visible: Dict[str, Tuple[List[str], float]] = {}


def forget(user_id: Optional[str] = None):
    """Drop caches after an admin change (token revoked, grant given, visibility changed)."""
    _visible.clear()
    if user_id is None:
        _users.clear()
    else:
        for h, (u, _) in list(_users.items()):
            if u.user_id == user_id:
                _users.pop(h, None)


async def _bootstrap_admin(conn: asyncpg.Connection, h: str) -> User:
    """The environment token becomes the first admin's token (once)."""
    row = await conn.fetchrow("SELECT user_id, name, role FROM users WHERE role = 'admin' ORDER BY created_at LIMIT 1")
    if not row:
        row = await conn.fetchrow("INSERT INTO users (name, role) VALUES ('owner', 'admin') RETURNING user_id, name, role")
    await conn.execute("""
        INSERT INTO api_tokens (token_hash, user_id, label) VALUES ($1, $2, 'bootstrap (PIA_API_TOKEN)')
        ON CONFLICT (token_hash) DO NOTHING
    """, h, row["user_id"])
    return User(str(row["user_id"]), row["name"], row["role"], h)


async def user_for_token(pool: asyncpg.Pool, token: str) -> Optional[User]:
    if not token:
        return None
    h = token_hash(token)
    hit = _users.get(h)
    if hit and time.time() - hit[1] < CACHE_TTL:
        return hit[0]
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT u.user_id, u.name, u.role FROM api_tokens t JOIN users u ON u.user_id = t.user_id
            WHERE t.token_hash = $1 AND t.revoked_at IS NULL AND (t.expires_at IS NULL OR t.expires_at > NOW())
              AND u.disabled_at IS NULL
        """, h)
        if row:
            user = User(str(row["user_id"]), row["name"], row["role"], h)
            await conn.execute("UPDATE api_tokens SET last_used_at = NOW() WHERE token_hash = $1", h)
        elif PIA_API_TOKEN and secrets.compare_digest(token, PIA_API_TOKEN):
            user = await _bootstrap_admin(conn, h)
        else:
            return None
    _users[h] = (user, time.time())
    return user


def _token_from_request(request: Request, creds: Optional[HTTPAuthorizationCredentials]) -> str:
    if creds is not None and creds.scheme.lower() == "bearer":
        return creds.credentials
    if request.method == "GET" and request.url.path.endswith(_QUERY_TOKEN_SUFFIXES):
        return request.query_params.get("token", "")
    return ""


async def current_user(request: Request, creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> User:
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database not connected")
    user = await user_for_token(pool, _token_from_request(request, creds))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid bearer token", headers={"WWW-Authenticate": "Bearer"})
    request.state.user = user
    return user


async def require_token(user: User = Depends(current_user)) -> User:
    """Any signed-in user. Kept under its old name: every router depends on it."""
    return user


def require_role(role: str):
    async def dep(user: User = Depends(current_user)) -> User:
        if ROLE_RANK.get(user.role, -1) < ROLE_RANK[role]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"This needs the {role} role")
        return user
    return dep


require_analyst = require_role("analyst")
require_admin = require_role("admin")


async def ws_user(websocket: WebSocket) -> Optional[User]:
    """?token=... or an Authorization: Bearer header on the WebSocket handshake."""
    token = websocket.query_params.get("token", "")
    if not token:
        header = websocket.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = header[7:].strip()
    pool = getattr(websocket.app.state, "pool", None)
    if pool is None:
        return None
    return await user_for_token(pool, token)


# ── visibility ────────────────────────────────────────────────────────────────

async def visible_sources(pool: asyncpg.Pool, user: User) -> List[str]:
    """Source ids this user may read from. Admins: every source (None means 'no filter' to callers)."""
    hit = _visible.get(user.user_id)
    if hit and time.time() - hit[1] < CACHE_TTL:
        return hit[0]
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT s.source_id FROM sources s
            WHERE s.visibility IN ('public', 'org') OR $1
               OR EXISTS (SELECT 1 FROM source_grants g WHERE g.source_id = s.source_id AND g.user_id = $2::uuid)
        """, user.is_admin, user.user_id)
    ids = [r["source_id"] for r in rows]
    _visible[user.user_id] = (ids, time.time())
    return ids


async def restricted_sources(pool: asyncpg.Pool) -> List[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT source_id FROM sources WHERE visibility = 'restricted'")
    return [r["source_id"] for r in rows]


class Visibility:
    """What one request may see. `sql(col)` gives the WHERE fragment for a source column;
    `hidden` is the set of restricted sources this user may NOT read (usually empty → no filtering cost)."""

    def __init__(self, user: User, hidden: List[str], granted: Optional[List[str]] = None):
        self.user = user
        self.hidden = hidden
        self.granted = granted or []      # restricted sources this user MAY read: shown "on top" of the shared picture

    def sql(self, col: str) -> str:
        """`AND (col IS NULL OR col <> ALL(...))` — restricted sources the user may not read are excluded.
        Rows with no source (system rows) are visible to everyone signed in."""
        if not self.hidden:
            return ""
        lst = ", ".join("'" + s.replace("'", "''") + "'" for s in self.hidden)
        return f" AND ({col} IS NULL OR {col} NOT IN ({lst}))"

    def allows(self, source_id: Optional[str]) -> bool:
        return source_id is None or source_id not in self.hidden


async def visibility(request: Request, user: User = Depends(current_user)) -> Visibility:
    pool = request.app.state.pool
    restricted = await restricted_sources(pool)
    if user.is_admin:
        return Visibility(user, [], restricted)
    allowed = set(await visible_sources(pool, user))
    hidden = [s for s in restricted if s not in allowed]
    return Visibility(user, hidden, [s for s in restricted if s in allowed])


async def note_restricted(pool: asyncpg.Pool, vis: "Visibility", obj: str, sources) -> None:
    """Audit a read that returned rows from restricted sources the user is granted (admins included)."""
    seen = sorted({s for s in sources if s and s in vis.granted})
    if not seen:
        return
    import json
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO audit_log (user_id, action, object, detail) VALUES ($1::uuid, 'read_restricted', $2, $3::jsonb)",
                           vis.user.user_id, obj[:300], json.dumps({"sources": seen}))
