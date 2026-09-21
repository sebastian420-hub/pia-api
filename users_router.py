"""
Who is who on the system: me, users, tokens, source visibility, grants, the audit log.
Admin only, except /me.
"""
import json
import uuid
from typing import List, Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from auth import User, current_user, forget, new_token, require_admin, token_hash
from routers import get_pool

router = APIRouter()


async def audit(pool: asyncpg.Pool, user: Optional[User], action: str, obj: str, detail: Optional[dict] = None):
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO audit_log (user_id, action, object, detail) VALUES ($1::uuid, $2, $3, $4::jsonb)",
                           user.user_id if user else None, action, obj[:300], json.dumps(detail or {}))


@router.get("/me")
async def me(user: User = Depends(current_user), pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        grants = await conn.fetch("SELECT source_id FROM source_grants WHERE user_id = $1::uuid", user.user_id)
        restricted = await conn.fetchval("SELECT COUNT(*) FROM sources WHERE visibility = 'restricted'")
    return {"status": "success", "data": {"user_id": user.user_id, "name": user.name, "role": user.role,
                                          "grants": [g["source_id"] for g in grants], "restricted_sources": restricted}}


# ── users ──
class UserIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    email: Optional[str] = None
    role: str = Field("viewer", pattern="^(viewer|analyst|admin)$")


@router.get("/users", dependencies=[Depends(require_admin)])
async def list_users(pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT u.user_id, u.name, u.email, u.role, u.created_at, u.disabled_at,
                   (SELECT COUNT(*) FROM api_tokens t WHERE t.user_id = u.user_id AND t.revoked_at IS NULL) AS tokens,
                   (SELECT MAX(t.last_used_at) FROM api_tokens t WHERE t.user_id = u.user_id) AS last_seen,
                   (SELECT ARRAY_AGG(g.source_id ORDER BY g.source_id) FROM source_grants g WHERE g.user_id = u.user_id) AS grants
            FROM users u ORDER BY u.created_at
        """)
    return {"status": "success", "data": [dict(r, user_id=str(r["user_id"])) for r in rows]}


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(body: UserIn, admin: User = Depends(require_admin), pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow("INSERT INTO users (name, email, role) VALUES ($1, $2, $3) RETURNING user_id, name, email, role, created_at",
                                      body.name, body.email, body.role)
        except asyncpg.exceptions.UniqueViolationError:
            raise HTTPException(status.HTTP_409_CONFLICT, "a user with this email exists")
    await audit(pool, admin, "admin", f"users/{row['user_id']}", {"created": body.name, "role": body.role})
    return {"status": "success", "data": dict(row, user_id=str(row["user_id"]))}


@router.put("/users/{user_id}")
async def update_user(user_id: uuid.UUID, body: UserIn, admin: User = Depends(require_admin), pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        row = await conn.fetchrow("UPDATE users SET name = $2, email = $3, role = $4 WHERE user_id = $1 RETURNING user_id, name, email, role",
                                  user_id, body.name, body.email, body.role)
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such user")
    forget(str(user_id))
    await audit(pool, admin, "admin", f"users/{user_id}", {"role": body.role})
    return {"status": "success", "data": dict(row, user_id=str(row["user_id"]))}


@router.post("/users/{user_id}/disable")
async def disable_user(user_id: uuid.UUID, admin: User = Depends(require_admin), pool: asyncpg.Pool = Depends(get_pool)):
    if str(user_id) == admin.user_id:
        raise HTTPException(status.HTTP_409_CONFLICT, "you cannot disable yourself")
    async with pool.acquire() as conn:
        await conn.execute("UPDATE users SET disabled_at = NOW() WHERE user_id = $1", user_id)
        await conn.execute("UPDATE api_tokens SET revoked_at = NOW() WHERE user_id = $1 AND revoked_at IS NULL", user_id)
    forget(str(user_id))
    await audit(pool, admin, "admin", f"users/{user_id}", {"disabled": True})
    return {"status": "success"}


# ── tokens ──
class TokenIn(BaseModel):
    label: Optional[str] = None
    expires_days: Optional[int] = Field(None, ge=1, le=3650)


@router.post("/users/{user_id}/tokens", status_code=status.HTTP_201_CREATED)
async def mint_token(user_id: uuid.UUID, body: TokenIn, admin: User = Depends(require_admin), pool: asyncpg.Pool = Depends(get_pool)):
    """The token is returned once, here, and never stored in clear."""
    tok = new_token()
    async with pool.acquire() as conn:
        if not await conn.fetchval("SELECT 1 FROM users WHERE user_id = $1 AND disabled_at IS NULL", user_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such (active) user")
        await conn.execute("""
            INSERT INTO api_tokens (token_hash, user_id, label, expires_at)
            VALUES ($1, $2, $3, CASE WHEN $4::int IS NULL THEN NULL ELSE NOW() + make_interval(days => $4) END)
        """, token_hash(tok), user_id, body.label, body.expires_days)
    await audit(pool, admin, "admin", f"users/{user_id}/tokens", {"label": body.label})
    return {"status": "success", "data": {"token": tok, "label": body.label, "expires_days": body.expires_days}}


@router.get("/users/{user_id}/tokens", dependencies=[Depends(require_admin)])
async def list_tokens(user_id: uuid.UUID, pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT token_hash, label, created_at, expires_at, last_used_at, revoked_at FROM api_tokens WHERE user_id = $1 ORDER BY created_at", user_id)
    return {"status": "success", "data": [dict(r, token_hash=r["token_hash"][:12] + "…") for r in rows]}


@router.delete("/tokens/{prefix}")
async def revoke_token(prefix: str, admin: User = Depends(require_admin), pool: asyncpg.Pool = Depends(get_pool)):
    """Revoke by the 12-character prefix shown in the list."""
    async with pool.acquire() as conn:
        n = await conn.fetchval("WITH d AS (UPDATE api_tokens SET revoked_at = NOW() WHERE token_hash LIKE $1 || '%' AND revoked_at IS NULL RETURNING 1) SELECT COUNT(*) FROM d", prefix[:12])
    forget()
    await audit(pool, admin, "admin", f"tokens/{prefix[:12]}", {"revoked": n})
    return {"status": "success", "data": {"revoked": n}}


# ── sources: visibility and grants ──
class VisibilityIn(BaseModel):
    visibility: str = Field(..., pattern="^(public|org|restricted)$")


@router.get("/sources", dependencies=[Depends(require_admin)])
async def list_sources(pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT s.source_id, s.label, s.kind, s.trust, s.visibility,
                   (SELECT COUNT(*) FROM intelligence_records r WHERE r.source_id = s.source_id) AS reports,
                   (SELECT COUNT(*) FROM events e WHERE e.source_id = s.source_id) AS events,
                   (SELECT ARRAY_AGG(u.name ORDER BY u.name) FROM source_grants g JOIN users u ON u.user_id = g.user_id WHERE g.source_id = s.source_id) AS granted_to
            FROM sources s ORDER BY (s.visibility = 'restricted') DESC, s.kind, s.label
        """)
    return {"status": "success", "data": [dict(r) for r in rows]}


@router.put("/sources/{source_id}/visibility")
async def set_visibility(source_id: str, body: VisibilityIn, admin: User = Depends(require_admin), pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        row = await conn.fetchrow("UPDATE sources SET visibility = $2 WHERE source_id = $1 RETURNING source_id, visibility", source_id, body.visibility)
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such source")
    forget()
    await audit(pool, admin, "admin", f"sources/{source_id}", {"visibility": body.visibility})
    return {"status": "success", "data": dict(row)}


class GrantIn(BaseModel):
    user_id: str


@router.post("/sources/{source_id}/grants", status_code=status.HTTP_201_CREATED)
async def grant(source_id: str, body: GrantIn, admin: User = Depends(require_admin), pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        try:
            await conn.execute("INSERT INTO source_grants (source_id, user_id, granted_by) VALUES ($1, $2::uuid, $3::uuid) ON CONFLICT DO NOTHING",
                               source_id, body.user_id, admin.user_id)
        except asyncpg.exceptions.ForeignKeyViolationError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such source or user")
    forget(body.user_id)
    await audit(pool, admin, "admin", f"sources/{source_id}/grants", {"user_id": body.user_id, "granted": True})
    return {"status": "success"}


@router.delete("/sources/{source_id}/grants/{user_id}")
async def revoke_grant(source_id: str, user_id: uuid.UUID, admin: User = Depends(require_admin), pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM source_grants WHERE source_id = $1 AND user_id = $2", source_id, user_id)
    forget(str(user_id))
    await audit(pool, admin, "admin", f"sources/{source_id}/grants", {"user_id": str(user_id), "granted": False})
    return {"status": "success"}


# ── audit ──
@router.get("/audit", dependencies=[Depends(require_admin)])
async def audit_log(limit: int = Query(100, ge=1, le=1000), action: Optional[str] = None, pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT a.id, a.at, a.user_id, u.name AS user_name, a.action, a.object, a.detail FROM audit_log a
            LEFT JOIN users u ON u.user_id = a.user_id
            WHERE $2::text IS NULL OR a.action = $2 ORDER BY a.at DESC LIMIT $1
        """, limit, action)
    out = []
    for r in rows:
        d = dict(r); d["user_id"] = str(d["user_id"]) if d["user_id"] else None
        d["detail"] = json.loads(d["detail"]) if isinstance(d["detail"], str) else d["detail"]
        out.append(d)
    return {"status": "success", "data": out}
