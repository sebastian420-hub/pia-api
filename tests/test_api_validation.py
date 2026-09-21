"""
Unit tests that need no database: auth gate, input validation, error shape.
Run: PIA_API_TOKEN=test pytest tests/
"""
import os

os.environ.setdefault("PIA_API_TOKEN", "test-token")
os.environ.setdefault("DB_HOST", "127.0.0.1")

import pytest
from fastapi.testclient import TestClient

import main

AUTH = {"Authorization": "Bearer test-token"}


class FakeConn:
    """Just enough Postgres for auth: users/tokens/sources come from an in-memory table."""
    users = {"test-token": ("11111111-1111-1111-1111-111111111111", "owner", "admin"),
             "viewer-token": ("22222222-2222-2222-2222-222222222222", "vic", "viewer"),
             "analyst-token": ("33333333-3333-3333-3333-333333333333", "ana", "analyst")}
    audit = []

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def fetchrow(self, sql, *args):
        import auth
        if "FROM api_tokens t JOIN users u" in sql:
            for tok, (uid, name, role) in self.users.items():
                if auth.token_hash(tok) == args[0]:
                    return {"user_id": uid, "name": name, "role": role}
            return None
        if "WHERE role = 'admin'" in sql:
            return {"user_id": self.users["test-token"][0], "name": "owner", "role": "admin"}
        return None

    async def fetch(self, sql, *args):
        if "FROM sources s" in sql:
            return [{"source_id": "bbc.co.uk"}]
        if "visibility = 'restricted'" in sql:
            return [{"source_id": "reporter:crow"}]
        return []

    async def fetchval(self, sql, *args): return 0

    async def execute(self, sql, *args):
        if "INSERT INTO audit_log" in sql:
            self.audit.append(args)


class FakePool:
    def acquire(self): return FakeConn()


@pytest.fixture(scope="module")
def client():
    # Do not run the lifespan (it would connect to Postgres). The fake pool answers the
    # auth queries; the routes under test fail validation (or role) before they touch data.
    main.app.state.pool = FakePool()
    return TestClient(main.app)


VIEWER = {"Authorization": "Bearer viewer-token"}
ANALYST = {"Authorization": "Bearer analyst-token"}


def test_root_is_public(client):
    assert client.get("/").status_code == 200


def test_missing_token_is_401(client):
    r = client.get("/api/v1/users")
    assert r.status_code == 401
    assert r.json() == {"status": "error", "message": "Missing or invalid bearer token"}


def test_wrong_token_is_401(client):
    assert client.get("/api/v1/users", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_pagination_limit_zero_is_422(client):
    r = client.get("/api/v1/archive?page=1&limit=0", headers=AUTH)
    assert r.status_code == 422
    assert r.json()["status"] == "error"


def test_negative_page_is_422(client):
    assert client.get("/api/v1/entities?page=-1", headers=AUTH).status_code == 422


def test_bbox_rejects_bad_latitude(client):
    r = client.get("/api/v1/entities/bbox?minLat=-95&minLon=0&maxLat=10&maxLon=10", headers=AUTH)
    assert r.status_code == 422


def test_event_rejects_non_uuid(client):
    assert client.get("/api/v1/event/not-a-uuid", headers=AUTH).status_code == 422


def test_semantic_search_rejects_bad_target(client):
    r = client.post("/api/v1/search/semantic", json={"query": "x", "target": "tables"}, headers=AUTH)
    assert r.status_code == 422


def test_upload_rejects_extension(client):
    r = client.post("/api/v1/documents/upload", files={"file": ("evil.exe", b"x", "application/octet-stream")}, headers=AUTH)
    assert r.status_code == 422


def test_upload_sanitises_traversal_name(client, tmp_path, monkeypatch):
    import routers
    monkeypatch.setattr(routers, "DOC_DIR", str(tmp_path))
    r = client.post("/api/v1/documents/upload",
                    files={"file": ("../../escape.txt", b"hello", "text/plain")}, headers=AUTH)
    assert r.status_code == 200, r.text
    stored = r.json()["stored_as"]
    assert (tmp_path / stored).exists()
    assert ".." not in stored and "/" not in stored


def test_feedback_type_is_validated(client):
    r = client.post("/api/v1/feedback",
                    json={"relationship_id": "00000000-0000-0000-0000-000000000001", "feedback_type": "MEH"},
                    headers=AUTH)
    assert r.status_code == 422


def test_websocket_without_token_is_refused(client):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/live"):
            pass


# ── roles ──

def test_viewer_cannot_write(client):
    assert client.post("/api/v1/missions", json={"name": "x"}, headers=VIEWER).status_code == 403
    assert client.post("/api/v1/documents/upload", files={"file": ("a.txt", b"x", "text/plain")}, headers=VIEWER).status_code == 403
    assert client.get("/api/v1/users", headers=VIEWER).status_code == 403


def test_analyst_may_not_delete_missions_or_manage_users(client):
    assert client.delete("/api/v1/missions/00000000-0000-0000-0000-000000000001", headers=ANALYST).status_code == 403
    assert client.get("/api/v1/users", headers=ANALYST).status_code == 403
    assert client.post("/api/v1/feedback", json={"relationship_id": "x", "feedback_type": "MEH"}, headers=ANALYST).status_code == 422  # past the role gate


def test_me_reports_role_and_bootstrap_admin(client):
    r = client.get("/api/v1/me", headers=AUTH)
    assert r.status_code == 200 and r.json()["data"]["role"] == "admin"
    r = client.get("/api/v1/me", headers=VIEWER)
    assert r.json()["data"]["role"] == "viewer"


def test_visibility_fragment_hides_restricted_sources():
    from auth import User, Visibility
    v = Visibility(User("u", "vic", "viewer", "h"), ["reporter:crow", "it's"])
    frag = v.sql("ev.source_id")
    assert "ev.source_id IS NULL OR ev.source_id NOT IN ('reporter:crow', 'it''s')" in frag
    assert not v.allows("reporter:crow") and v.allows("bbc.co.uk") and v.allows(None)
    assert Visibility(User("a", "owner", "admin", "h"), []).sql("x") == ""
