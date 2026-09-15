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


@pytest.fixture(scope="module")
def client():
    # Do not run the lifespan (it would connect to Postgres). A sentinel pool lets
    # dependency resolution pass; the routes under test fail validation before
    # they ever touch it.
    main.app.state.pool = object()
    return TestClient(main.app)


def test_root_is_public(client):
    assert client.get("/").status_code == 200


def test_missing_token_is_401(client):
    r = client.get("/api/v1/logs")
    assert r.status_code == 401
    assert r.json() == {"status": "error", "message": "Missing or invalid bearer token"}


def test_wrong_token_is_401(client):
    assert client.get("/api/v1/logs", headers={"Authorization": "Bearer nope"}).status_code == 401


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
