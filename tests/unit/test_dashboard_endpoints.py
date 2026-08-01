"""
tests/unit/test_dashboard_endpoints.py
GET /v1/usage, GET /v1/batch/jobs, GET /v1/predictions/history,
GET /admin/tenants — the new read endpoints the frontend dashboard needs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
from unittest.mock import patch, MagicMock

os.environ["API_TOKENS"]         = "dev-token"
os.environ["USE_REAL_ARTEFACTS"] = "false"
os.environ["JWT_SECRET_KEY"]     = "unit-test-secret-at-least-32-bytes-long"

with patch("redis.Redis") as mock_redis_cls, \
     patch("clickhouse_connect.get_client") as mock_ch, \
     patch("psycopg2.connect") as mock_pg_connect:
    mock_redis_cls.return_value = MagicMock()
    mock_ch.return_value        = MagicMock()
    mock_pg_connect.return_value = MagicMock()
    from fastapi.testclient import TestClient
    import app.auth as auth_module
    import app.main as main_module
    from app.main import app

from app.plans import PLANS, DEFAULT_PLAN

client = TestClient(app)
HEADERS = {"Authorization": "Bearer dev-token"}


def _mock_cursor_cm(fetchone_results=None):
    cur = MagicMock()
    if fetchone_results is not None:
        cur.fetchone.side_effect = fetchone_results
    cur.fetchall.return_value = []
    cm = MagicMock()
    cm.__enter__.return_value = cur
    cm.__exit__.return_value = False
    return cur, cm


# ── GET /v1/usage ─────────────────────────────────────────────────────────────

def test_usage_reads_current_quota():
    """No app.tenants row for this static token (psycopg2 is mocked, so the
    cursor's fetchone() returns a non-matching MagicMock rather than a real
    row) — get_tenant_plan falls back to the default plan, exactly like it
    would for a real, never-registered static API_TOKENS tenant."""
    with patch.object(main_module.redis_client, "get", return_value="42"):
        r = client.get("/v1/usage", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["used"] == 42
    assert body["plan"] == DEFAULT_PLAN
    assert body["limit"] == PLANS[DEFAULT_PLAN]["monthly_quota"]

def test_usage_defaults_to_zero_when_no_key():
    with patch.object(main_module.redis_client, "get", return_value=None):
        r = client.get("/v1/usage", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["used"] == 0

def test_usage_requires_auth():
    r = client.get("/v1/usage")
    assert r.status_code == 401


# ── GET /v1/batch/jobs ────────────────────────────────────────────────────────

def test_batch_jobs_lists_tenant_jobs():
    with patch.object(main_module.redis_client, "lrange", return_value=["job-1", "job-2"]), \
         patch.object(main_module.redis_client, "get", side_effect=["queued", "complete"]):
        r = client.get("/v1/batch/jobs", headers=HEADERS)
    assert r.status_code == 200
    jobs = r.json()["jobs"]
    assert jobs == [
        {"job_id": "job-1", "status": "queued"},
        {"job_id": "job-2", "status": "complete"},
    ]

def test_batch_jobs_empty_when_none():
    with patch.object(main_module.redis_client, "lrange", return_value=[]):
        r = client.get("/v1/batch/jobs", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["jobs"] == []


# ── GET /v1/predictions/history ───────────────────────────────────────────────

def test_predictions_history_returns_empty_when_clickhouse_unavailable():
    with patch("app.main.get_ch_client", return_value=None):
        r = client.get("/v1/predictions/history", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["predictions"] == []

def test_predictions_history_handles_query_failure_gracefully():
    mock_client = MagicMock()
    mock_client.query.side_effect = Exception("clickhouse down")
    with patch("app.main.get_ch_client", return_value=mock_client):
        r = client.get("/v1/predictions/history", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["predictions"] == []


# ── GET /admin/tenants ────────────────────────────────────────────────────────

def test_admin_tenants_rejects_non_admin_jwt():
    token = auth_module.create_jwt(user_id=1, tenant_id="t1", role="user")
    r = client.get("/admin/tenants", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403

def test_admin_tenants_rejects_static_token():
    """A static API_TOKENS value / API key identifies a tenant, not a user —
    it can never carry role=admin, so admin endpoints must reject it too."""
    r = client.get("/admin/tenants", headers=HEADERS)
    assert r.status_code == 401  # no valid JWT at all -> get_current_user rejects it

def test_admin_tenants_allows_admin_jwt():
    token = auth_module.create_jwt(user_id=1, tenant_id="t1", role="admin")
    _, cm = _mock_cursor_cm()
    # admin_tenants() does `from app.db import get_cursor` locally, so the
    # patch target is the source (app.db), not app.main's namespace.
    with patch("app.db.get_cursor", return_value=cm), \
         patch.object(main_module.redis_client, "get", return_value="10"):
        r = client.get("/admin/tenants", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json() == {"tenants": []}
