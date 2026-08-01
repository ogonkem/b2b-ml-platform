"""
tests/unit/test_plans.py
GET /v1/plans, POST /v1/tenant/plan, and GET /v1/usage's plan-aware limit —
subscription tier catalog and self-service plan switching.
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
    mock_redis_cls.return_value  = MagicMock()
    mock_ch.return_value         = MagicMock()
    mock_pg_connect.return_value = MagicMock()
    from fastapi.testclient import TestClient
    import app.main as main_module
    from app.main import app

from app.plans import PLANS, DEFAULT_PLAN

client = TestClient(app)
HEADERS = {"Authorization": "Bearer dev-token"}


def _mock_cursor_cm(rowcount=1):
    cur = MagicMock()
    cur.rowcount = rowcount
    cm = MagicMock()
    cm.__enter__.return_value = cur
    cm.__exit__.return_value = False
    return cur, cm


# ── GET /v1/plans ────────────────────────────────────────────────────────────

def test_list_plans_returns_all_tiers():
    r = client.get("/v1/plans", headers=HEADERS)
    assert r.status_code == 200
    plans = r.json()["plans"]
    ids = {p["id"] for p in plans}
    assert ids == set(PLANS.keys())

def test_list_plans_includes_quota_and_price():
    r = client.get("/v1/plans", headers=HEADERS)
    free = next(p for p in r.json()["plans"] if p["id"] == "free")
    assert free["monthly_quota"] == PLANS["free"]["monthly_quota"]
    assert free["price_label"] == PLANS["free"]["price_label"]

def test_list_plans_requires_auth():
    r = client.get("/v1/plans")
    assert r.status_code == 401


# ── POST /v1/tenant/plan ─────────────────────────────────────────────────────

def test_change_plan_rejects_unknown_plan():
    r = client.post("/v1/tenant/plan", json={"plan": "ultra-mega"}, headers=HEADERS)
    assert r.status_code == 400
    assert "Unknown plan" in r.json()["detail"]

def test_change_plan_rejects_static_token_with_no_tenant_row():
    """set_tenant_plan's UPDATE matches zero rows for a tenant_id that was
    never registered — a static API_TOKENS value, exactly like HEADERS
    here — so this must be a clear rejection, not a silent no-op."""
    _, cm = _mock_cursor_cm(rowcount=0)
    with patch("app.db.get_cursor", return_value=cm):
        r = client.post("/v1/tenant/plan", json={"plan": "starter"}, headers=HEADERS)
    assert r.status_code == 400
    assert "register" in r.json()["detail"].lower()

def test_change_plan_succeeds_for_registered_tenant():
    _, cm = _mock_cursor_cm(rowcount=1)
    with patch("app.db.get_cursor", return_value=cm):
        r = client.post("/v1/tenant/plan", json={"plan": "starter"}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["plan"] == "starter"
    assert body["monthly_quota"] == PLANS["starter"]["monthly_quota"]

def test_change_plan_requires_auth():
    r = client.post("/v1/tenant/plan", json={"plan": "starter"})
    assert r.status_code == 401


# ── GET /v1/usage reflects plan-based limit ──────────────────────────────────

def test_usage_limit_matches_tenants_plan():
    row = {"plan": "pro"}
    cur = MagicMock()
    cur.fetchone.return_value = row
    cm = MagicMock()
    cm.__enter__.return_value = cur
    cm.__exit__.return_value = False
    with patch("app.db.get_cursor", return_value=cm), \
         patch.object(main_module.redis_client, "get", return_value="5"):
        r = client.get("/v1/usage", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["plan"] == "pro"
    assert body["limit"] == PLANS["pro"]["monthly_quota"]
