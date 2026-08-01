"""
tests/unit/test_payments.py
POST /v1/tenant/checkout, POST /webhooks/paystack, GET /v1/tenant/billing-portal,
and the cancel guard on POST /v1/tenant/plan — all Paystack HTTP calls are
mocked, no real network or API key needed to run this.
"""
import hashlib
import hmac
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
from unittest.mock import patch, MagicMock

os.environ["API_TOKENS"]          = "dev-token"
os.environ["USE_REAL_ARTEFACTS"]  = "false"
os.environ["JWT_SECRET_KEY"]      = "unit-test-secret-at-least-32-bytes-long"
os.environ["PAYSTACK_SECRET_KEY"] = "sk_test_unit_secret"
os.environ["PAYSTACK_PLAN_STARTER"] = "PLN_starter_test"
os.environ["PAYSTACK_PLAN_PRO"]     = "PLN_pro_test"

with patch("redis.Redis") as mock_redis_cls, \
     patch("clickhouse_connect.get_client") as mock_ch, \
     patch("psycopg2.connect") as mock_pg_connect:
    mock_redis_cls.return_value  = MagicMock()
    mock_ch.return_value         = MagicMock()
    mock_pg_connect.return_value = MagicMock()
    from fastapi.testclient import TestClient
    import app.auth as auth_module
    from app.main import app

client = TestClient(app)
HEADERS = {"Authorization": "Bearer dev-token"}
PAYSTACK_SECRET = os.environ["PAYSTACK_SECRET_KEY"]


def _jwt_headers(role="user", tenant_id="t1"):
    token = auth_module.create_jwt(user_id=1, tenant_id=tenant_id, role=role)
    return {"Authorization": f"Bearer {token}"}


def _mock_cursor_cm(fetchone_result=None, rowcount=1):
    cur = MagicMock()
    cur.fetchone.return_value = fetchone_result
    cur.rowcount = rowcount
    cm = MagicMock()
    cm.__enter__.return_value = cur
    cm.__exit__.return_value = False
    return cur, cm


def _sign(body: bytes) -> str:
    return hmac.new(PAYSTACK_SECRET.encode(), body, hashlib.sha512).hexdigest()


# ── POST /v1/tenant/checkout ─────────────────────────────────────────────────

def test_checkout_rejects_free_plan():
    r = client.post("/v1/tenant/checkout", json={"plan": "free"}, headers=_jwt_headers())
    assert r.status_code == 400
    assert "checkout" in r.json()["detail"]

def test_checkout_requires_jwt():
    r = client.post("/v1/tenant/checkout", json={"plan": "starter"}, headers=HEADERS)
    assert r.status_code == 401

def test_checkout_starts_paystack_transaction():
    _, cm = _mock_cursor_cm(fetchone_result={"email": "user@example.com"})
    paystack_resp = MagicMock()
    paystack_resp.raise_for_status.return_value = None
    paystack_resp.json.return_value = {
        "data": {"authorization_url": "https://checkout.paystack.com/abc123", "reference": "ref1"}
    }
    with patch("app.db.get_cursor", return_value=cm), \
         patch("requests.post", return_value=paystack_resp) as mock_post:
        r = client.post("/v1/tenant/checkout", json={"plan": "starter"}, headers=_jwt_headers())
    assert r.status_code == 200
    assert r.json()["authorization_url"] == "https://checkout.paystack.com/abc123"
    sent_json = mock_post.call_args.kwargs["json"]
    assert sent_json["email"] == "user@example.com"
    assert sent_json["plan"] == "PLN_starter_test"
    # Paystack requires an explicit amount even with a plan code — must be a
    # positive integer in the currency's smallest subunit, not left out.
    assert isinstance(sent_json["amount"], int)
    assert sent_json["amount"] > 0


# ── POST /webhooks/paystack ───────────────────────────────────────────────────

def test_webhook_rejects_bad_signature():
    body = json.dumps({"event": "subscription.create", "data": {}}).encode()
    r = client.post(
        "/webhooks/paystack",
        content=body,
        headers={"x-paystack-signature": "not-the-real-signature", "Content-Type": "application/json"},
    )
    assert r.status_code == 401

def test_webhook_rejects_missing_signature():
    body = json.dumps({"event": "subscription.create", "data": {}}).encode()
    r = client.post("/webhooks/paystack", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 401

def test_webhook_subscription_create_updates_tenant_plan():
    body = json.dumps({
        "event": "subscription.create",
        "data": {
            "customer": {"email": "payer@example.com", "customer_code": "CUS_1"},
            "plan": {"plan_code": "PLN_starter_test"},
            "subscription_code": "SUB_1",
        },
    }).encode()
    with patch("app.db.get_tenant_id_by_email", return_value="tenant-xyz"), \
         patch("app.db.set_tenant_subscription") as mock_set:
        r = client.post(
            "/webhooks/paystack",
            content=body,
            headers={"x-paystack-signature": _sign(body), "Content-Type": "application/json"},
        )
    assert r.status_code == 200
    mock_set.assert_called_once_with(
        "tenant-xyz", "starter", customer_code="CUS_1", subscription_code="SUB_1", status="active"
    )

def test_webhook_subscription_disable_reverts_to_free():
    body = json.dumps({
        "event": "subscription.disable",
        "data": {"customer": {"email": "payer@example.com"}},
    }).encode()
    with patch("app.db.get_tenant_id_by_email", return_value="tenant-xyz"), \
         patch("app.db.get_tenant_row", return_value={
             "paystack_customer_code": "CUS_1", "paystack_subscription_code": "SUB_1"
         }), \
         patch("app.db.set_tenant_subscription") as mock_set:
        r = client.post(
            "/webhooks/paystack",
            content=body,
            headers={"x-paystack-signature": _sign(body), "Content-Type": "application/json"},
        )
    assert r.status_code == 200
    mock_set.assert_called_once_with(
        "tenant-xyz", "free", customer_code="CUS_1", subscription_code="SUB_1", status="canceled"
    )

def test_webhook_still_returns_200_when_processing_fails():
    """Signature is valid — this is a genuine Paystack event — so we must
    not make Paystack retry-storm us over an internal error."""
    body = json.dumps({
        "event": "subscription.create",
        "data": {"customer": {"email": "x@example.com"}, "plan": {"plan_code": "PLN_starter_test"}},
    }).encode()
    with patch("app.db.get_tenant_id_by_email", side_effect=Exception("db down")):
        r = client.post(
            "/webhooks/paystack",
            content=body,
            headers={"x-paystack-signature": _sign(body), "Content-Type": "application/json"},
        )
    assert r.status_code == 200


# ── Cancel guard on POST /v1/tenant/plan ──────────────────────────────────────

def test_cancel_to_free_blocked_while_subscription_active():
    with patch("app.db.get_tenant_row", return_value={"subscription_status": "active"}):
        r = client.post("/v1/tenant/plan", json={"plan": "free"}, headers=HEADERS)
    assert r.status_code == 400
    assert "billing portal" in r.json()["detail"]

def test_switch_to_free_allowed_when_no_active_subscription():
    _, cm = _mock_cursor_cm(rowcount=1)
    with patch("app.db.get_tenant_row", return_value={"subscription_status": "none"}), \
         patch("app.db.get_cursor", return_value=cm):
        r = client.post("/v1/tenant/plan", json={"plan": "free"}, headers=HEADERS)
    assert r.status_code == 200


# ── GET /v1/tenant/billing-portal ─────────────────────────────────────────────

def test_billing_portal_requires_existing_subscription():
    with patch("app.db.get_tenant_row", return_value={"paystack_subscription_code": None}):
        r = client.get("/v1/tenant/billing-portal", headers=_jwt_headers())
    assert r.status_code == 400

def test_billing_portal_returns_manage_link():
    paystack_resp = MagicMock()
    paystack_resp.raise_for_status.return_value = None
    paystack_resp.json.return_value = {"data": {"link": "https://paystack.com/manage/subscriptions/abc"}}
    with patch("app.db.get_tenant_row", return_value={"paystack_subscription_code": "SUB_1"}), \
         patch("requests.get", return_value=paystack_resp):
        r = client.get("/v1/tenant/billing-portal", headers=_jwt_headers())
    assert r.status_code == 200
    assert r.json()["link"] == "https://paystack.com/manage/subscriptions/abc"

def test_billing_portal_requires_auth():
    r = client.get("/v1/tenant/billing-portal")
    assert r.status_code == 401
