"""
tests/unit/test_auth_accounts.py
User accounts / JWT sessions / API keys (app/auth.py) — the new auth layer
alongside the existing static-token system already covered by
tests/unit/test_auth.py. All Postgres calls are mocked (no live DB in CI).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
from unittest.mock import patch, MagicMock

import bcrypt
import psycopg2
import pytest

os.environ["API_TOKENS"]         = "dev-token"
os.environ["USE_REAL_ARTEFACTS"] = "false"
os.environ["JWT_SECRET_KEY"]     = "unit-test-secret-at-least-32-bytes-long"

# Patch Redis/ClickHouse/Postgres at import time — app/main.py calls
# init_schema() (app/db.py) as soon as it's imported, which would otherwise
# try a real Postgres connection.
with patch("redis.Redis") as mock_redis, \
     patch("clickhouse_connect.get_client") as mock_ch, \
     patch("psycopg2.connect") as mock_pg_connect:
    mock_redis.return_value = MagicMock()
    mock_ch.return_value    = MagicMock()
    mock_pg_connect.return_value = MagicMock()
    from fastapi.testclient import TestClient
    import app.auth as auth_module
    from app.main import app

client = TestClient(app)


def _mock_cursor(fetchone_results=None, execute_side_effect=None):
    """A cursor whose fetchone() returns each of fetchone_results in turn,
    wired into app.auth.get_cursor's context-manager protocol."""
    cur = MagicMock()
    if fetchone_results is not None:
        cur.fetchone.side_effect = fetchone_results
    if execute_side_effect is not None:
        cur.execute.side_effect = execute_side_effect

    cm = MagicMock()
    cm.__enter__.return_value = cur
    cm.__exit__.return_value = False
    return cur, cm


# ── Registration ──────────────────────────────────────────────────────────────

def test_register_rejects_neither_tenant_name_nor_invite_code():
    r = client.post("/auth/register", json={"email": "a@a.com", "password": "longenough1"})
    assert r.status_code == 400

def test_register_rejects_both_tenant_name_and_invite_code():
    r = client.post("/auth/register", json={
        "email": "a@a.com", "password": "longenough1",
        "tenant_name": "Acme", "invite_code": "ABC123",
    })
    assert r.status_code == 400

def test_register_creates_new_tenant_and_returns_jwt():
    _, cm = _mock_cursor(fetchone_results=[{"n": 0}, {"id": 1}])
    with patch("app.auth.get_cursor", return_value=cm):
        r = client.post("/auth/register", json={
            "email": "founder@bank.com", "password": "longenough1",
            "tenant_name": "Bank of America",
        })
    assert r.status_code == 201
    body = r.json()
    assert body["access_token"]
    assert body["invite_code"] is not None          # shown once, on tenant creation
    assert body["role"] == "admin"                  # first-ever user bootstrap

def test_register_joins_existing_tenant_via_invite_code():
    _, cm = _mock_cursor(fetchone_results=[
        {"tenant_id": "existing-tenant"},  # invite code lookup
        {"n": 5},                          # not the first user
        {"id": 6},
    ])
    with patch("app.auth.get_cursor", return_value=cm):
        r = client.post("/auth/register", json={
            "email": "colleague@bank.com", "password": "longenough1",
            "invite_code": "ABC123",
        })
    assert r.status_code == 201
    body = r.json()
    assert body["tenant_id"] == "existing-tenant"
    assert body["role"] == "user"
    assert body["invite_code"] is None              # only shown when creating a tenant

def test_register_rejects_invalid_invite_code():
    _, cm = _mock_cursor(fetchone_results=[None])
    with patch("app.auth.get_cursor", return_value=cm):
        r = client.post("/auth/register", json={
            "email": "x@x.com", "password": "longenough1", "invite_code": "NOPE",
        })
    assert r.status_code == 404

def test_register_duplicate_email_returns_409():
    cur, cm = _mock_cursor(fetchone_results=[{"n": 0}])
    cur.execute.side_effect = [None, None, psycopg2.errors.UniqueViolation()]
    with patch("app.auth.get_cursor", return_value=cm):
        r = client.post("/auth/register", json={
            "email": "dup@bank.com", "password": "longenough1", "tenant_name": "Acme",
        })
    assert r.status_code == 409

@pytest.mark.parametrize("password", ["short", "a" * 73])
def test_register_rejects_invalid_password_length(password):
    r = client.post("/auth/register", json={
        "email": "x@x.com", "password": password, "tenant_name": "Acme",
    })
    assert r.status_code == 422


# ── Login ─────────────────────────────────────────────────────────────────────

def test_login_success():
    password_hash = bcrypt.hashpw(b"correctpassword", bcrypt.gensalt()).decode()
    _, cm = _mock_cursor(fetchone_results=[
        {"id": 1, "password_hash": password_hash, "tenant_id": "t1", "role": "user"},
    ])
    with patch("app.auth.get_cursor", return_value=cm):
        r = client.post("/auth/login", json={"email": "u@u.com", "password": "correctpassword"})
    assert r.status_code == 200
    assert r.json()["tenant_id"] == "t1"

def test_login_wrong_password():
    password_hash = bcrypt.hashpw(b"correctpassword", bcrypt.gensalt()).decode()
    _, cm = _mock_cursor(fetchone_results=[
        {"id": 1, "password_hash": password_hash, "tenant_id": "t1", "role": "user"},
    ])
    with patch("app.auth.get_cursor", return_value=cm):
        r = client.post("/auth/login", json={"email": "u@u.com", "password": "wrongpassword"})
    assert r.status_code == 401

def test_login_unknown_email():
    _, cm = _mock_cursor(fetchone_results=[None])
    with patch("app.auth.get_cursor", return_value=cm):
        r = client.post("/auth/login", json={"email": "nobody@u.com", "password": "whatever1"})
    assert r.status_code == 401


# ── /auth/me ──────────────────────────────────────────────────────────────────

def test_me_returns_user_info():
    token = auth_module.create_jwt(user_id=1, tenant_id="t1", role="user")
    _, cm = _mock_cursor(fetchone_results=[{"email": "u@u.com", "tenant_id": "t1", "role": "user"}])
    with patch("app.auth.get_cursor", return_value=cm):
        r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["email"] == "u@u.com"

def test_me_rejects_garbage_token():
    r = client.get("/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401

def test_me_rejects_missing_token():
    r = client.get("/auth/me")
    assert r.status_code == 401


# ── /auth/api-key ─────────────────────────────────────────────────────────────

def test_api_key_regenerate_deletes_old_key_first():
    token = auth_module.create_jwt(user_id=1, tenant_id="t1", role="user")
    cur, cm = _mock_cursor()
    with patch("app.auth.get_cursor", return_value=cm):
        r = client.post("/auth/api-key", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["api_key"].startswith("sk_")

    executed = [call.args[0] for call in cur.execute.call_args_list]
    delete_idx = next(i for i, sql in enumerate(executed) if "DELETE" in sql)
    insert_idx = next(i for i, sql in enumerate(executed) if "INSERT" in sql)
    assert delete_idx < insert_idx
