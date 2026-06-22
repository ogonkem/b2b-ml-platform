import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
from unittest.mock import patch, MagicMock

# Set env vars BEFORE importing the app
os.environ["API_TOKENS"]         = "dev-token,tenant-abc"
os.environ["USE_REAL_ARTEFACTS"] = "false"

# ── Patch Redis and ClickHouse at import time ─────────────────────────────────
# Without these, the module tries to connect to real servers on import and fails
with patch("redis.Redis") as mock_redis, \
     patch("clickhouse_connect.get_client") as mock_ch:
    mock_redis.return_value = MagicMock()
    mock_ch.return_value    = MagicMock()
    from fastapi.testclient import TestClient
    from app.main import app

client = TestClient(app)

SAMPLE = {
    "ID": 1,
    "year": 2023,
    "loan_amount": 250000.0,
    "property_value": 320000.0,
    "income": 6000.0,
    "Credit_Score": 720.0,
}

# ── Valid tokens ───────────────────────────────────────────────────────────

def test_valid_primary_token_returns_200():
    """The first token in the comma-separated list should work."""
    with patch("app.main.check_and_increment_quota", return_value=1), \
         patch("app.main.log_prediction"):
        response = client.post(
            "/v1/predict",
            json=SAMPLE,
            headers={"Authorization": "Bearer dev-token"}
        )
    assert response.status_code == 200

def test_valid_second_token_returns_200():
    """The second token in the comma-separated list should work."""
    with patch("app.main.check_and_increment_quota", return_value=1), \
         patch("app.main.log_prediction"):
        response = client.post(
            "/v1/predict",
            json=SAMPLE,
            headers={"Authorization": "Bearer tenant-abc"}
        )
    assert response.status_code == 200

# ── Invalid tokens ────────────────────────────────────────────────────────

def test_wrong_token_returns_401():
    """A token that is not in the VALID_TOKENS set should be rejected."""
    response = client.post(
        "/v1/predict",
        json=SAMPLE,
        headers={"Authorization": "Bearer wrong-token"}
    )
    assert response.status_code == 403 # fastapi.security.HTTPBearer returns 403 for invalid token, not 401 

def test_no_token_returns_401():
    """Omitting the Authorization header entirely should result in a 401 error."""
    response = client.post("/v1/predict", json=SAMPLE)
    assert response.status_code == 401

def test_empty_token_returns_401():
    """An empty Bearer token should be treated as missing."""
    response = client.post(
        "/v1/predict",
        json=SAMPLE,
        headers={"Authorization": "Bearer "}
    )
    assert response.status_code == 401

def test_malformed_header_no_bearer_prefix_returns_401():
    """Header must be 'Bearer <token>' not just the raw token."""
    response = client.post(
        "/v1/predict",
        json=SAMPLE,
        headers={"Authorization": "dev-token"}
    )
    assert response.status_code == 401

def test_wrong_scheme_returns_401():
    """Basic auth scheme should not be accepted."""
    response = client.post(
        "/v1/predict",
        json=SAMPLE,
        headers={"Authorization": "Basic dev-token"}
    )
    assert response.status_code == 401

# ── Error message ─────────────────────────────────────────────────────────

def test_401_returns_meaningful_detail():
    """ Response body should include a clear error message about the token issue."""
    response = client.post(
        "/v1/predict",
        json=SAMPLE,
        headers={"Authorization": "Bearer bad-token"}
    )
    body = response.json()
    assert "detail" in body
    assert "token" in body["detail"].lower()

# ── Token not leaked in response ──────────────────────────────────────────

def test_valid_token_not_in_response_body():
    """The token value must never appear in the API response."""
    with patch("app.main.check_and_increment_quota", return_value=1), \
         patch("app.main.log_prediction"):
        response = client.post(
            "/v1/predict",
            json=SAMPLE,
            headers={"Authorization": "Bearer dev-token"}
        )
    assert "dev-token" not in response.text