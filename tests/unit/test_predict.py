"""
tests/unit/test_predict.py
Essential tests for POST /v1/predict and GET /health in app/main.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import pytest
from unittest.mock import patch, MagicMock

# Set env vars BEFORE importing the app
os.environ["API_TOKENS"]         = "dev-token,tenant-abc"
os.environ["USE_REAL_ARTEFACTS"] = "false"

# Patch Redis and ClickHouse at import time
with patch("redis.Redis") as mock_redis, \
     patch("clickhouse_connect.get_client") as mock_ch:
    mock_redis.return_value = MagicMock()
    mock_ch.return_value    = MagicMock()
    from fastapi.testclient import TestClient
    from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "ID":             1,
    "year":           2023,
    "loan_amount":    250000.0,
    "property_value": 320000.0,
    "income":         6000.0,
    "Credit_Score":   720.0,
}
HEADERS = {"Authorization": "Bearer dev-token"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def predict(payload=None, headers=None):
    return client.post(
        "/v1/predict",
        json=payload or VALID_PAYLOAD,
        headers=headers or HEADERS
    )


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_200():
    assert client.get("/health").status_code == 200

def test_health_returns_correct_fields():
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert "pipeline_fitted" in body


# ── Response structure ────────────────────────────────────────────────────────

def test_predict_returns_200():
    with patch("app.main.check_and_increment_quota", return_value=1), \
         patch("app.main.log_prediction"):
        assert predict().status_code == 200

def test_predict_response_schema():
    with patch("app.main.check_and_increment_quota", return_value=1), \
         patch("app.main.log_prediction"):
        body = predict().json()
        assert body["application_id"]      == VALID_PAYLOAD["ID"]
        assert body["default_prediction"]  in (0, 1)
        assert 0.0 <= body["default_probability"] <= 1.0
        assert body["status"]              == "success"

def test_probability_rounded_to_4dp():
    with patch("app.main.check_and_increment_quota", return_value=1), \
         patch("app.main.log_prediction"):
        prob = predict().json()["default_probability"]
        assert prob == round(prob, 4)


# ── Input validation ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("missing_field", [
    "ID", "year", "loan_amount", "property_value", "income", "Credit_Score"
])
def test_missing_required_field_returns_422(missing_field):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing_field}
    with patch("app.main.check_and_increment_quota", return_value=1), \
         patch("app.main.log_prediction"):
        assert predict(payload=payload).status_code == 422


# ── Authentication ────────────────────────────────────────────────────────────

def test_no_token_returns_401():
    assert client.post("/v1/predict", json=VALID_PAYLOAD).status_code == 401

def test_wrong_token_returns_403():
    with patch("app.main.check_and_increment_quota", return_value=1), \
         patch("app.main.log_prediction"):
        assert predict(headers={"Authorization": "Bearer wrong"}).status_code == 403

def test_wrong_scheme_returns_401():
    assert predict(headers={"Authorization": "Basic dev-token"}).status_code == 401


# ── Quota ─────────────────────────────────────────────────────────────────────

def test_exceeded_quota_returns_429():
    from fastapi import HTTPException
    def quota_exceeded(tenant_id, monthly_limit=1000):
        raise HTTPException(status_code=429, detail="Monthly prediction quota exceeded")

    with patch("app.main.check_and_increment_quota", side_effect=quota_exceeded), \
         patch("app.main.log_prediction"):
        assert predict().status_code == 429


# ── Logging ───────────────────────────────────────────────────────────────────

def test_log_prediction_called_on_success():
    with patch("app.main.check_and_increment_quota", return_value=1), \
         patch("app.main.log_prediction") as mock_log:
        predict()
        mock_log.assert_called_once()

def test_log_prediction_not_called_on_auth_failure():
    with patch("app.main.log_prediction") as mock_log:
        predict(headers={"Authorization": "Bearer bad-token"})
        mock_log.assert_not_called()


# ── Pipeline edge cases ───────────────────────────────────────────────────────

@pytest.mark.parametrize("field,value", [
    ("income", 0.0),
    ("loan_amount", 999_000_000.0),
    ("Credit_Score", 300.0),
])
def test_edge_case_values_do_not_crash(field, value):
    with patch("app.main.check_and_increment_quota", return_value=1), \
         patch("app.main.log_prediction"):
        assert predict(payload={**VALID_PAYLOAD, field: value}).status_code == 200

def test_same_input_returns_same_output():
    with patch("app.main.check_and_increment_quota", return_value=1), \
         patch("app.main.log_prediction"):
        results = [predict().json()["default_probability"] for _ in range(3)]
        assert len(set(results)) == 1