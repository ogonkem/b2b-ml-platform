"""
tests/unit/test_batch_results.py
Tests for GET /v1/batch/results/{job_id} in app/main.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import pytest
from unittest.mock import patch, MagicMock

os.environ["API_TOKENS"]         = "dev-token,tenant-abc"
os.environ["USE_REAL_ARTEFACTS"] = "false"

with patch("redis.Redis") as _mr, \
     patch("clickhouse_connect.get_client") as _mc, \
     patch("minio.Minio") as _mm, \
     patch("app.main._celery_app", None):
    _mr.return_value = MagicMock()
    _mc.return_value = MagicMock()
    _mm.return_value = MagicMock()
    from fastapi.testclient import TestClient
    from app.main import app

client  = TestClient(app)
HEADERS = {"Authorization": "Bearer dev-token"}
JOB_ID  = "abc-123-job"


def get_results(job_id=JOB_ID, headers=None):
    return client.get(f"/v1/batch/results/{job_id}", headers=headers or HEADERS)


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_no_token_returns_401():
    assert client.get(f"/v1/batch/results/{JOB_ID}").status_code == 401

def test_wrong_token_returns_403_after_tenant_check():
    with patch("app.main.redis_client") as mock_r:
        mock_r.get.return_value = "dev-token"          # job belongs to dev-token
        response = get_results(headers={"Authorization": "Bearer wrong-token"})
    assert response.status_code == 403


# ── Job not found ─────────────────────────────────────────────────────────────

def test_unknown_job_id_returns_404():
    with patch("app.main.redis_client") as mock_r:
        mock_r.get.return_value = None
        assert get_results().status_code == 404

def test_wrong_tenant_cannot_see_job():
    """Tenant B must not see tenant A's job."""
    def side_effect(key):
        if "tenant" in key: return "other-tenant"
        return "queued"
    with patch("app.main.redis_client") as mock_r:
        mock_r.get.side_effect = side_effect
        assert get_results().status_code == 403


# ── Status: queued / processing ───────────────────────────────────────────────

def test_queued_status_returned():
    def side_effect(key):
        if "tenant" in key: return "dev-token"
        if "status" in key: return "queued"
        return None
    with patch("app.main.redis_client") as mock_r:
        mock_r.get.side_effect = side_effect
        body = get_results().json()
    assert body["status"]  == "queued"
    assert body["job_id"]  == JOB_ID

def test_processing_status_returned():
    def side_effect(key):
        if "tenant" in key: return "dev-token"
        if "status" in key: return "processing"
        return None
    with patch("app.main.redis_client") as mock_r:
        mock_r.get.side_effect = side_effect
        body = get_results().json()
    assert body["status"] == "processing"
    assert "download_url" not in body

def test_queued_job_has_no_download_url():
    def side_effect(key):
        if "tenant" in key: return "dev-token"
        if "status" in key: return "queued"
        return None
    with patch("app.main.redis_client") as mock_r:
        mock_r.get.side_effect = side_effect
        body = get_results().json()
    assert "download_url" not in body


# ── Status: complete ──────────────────────────────────────────────────────────

def test_complete_status_includes_download_url():
    def side_effect(key):
        if "tenant"        in key: return "dev-token"
        if "status"        in key: return "complete"
        if "result_object" in key: return "dev-token/20240101/abc_results.csv"
        if "rows_scored"   in key: return "50"
        return None
    with patch("app.main.redis_client") as mock_r, \
         patch("app.main.minio_client") as mock_minio:
        mock_r.get.side_effect = side_effect
        mock_minio.presigned_get_object.return_value = "http://minio/signed-url"
        body = get_results().json()
    assert body["status"]      == "complete"
    assert "download_url"      in body
    assert "rows_scored"       in body

def test_complete_rows_scored_is_integer():
    def side_effect(key):
        if "tenant"        in key: return "dev-token"
        if "status"        in key: return "complete"
        if "result_object" in key: return "dev-token/date/abc_results.csv"
        if "rows_scored"   in key: return "75"
        return None
    with patch("app.main.redis_client") as mock_r, \
         patch("app.main.minio_client") as mock_minio:
        mock_r.get.side_effect = side_effect
        mock_minio.presigned_get_object.return_value = "http://signed"
        body = get_results().json()
    assert body["rows_scored"] == 75
    assert isinstance(body["rows_scored"], int)

def test_complete_presigned_url_uses_batch_results_bucket():
    def side_effect(key):
        if "tenant"        in key: return "dev-token"
        if "status"        in key: return "complete"
        if "result_object" in key: return "dev-token/date/abc_results.csv"
        if "rows_scored"   in key: return "10"
        return None
    with patch("app.main.redis_client") as mock_r, \
         patch("app.main.minio_client") as mock_minio:
        mock_r.get.side_effect = side_effect
        mock_minio.presigned_get_object.return_value = "http://signed"
        get_results()
        call_args = mock_minio.presigned_get_object.call_args[0]
        assert call_args[0] == "batch-results"


# ── Status: failed ────────────────────────────────────────────────────────────

def test_failed_status_includes_error_field():
    def side_effect(key):
        if "tenant" in key: return "dev-token"
        if "status" in key: return "failed"
        if "error"  in key: return "MinIO connection refused"
        return None
    with patch("app.main.redis_client") as mock_r:
        mock_r.get.side_effect = side_effect
        body = get_results().json()
    assert body["status"] == "failed"
    assert body["error"]  == "MinIO connection refused"

def test_failed_status_has_no_download_url():
    def side_effect(key):
        if "tenant" in key: return "dev-token"
        if "status" in key: return "failed"
        if "error"  in key: return "timeout"
        return None
    with patch("app.main.redis_client") as mock_r:
        mock_r.get.side_effect = side_effect
        body = get_results().json()
    assert "download_url" not in body
