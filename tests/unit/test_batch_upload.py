"""
tests/unit/test_batch_upload.py
Tests for POST /v1/batch/upload in app/main.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import io
import pytest
from unittest.mock import patch, MagicMock

# Set env vars BEFORE importing the app
os.environ["API_TOKENS"]         = "dev-token,tenant-abc"
os.environ["USE_REAL_ARTEFACTS"] = "false"

# Patch all external services at import time
with patch("redis.Redis") as mock_redis, \
     patch("clickhouse_connect.get_client") as mock_ch, \
     patch("minio.Minio") as mock_minio:
    mock_redis.return_value = MagicMock()
    mock_ch.return_value    = MagicMock()
    mock_minio.return_value = MagicMock()
    from fastapi.testclient import TestClient
    from app.main import app

client = TestClient(app)

HEADERS = {"Authorization": "Bearer dev-token"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_csv(rows: int = 5) -> bytes:
    """Generate a valid CSV matching the loan schema."""
    lines = ["ID,year,loan_amount,property_value,income,Credit_Score"]
    for i in range(1, rows + 1):
        lines.append(f"{i},2023,250000,320000,6000,720")
    return "\n".join(lines).encode()

def upload(content=None, filename="test.csv", headers=None, content_type="text/csv"):
    """Helper to POST to /v1/batch/upload."""
    file_content = make_csv() if content is None else content
    return client.post(
        "/v1/batch/upload",
        headers=headers or HEADERS,
        files={"file": (filename, file_content, content_type)},
    )


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_no_token_returns_401():
    response = client.post(
        "/v1/batch/upload",
        files={"file": ("test.csv", make_csv(), "text/csv")}
    )
    assert response.status_code == 401

def test_wrong_token_returns_403():
    with patch("app.main.check_and_increment_quota", return_value=5), \
         patch("app.main.minio_client") as mock_minio:
        mock_minio.put_object.return_value = None
        response = upload(headers={"Authorization": "Bearer wrong-token"})
    assert response.status_code == 403


# ── Happy path ────────────────────────────────────────────────────────────────

def test_valid_upload_returns_200():
    with patch("app.main.check_and_increment_quota", return_value=5), \
         patch("app.main.minio_client") as mock_minio:
        mock_minio.put_object.return_value = None
        response = upload()
    assert response.status_code == 200

def test_response_schema():
    with patch("app.main.check_and_increment_quota", return_value=5), \
         patch("app.main.minio_client") as mock_minio:
        mock_minio.put_object.return_value = None
        body = upload().json()

    assert "job_id"        in body
    assert "tenant_id"     in body
    assert "rows_received" in body
    assert "object"        in body
    assert "status"        in body

def test_response_status_is_queued():
    with patch("app.main.check_and_increment_quota", return_value=5), \
         patch("app.main.minio_client") as mock_minio:
        mock_minio.put_object.return_value = None
        body = upload().json()
    assert body["status"] == "queued"

def test_response_tenant_id_matches_token():
    with patch("app.main.check_and_increment_quota", return_value=5), \
         patch("app.main.minio_client") as mock_minio:
        mock_minio.put_object.return_value = None
        body = upload().json()
    assert body["tenant_id"] == "dev-token"

def test_row_count_matches_csv():
    with patch("app.main.check_and_increment_quota", return_value=10), \
         patch("app.main.minio_client") as mock_minio:
        mock_minio.put_object.return_value = None
        body = upload(content=make_csv(rows=10)).json()
    assert body["rows_received"] == 10

def test_job_id_is_uuid():
    import uuid
    with patch("app.main.check_and_increment_quota", return_value=5), \
         patch("app.main.minio_client") as mock_minio:
        mock_minio.put_object.return_value = None
        body = upload().json()
    # Should not raise — valid UUID
    uuid.UUID(body["job_id"])

def test_object_path_contains_tenant():
    with patch("app.main.check_and_increment_quota", return_value=5), \
         patch("app.main.minio_client") as mock_minio:
        mock_minio.put_object.return_value = None
        body = upload().json()
    assert "dev-token" in body["object"]

def test_two_uploads_have_different_job_ids():
    with patch("app.main.check_and_increment_quota", return_value=5), \
         patch("app.main.minio_client") as mock_minio:
        mock_minio.put_object.return_value = None
        id1 = upload().json()["job_id"]
        id2 = upload().json()["job_id"]
    assert id1 != id2


# ── Input validation ──────────────────────────────────────────────────────────

def test_invalid_csv_returns_400():
    with patch("app.main.check_and_increment_quota", return_value=1), \
         patch("app.main.minio_client"):
        response = upload(content=b"\xff\xfe not a csv at all \x00")
        print(response.json())
    assert response.status_code == 400

def test_empty_file_returns_400():
    with patch("app.main.check_and_increment_quota", return_value=0), \
         patch("app.main.minio_client"):
        response = upload(content=b"")
    assert response.status_code == 400

def test_no_file_field_returns_422():
    response = client.post("/v1/batch/upload", headers=HEADERS)
    assert response.status_code == 422


# ── Quota ─────────────────────────────────────────────────────────────────────

def test_quota_checked_with_row_count():
    """Quota must be incremented by number of CSV rows, not just 1."""
    with patch("app.main.check_and_increment_quota") as mock_quota, \
         patch("app.main.minio_client") as mock_minio:
        mock_minio.put_object.return_value = None
        upload(content=make_csv(rows=7))
        _, kwargs = mock_quota.call_args
        # Second argument (monthly_limit override or row count) should be 7
        assert mock_quota.call_args[0][1] == 7

def test_exceeded_quota_returns_429():
    from fastapi import HTTPException
    def quota_exceeded(tenant_id, row_count):
        raise HTTPException(status_code=429, detail="Monthly prediction quota exceeded")

    with patch("app.main.check_and_increment_quota", side_effect=quota_exceeded):
        response = upload()
    assert response.status_code == 429


# ── MinIO interaction ─────────────────────────────────────────────────────────

def test_file_written_to_minio():
    with patch("app.main.check_and_increment_quota", return_value=5), \
         patch("app.main.minio_client") as mock_minio:
        mock_minio.put_object.return_value = None
        upload()
        mock_minio.put_object.assert_called_once()

def test_minio_bucket_is_raw_landing():
    with patch("app.main.check_and_increment_quota", return_value=5), \
         patch("app.main.minio_client") as mock_minio:
        mock_minio.put_object.return_value = None
        upload()
        args = mock_minio.put_object.call_args[0]
        assert args[0] == "raw-landing"

def test_minio_failure_returns_500():
    with patch("app.main.check_and_increment_quota", return_value=5), \
         patch("app.main.minio_client") as mock_minio:
        mock_minio.put_object.side_effect = Exception("MinIO connection refused")
        response = upload()
    assert response.status_code == 500
