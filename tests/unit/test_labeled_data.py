"""
tests/unit/test_labeled_data.py
Tests for POST /v1/labeled-data in app/main.py.
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_labeled_csv(rows: int = 5, target_col: str = "actual_outcome") -> bytes:
    lines = [f"ID,year,loan_amount,property_value,income,Credit_Score,{target_col}"]
    for i in range(1, rows + 1):
        lines.append(f"{i},2023,250000,320000,6000,720,{i % 2}")
    return "\n".join(lines).encode()


def upload(content=None, headers=None):
    file_content = make_labeled_csv() if content is None else content
    return client.post(
        "/v1/labeled-data",
        headers=headers or HEADERS,
        files={"file": ("labeled.csv", file_content, "text/csv")},
    )


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_no_token_returns_401():
    response = client.post(
        "/v1/labeled-data",
        files={"file": ("f.csv", make_labeled_csv(), "text/csv")}
    )
    assert response.status_code == 401

def test_wrong_token_returns_403():
    with patch("app.main.minio_client") as m, \
         patch("app.main.log_labeled_upload"):
        m.bucket_exists.return_value = True
        response = upload(headers={"Authorization": "Bearer wrong-token"})
    assert response.status_code == 403


# ── Happy path ────────────────────────────────────────────────────────────────

def test_valid_csv_with_actual_outcome_returns_200():
    with patch("app.main.minio_client") as m, \
         patch("app.main.log_labeled_upload"):
        m.bucket_exists.return_value = True
        m.put_object.return_value    = None
        assert upload().status_code == 200

def test_valid_csv_with_status_col_returns_200():
    with patch("app.main.minio_client") as m, \
         patch("app.main.log_labeled_upload"):
        m.bucket_exists.return_value = True
        m.put_object.return_value    = None
        assert upload(content=make_labeled_csv(target_col="Status")).status_code == 200

def test_response_schema():
    with patch("app.main.minio_client") as m, \
         patch("app.main.log_labeled_upload"):
        m.bucket_exists.return_value = True
        m.put_object.return_value    = None
        body = upload().json()
    assert {"object", "tenant_id", "rows_received", "status"} <= set(body.keys())

def test_status_is_stored():
    with patch("app.main.minio_client") as m, \
         patch("app.main.log_labeled_upload"):
        m.bucket_exists.return_value = True
        m.put_object.return_value    = None
        assert upload().json()["status"] == "stored"

def test_row_count_matches_csv():
    with patch("app.main.minio_client") as m, \
         patch("app.main.log_labeled_upload"):
        m.bucket_exists.return_value = True
        m.put_object.return_value    = None
        assert upload(content=make_labeled_csv(rows=12)).json()["rows_received"] == 12

def test_tenant_id_matches_token():
    with patch("app.main.minio_client") as m, \
         patch("app.main.log_labeled_upload"):
        m.bucket_exists.return_value = True
        m.put_object.return_value    = None
        assert upload().json()["tenant_id"] == "dev-token"

def test_object_path_contains_tenant():
    with patch("app.main.minio_client") as m, \
         patch("app.main.log_labeled_upload"):
        m.bucket_exists.return_value = True
        m.put_object.return_value    = None
        assert "dev-token" in upload().json()["object"]

def test_object_path_ends_with_labeled_csv():
    with patch("app.main.minio_client") as m, \
         patch("app.main.log_labeled_upload"):
        m.bucket_exists.return_value = True
        m.put_object.return_value    = None
        assert upload().json()["object"].endswith("_labeled.csv")


# ── Input validation ──────────────────────────────────────────────────────────

def test_empty_file_returns_400():
    assert upload(content=b"").status_code == 400

def test_invalid_csv_returns_400():
    assert upload(content=b"\xff\xfe not csv \x00").status_code == 400

def test_csv_without_target_column_returns_400():
    content = b"ID,loan_amount,income\n1,250000,6000\n"
    assert upload(content=content).status_code == 400

def test_no_file_field_returns_422():
    assert client.post("/v1/labeled-data", headers=HEADERS).status_code == 422


# ── MinIO interaction ─────────────────────────────────────────────────────────

def test_bucket_created_when_missing():
    with patch("app.main.minio_client") as m, \
         patch("app.main.log_labeled_upload"):
        m.bucket_exists.return_value = False
        m.put_object.return_value    = None
        upload()
        m.make_bucket.assert_called_once_with("labeled-data")

def test_bucket_not_created_when_exists():
    with patch("app.main.minio_client") as m, \
         patch("app.main.log_labeled_upload"):
        m.bucket_exists.return_value = True
        m.put_object.return_value    = None
        upload()
        m.make_bucket.assert_not_called()

def test_writes_to_labeled_data_bucket():
    with patch("app.main.minio_client") as m, \
         patch("app.main.log_labeled_upload"):
        m.bucket_exists.return_value = True
        m.put_object.return_value    = None
        upload()
        assert m.put_object.call_args[0][0] == "labeled-data"

def test_minio_failure_returns_500():
    with patch("app.main.minio_client") as m:
        m.bucket_exists.return_value = True
        m.put_object.side_effect     = Exception("MinIO down")
        assert upload().status_code == 500

def test_audit_log_called_on_success():
    with patch("app.main.minio_client") as m, \
         patch("app.main.log_labeled_upload") as mock_log:
        m.bucket_exists.return_value = True
        m.put_object.return_value    = None
        upload()
        mock_log.assert_called_once()

def test_audit_log_receives_row_count():
    with patch("app.main.minio_client") as m, \
         patch("app.main.log_labeled_upload") as mock_log:
        m.bucket_exists.return_value = True
        m.put_object.return_value    = None
        upload(content=make_labeled_csv(rows=8))
        _, _, rows = mock_log.call_args[0]
        assert rows == 8
