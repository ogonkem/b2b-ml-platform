"""
tests/unit/test_celery_tasks.py
Tests for celery_worker/tasks.py — process_batch task.
All MinIO and Redis calls are mocked; no broker connection is made.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from io import BytesIO
from unittest.mock import patch, MagicMock, call


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def r():
    return MagicMock()


@pytest.fixture
def minio():
    m = MagicMock()
    m.bucket_exists.return_value = True
    return m


@pytest.fixture
def sample_csv():
    return pd.DataFrame({
        "ID":             [1, 2, 3],
        "loan_amount":    [200_000.0, 300_000.0, 150_000.0],
        "property_value": [300_000.0, 400_000.0, 200_000.0],
        "income":         [6_000.0,   8_000.0,   5_000.0],
        "Credit_Score":   [720.0,     780.0,     650.0],
    }).to_csv(index=False).encode()


@pytest.fixture
def mock_model():
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]])
    return m


@pytest.fixture
def mock_pipeline():
    p = MagicMock()
    p.transform.return_value = np.zeros((3, 14))
    return p


def run_task(r, minio, sample_csv, mock_model, mock_pipeline,
             job_id="job-1", object_name="t/d/job-1.csv", tenant="tenant"):
    """Helper: run the task with all external deps mocked."""
    minio.get_object.return_value = MagicMock(read=lambda: sample_csv)

    with patch("celery_worker.tasks._redis",                   return_value=r), \
         patch("celery_worker.tasks._minio",                   return_value=minio), \
         patch("celery_worker.tasks._load_model_and_pipeline", return_value=(mock_model, mock_pipeline)):
        from celery_worker.tasks import process_batch
        process_batch.run(job_id, object_name, tenant)


# ── Status tracking ───────────────────────────────────────────────────────────

def test_sets_processing_status_first(r, minio, sample_csv, mock_model, mock_pipeline):
    run_task(r, minio, sample_csv, mock_model, mock_pipeline)
    first_status_call = [c for c in r.set.call_args_list if "status" in c[0][0]][0]
    assert first_status_call[0][1] == "processing"

def test_sets_complete_status_on_success(r, minio, sample_csv, mock_model, mock_pipeline):
    run_task(r, minio, sample_csv, mock_model, mock_pipeline)
    status_calls  = [c for c in r.set.call_args_list if "status" in c[0][0]]
    final_status  = status_calls[-1][0][1]
    assert final_status == "complete"

def test_records_rows_scored(r, minio, sample_csv, mock_model, mock_pipeline):
    run_task(r, minio, sample_csv, mock_model, mock_pipeline)
    rows_call = [c for c in r.set.call_args_list if "rows_scored" in c[0][0]]
    assert rows_call[0][0][1] == "3"

def test_records_result_object_key(r, minio, sample_csv, mock_model, mock_pipeline):
    run_task(r, minio, sample_csv, mock_model, mock_pipeline, job_id="my-job")
    obj_call = [c for c in r.set.call_args_list if "result_object" in c[0][0]]
    assert len(obj_call) == 1
    assert "my-job_results.csv" in obj_call[0][0][1]

def test_redis_keys_use_job_id(r, minio, sample_csv, mock_model, mock_pipeline):
    run_task(r, minio, sample_csv, mock_model, mock_pipeline, job_id="unique-job-99")
    all_keys = [c[0][0] for c in r.set.call_args_list]
    assert all("unique-job-99" in k for k in all_keys)

def test_all_redis_sets_have_ttl(r, minio, sample_csv, mock_model, mock_pipeline):
    run_task(r, minio, sample_csv, mock_model, mock_pipeline)
    for c in r.set.call_args_list:
        assert c[1].get("ex") == 86400


# ── MinIO reads and writes ────────────────────────────────────────────────────

def test_reads_from_raw_landing_bucket(r, minio, sample_csv, mock_model, mock_pipeline):
    minio.get_object.return_value = MagicMock(read=lambda: sample_csv)
    with patch("celery_worker.tasks._redis",                   return_value=r), \
         patch("celery_worker.tasks._minio",                   return_value=minio), \
         patch("celery_worker.tasks._load_model_and_pipeline", return_value=(mock_model, mock_pipeline)):
        from celery_worker.tasks import process_batch
        process_batch.run("j", "tenant/d/j.csv", "tenant")
    assert minio.get_object.call_args[0][0] == "raw-landing"

def test_writes_to_batch_results_bucket(r, minio, sample_csv, mock_model, mock_pipeline):
    run_task(r, minio, sample_csv, mock_model, mock_pipeline)
    assert minio.put_object.call_args[0][0] == "batch-results"

def test_creates_batch_results_bucket_if_missing(r, minio, sample_csv, mock_model, mock_pipeline):
    minio.bucket_exists.return_value = False
    run_task(r, minio, sample_csv, mock_model, mock_pipeline)
    minio.make_bucket.assert_called_once_with("batch-results")

def test_does_not_create_bucket_if_already_exists(r, minio, sample_csv, mock_model, mock_pipeline):
    minio.bucket_exists.return_value = True
    run_task(r, minio, sample_csv, mock_model, mock_pipeline)
    minio.make_bucket.assert_not_called()

def test_result_object_path_contains_tenant(r, minio, sample_csv, mock_model, mock_pipeline):
    run_task(r, minio, sample_csv, mock_model, mock_pipeline, tenant="acme-corp")
    obj_call = [c for c in r.set.call_args_list if "result_object" in c[0][0]]
    assert "acme-corp" in obj_call[0][0][1]


# ── Scoring output ────────────────────────────────────────────────────────────

def test_pipeline_transform_called_once(r, minio, sample_csv, mock_model, mock_pipeline):
    run_task(r, minio, sample_csv, mock_model, mock_pipeline)
    mock_pipeline.transform.assert_called_once()

def test_model_predict_proba_called_once(r, minio, sample_csv, mock_model, mock_pipeline):
    run_task(r, minio, sample_csv, mock_model, mock_pipeline)
    mock_model.predict_proba.assert_called_once()

def test_csv_without_id_column_completes_successfully(r, minio, mock_model, mock_pipeline):
    csv_no_id = pd.DataFrame({
        "loan_amount":    [250_000.0],
        "income":         [6_000.0],
        "Credit_Score":   [720.0],
        "property_value": [320_000.0],
    }).to_csv(index=False).encode()
    mock_pipeline.transform.return_value = np.zeros((1, 14))
    mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
    run_task(r, minio, csv_no_id, mock_model, mock_pipeline)
    status_calls = [c for c in r.set.call_args_list if "status" in c[0][0]]
    assert status_calls[-1][0][1] == "complete"


# ── Error handling ────────────────────────────────────────────────────────────

def test_sets_failed_status_on_minio_read_error(r, minio):
    minio.get_object.side_effect = Exception("MinIO unreachable")
    with patch("celery_worker.tasks._redis",  return_value=r), \
         patch("celery_worker.tasks._minio",  return_value=minio), \
         patch("celery_worker.tasks.process_batch.retry", side_effect=Exception("no-retry")):
        from celery_worker.tasks import process_batch
        try:
            process_batch.run("fail-job", "obj", "tenant")
        except Exception:
            pass
    status_calls = [c for c in r.set.call_args_list if "status" in c[0][0]]
    assert any(c[0][1] == "failed" for c in status_calls)

def test_records_error_message_on_failure(r, minio):
    minio.get_object.side_effect = Exception("bucket not found")
    with patch("celery_worker.tasks._redis",  return_value=r), \
         patch("celery_worker.tasks._minio",  return_value=minio), \
         patch("celery_worker.tasks.process_batch.retry", side_effect=Exception("no-retry")):
        from celery_worker.tasks import process_batch
        try:
            process_batch.run("fail-job", "obj", "tenant")
        except Exception:
            pass
    error_calls = [c for c in r.set.call_args_list if "error" in c[0][0]]
    assert len(error_calls) == 1
    assert "bucket not found" in error_calls[0][0][1]
