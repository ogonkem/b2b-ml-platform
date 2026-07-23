# Manual Test Run Instructions

Run from the project root. Activate the venv first:

```
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

---

## Unit Tests — run one file at a time

### 1. Feature pipeline
```
pytest tests/unit/test_features.py -v
```
Tests FeaturePipeline.fit() and transform() — medians, modes, derived features,
missing value handling, determinism.

### 2. Model manager (hot-swap daemon)
```
pytest tests/unit/test_model_manager.py -v
```
Tests MLflow polling, thread-safe hot-swap, predict_proba delegation.
MLflow client is fully mocked — no server required.

### 3. Authentication
```
pytest tests/unit/test_auth.py -v
```
Tests every token/scheme permutation on POST /v1/predict.

### 4. Predict endpoint
```
pytest tests/unit/test_predict.py -v
```
Tests response schema, required fields, quota enforcement, logging, edge values.

### 5. Batch upload endpoint
```
pytest tests/unit/test_batch_upload.py -v
```
Tests file validation, MinIO write, quota, Redis job tracking, Celery enqueue.
Includes new tests that verify bucket auto-creation and Celery failure tolerance.

### 6. Labeled data endpoint  ← NEW
```
pytest tests/unit/test_labeled_data.py -v
```
Tests POST /v1/labeled-data — auth, target column validation, MinIO bucket
creation, audit log, row count.

### 7. Batch results endpoint  ← NEW
```
pytest tests/unit/test_batch_results.py -v
```
Tests GET /v1/batch/results/{job_id} — 404 for unknown jobs, tenant isolation,
queued/processing/complete/failed status shapes, presigned URL generation.

### 8. Celery batch scoring task  ← NEW
```
pytest tests/unit/test_celery_tasks.py -v
```
Tests process_batch task — Redis status lifecycle (processing → complete/failed),
MinIO reads and writes, bucket auto-creation, error recording, ID-less CSV handling.
No broker connection — all external deps mocked.

### 9. PSI drift detection  ← NEW
```
pytest tests/unit/test_drift.py -v
```
Tests compute_psi and check_drift — stable vs shifted distributions, column
skipping, custom thresholds, edge cases (constant arrays, no shared columns).

### 10. Weekly retrain DAG
```
pytest tests/unit/test_weekly_retrain_dag.py -v
```
Tests DAG structure: 3 tasks, sync_data → train_model → promote_model order,
Monday 02:00 schedule, catchup=False.

### 11. Model promotion logic
```
pytest tests/unit/test_promotion.py -v
```
Tests promote_if_better — threshold gate (>=2% AUC improvement), edge cases
(one run, no runs, exact threshold, challenger worse).

### 12. Daily ingestion DAG  ← NEW
```
pytest tests/unit/test_ingestion_dag.py -v
```
Tests DAG structure and all three task functions — pull_labeled_data (MinIO
listing and concatenation), check_psi_drift (sample count guard, baseline
missing guard, drift detection), commit_to_dvc (subprocess sequencing, column
rename, skip when no drift).

---

## Run all unit tests together
```
pytest tests/unit/ -v
```

---

## Integration Tests — requires docker stack

Start the stack first and wait for containers to be healthy (~60 seconds):
```
docker compose up -d
```

Check health:
```
curl http://localhost:8000/health
```

### 13. API endpoints (predict + quota)
```
pytest tests/integration/test_api_endpoints.py -v
```

### 14. Batch pipeline (upload + status)
```
pytest tests/integration/test_batch_pipeline.py -v
```
Note: this file targets port 8001 — confirm your API is on the correct port.

### 15. MLOps loop (MLflow reachability, model registration, hot-swap)
```
pytest tests/integration/test_mlops_loop.py -v -s
```
Note: test_model_manager_hot_swap sleeps 65s waiting for a poll cycle.

### 16. Labeled data pipeline  ← NEW
```
pytest tests/integration/test_labeled_data_pipeline.py -v
```
Tests full POST /v1/labeled-data flow and GET /v1/batch/results/{job_id} polling
against the live stack. Requires MinIO labeled-data bucket to be writable.

---

## Quick smoke run (unit only, no docker needed)
```
pytest tests/unit/ -q --tb=short
```
