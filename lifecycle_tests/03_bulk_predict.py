"""
lifecycle_tests/03_bulk_predict.py
────────────────────────────────────────────────────────────────────────────
Stage 3 — BULK PREDICT. celery_worker/tasks.py loads models/best_model.pkl +
feature_pipeline.pkl into a module-global on first use and never reloads, so
it needs a restart to pick up what stage 1 just trained. Then uploads
bulk_predict_holdout.csv to /v1/batch/upload, polls for completion,
downloads the result CSV straight from MinIO, and scores it against
bulk_predict_answer_key.csv — a cross-path sanity check against stage 2's
single-predict numbers (same model, different serving path).
"""
import io
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pandas as pd
from minio import Minio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    BASE_URL, HEADERS, HOLDOUT_DIR, REPO_ROOT,
    require, require_holdout_files, require_api_reachable,
    score, print_score, banner, minio_client,
)

BULK_FILE   = "bulk_predict_holdout.csv"
ANSWER_FILE = "bulk_predict_answer_key.csv"
# /v1/batch/upload counts every row against the tenant's 1,000-row/month quota
# (see check_and_increment_quota in app/main.py) in one shot — a single
# request can never use the full 2,000-row reserve, so take a safe subset.
UPLOAD_ROWS = 900


def restart_celery_worker():
    print("Restarting celery_worker so it reloads the freshly trained model...")
    result = subprocess.run(
        ["docker", "compose", "restart", "celery_worker"],
        cwd=REPO_ROOT,
    )
    require(result.returncode == 0, "docker compose restart celery_worker failed.")
    time.sleep(5)  # give the worker a moment to reconnect to Redis/Celery broker


def upload_and_wait(csv_bytes: bytes) -> dict:
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{BASE_URL}/v1/batch/upload",
            headers=HEADERS,
            files={"file": (BULK_FILE, csv_bytes, "text/csv")},
        )
        require(resp.status_code == 200, f"batch upload failed: HTTP {resp.status_code}: {resp.text}")
        job_id = resp.json()["job_id"]
        print(f"  job_id = {job_id}")

        deadline = time.time() + 180
        while time.time() < deadline:
            r = client.get(f"{BASE_URL}/v1/batch/results/{job_id}", headers=HEADERS)
            require(r.status_code == 200, f"results poll failed: HTTP {r.status_code}: {r.text}")
            body = r.json()
            if body["status"] == "complete":
                return body
            require(body["status"] != "failed", f"batch job failed: {body.get('error')}")
            time.sleep(3)

        require(False, f"batch job {job_id} did not complete within 180s")


def main():
    banner("STAGE 3 — BULK PREDICT on bulk_predict_holdout.csv")
    require_holdout_files(BULK_FILE, ANSWER_FILE)
    require_api_reachable()

    restart_celery_worker()

    upload_df = pd.read_csv(HOLDOUT_DIR / BULK_FILE).head(UPLOAD_ROWS)
    csv_bytes = upload_df.to_csv(index=False).encode()
    print(f"  Uploading {len(upload_df)} rows (quota-safe subset of the 2,000-row reserve)")
    result = upload_and_wait(csv_bytes)
    print(f"  rows_scored = {result['rows_scored']}")

    # The presigned URL is signed for the container-internal host (minio:9000)
    # and a hostname swap breaks the signature — fetch the object directly
    # with our own host-facing MinIO client instead, using the object path
    # out of the presigned URL.
    object_name = urlparse(result["download_url"]).path.split("/batch-results/", 1)[1]
    response = minio_client().get_object("batch-results", object_name)
    results_df = pd.read_csv(io.BytesIO(response.read()))

    answers_df = pd.read_csv(HOLDOUT_DIR / ANSWER_FILE)
    merged = results_df.merge(
        answers_df, left_on="application_id", right_on="ID", how="inner"
    )
    require(
        len(merged) == len(upload_df),
        f"expected {len(upload_df)} scored rows to match the answer key, got {len(merged)}",
    )

    result = score(merged["Status"], merged["default_probability"])
    print_score("bulk_predict_holdout", result)
    print("\n[OK] Stage 3 complete.")


if __name__ == "__main__":
    main()
