"""
lifecycle_tests/05_labeled_data_drift.py
────────────────────────────────────────────────────────────────────────────
Stage 5 — LABELED DATA, drift case. Empties the labeled-data MinIO bucket
(scoping this run's PSI check to exactly this slice), uploads
labeled_data_drift.csv (deliberately the lowest-Credit_Score rows — engineered
to blow PSI past 0.2) via /v1/labeled-data, then triggers the real
selastone_daily_ingestion DAG and confirms it DOES commit: this is the real,
approved side effect — a genuine git commit authored by the mlops bot,
closing the loop that the weekly retrain DAG (stage 6) picks up next.
"""
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    BASE_URL, HEADERS, HOLDOUT_DIR, REPO_ROOT,
    require, require_holdout_files, require_api_reachable,
    trigger_dag_and_wait, minio_client, banner, save_state,
)

DRIFT_FILE = "labeled_data_drift.csv"
DAG_ID     = "selastone_daily_ingestion"


def empty_labeled_data_bucket():
    client = minio_client()
    if not client.bucket_exists("labeled-data"):
        return
    objects = list(client.list_objects("labeled-data", recursive=True))
    for obj in objects:
        client.remove_object("labeled-data", obj.object_name)
    print(f"  Cleared {len(objects)} pre-existing object(s) from labeled-data bucket")


def git(*args):
    import subprocess
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()


def main():
    banner("STAGE 5 — LABELED DATA (drift case)")
    require_holdout_files(DRIFT_FILE)
    require_api_reachable()

    empty_labeled_data_bucket()

    csv_bytes = (HOLDOUT_DIR / DRIFT_FILE).read_bytes()
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{BASE_URL}/v1/labeled-data", headers=HEADERS,
            files={"file": (DRIFT_FILE, csv_bytes, "text/csv")},
        )
    require(resp.status_code == 200, f"labeled-data upload failed: HTTP {resp.status_code}: {resp.text}")
    print(f"  Uploaded {resp.json()['rows_received']} rows -> {resp.json()['object']}")

    head_before = git("rev-parse", "HEAD")
    run_id = f"lifecycle_drift_{int(time.time())}"
    state = trigger_dag_and_wait(DAG_ID, run_id)
    require(state == "success", f"{DAG_ID}/{run_id} finished with state={state}")

    head_after = git("rev-parse", "HEAD")
    require(
        head_after != head_before,
        f"expected a real commit from drift detection but HEAD did not move ({head_before[:8]})",
    )

    commit_msg    = git("log", "-1", "--format=%s")
    commit_author = git("log", "-1", "--format=%an <%ae>")
    feedback_csv  = REPO_ROOT / "notebooks" / "archive" / "feedback_labeled.csv"
    require(feedback_csv.exists(), f"expected {feedback_csv} to exist after commit_to_dvc")

    print(f"  HEAD moved {head_before[:8]} -> {head_after[:8]}")
    print(f"  commit: {commit_msg}")
    print(f"  author: {commit_author}")
    print(f"  {feedback_csv.name} written ({feedback_csv.stat().st_size} bytes)")

    save_state(feedback_commit=head_after)
    print("\n[OK] Stage 5 complete — drift detected and committed for real.")


if __name__ == "__main__":
    main()
