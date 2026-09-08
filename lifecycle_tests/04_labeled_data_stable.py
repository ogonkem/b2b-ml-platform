"""
lifecycle_tests/04_labeled_data_stable.py
────────────────────────────────────────────────────────────────────────────
Stage 4 — LABELED DATA, stable case. Empties the labeled-data MinIO bucket
(local-dev only), uploads labeled_data_stable.csv (a plain random sample — no
engineered drift) via /v1/labeled-data, then triggers the real
selastone_daily_ingestion DAG and confirms it DOES commit: the daily DAG
collects unconditionally now — PSI is no longer computed here at all, so
"stable" data is committed to DVC exactly like drifted data would be. The
negative control for drift now lives in stage 6's check_psi_drift gate: this
stage's commit becomes the feedback_labeled.csv that stage 5 immediately
overwrites, so stage 6 never actually sees this stable slice — it only
demonstrates that collection itself doesn't care about drift.
"""
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    BASE_URL, HEADERS, HOLDOUT_DIR, REPO_ROOT,
    require, require_holdout_files, require_api_reachable,
    trigger_dag_and_wait, minio_client, banner,
)

STABLE_FILE = "labeled_data_stable.csv"
DAG_ID      = "selastone_daily_ingestion"


def empty_labeled_data_bucket():
    client = minio_client()
    if not client.bucket_exists("labeled-data"):
        return
    objects = list(client.list_objects("labeled-data", recursive=True))
    for obj in objects:
        client.remove_object("labeled-data", obj.object_name)
    print(f"  Cleared {len(objects)} pre-existing object(s) from labeled-data bucket")


def git_head():
    import subprocess
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()


def main():
    banner("STAGE 4 — LABELED DATA (stable, unconditional collection)")
    require_holdout_files(STABLE_FILE)
    require_api_reachable()

    empty_labeled_data_bucket()

    csv_bytes = (HOLDOUT_DIR / STABLE_FILE).read_bytes()
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{BASE_URL}/v1/labeled-data", headers=HEADERS,
            files={"file": (STABLE_FILE, csv_bytes, "text/csv")},
        )
    require(resp.status_code == 200, f"labeled-data upload failed: HTTP {resp.status_code}: {resp.text}")
    print(f"  Uploaded {resp.json()['rows_received']} rows -> {resp.json()['object']}")

    head_before = git_head()
    run_id = f"lifecycle_stable_{int(time.time())}"
    state = trigger_dag_and_wait(DAG_ID, run_id)
    require(state == "success", f"{DAG_ID}/{run_id} finished with state={state}")

    head_after = git_head()
    require(
        head_after != head_before,
        f"expected a real commit from unconditional collection but HEAD did not move ({head_before[:8]})",
    )
    print(f"  HEAD moved {head_before[:8]} -> {head_after[:8]} — stable data committed regardless of drift")
    print("\n[OK] Stage 4 complete.")


if __name__ == "__main__":
    main()
