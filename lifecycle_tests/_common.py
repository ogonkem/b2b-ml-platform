"""
lifecycle_tests/_common.py
────────────────────────────────────────────────────────────────────────────
Shared helpers for the numbered lifecycle stage scripts. Runs on the host
(this repo's .venv) against the docker-compose services on their published
localhost ports — same convention as tests/integration/*.py.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

# .env's values are container-facing (minio/mlflow service hostnames) — every
# lifecycle_tests script runs on the host, so any in-process MLflow/boto3
# client needs these pointed back at the published localhost ports.
os.environ["MLFLOW_TRACKING_URI"]    = "http://localhost:5000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ENDPOINT_URL"]       = "http://localhost:9000"

BASE_URL     = "http://localhost:8000"
MLFLOW_URI   = "http://localhost:5000"
HOLDOUT_DIR  = REPO_ROOT / "notebooks" / "archive" / "holdout"
STATE_FILE   = Path(__file__).resolve().parent / ".state.json"

_raw      = os.environ.get("API_TOKENS", "dev-token")
API_TOKEN = _raw.split(",")[0].strip()
HEADERS   = {"Authorization": f"Bearer {API_TOKEN}"}

EXPERIMENT_NAME = "selastone_loan_default"
MODEL_NAME      = "selastone_credit_scorer"


def require(cond: bool, msg: str):
    """Exit with a clear, actionable message instead of a traceback."""
    if not cond:
        print(f"[BLOCKED] {msg}", file=sys.stderr)
        sys.exit(1)


def require_holdout_files(*names: str):
    missing = [n for n in names if not (HOLDOUT_DIR / n).exists()]
    require(
        not missing,
        f"Missing holdout file(s) {missing} in {HOLDOUT_DIR} — "
        f"run `python notebooks/make_holdout_splits.py` first.",
    )


def require_api_reachable():
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        require(r.status_code == 200, f"/health returned {r.status_code} — is `docker compose up -d` running?")
    except httpx.ConnectError:
        require(False, f"Could not reach {BASE_URL}/health — is `docker compose up -d` running?")


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    return json.loads(STATE_FILE.read_text())


def save_state(**kv):
    state = load_state()
    state.update(kv)
    STATE_FILE.write_text(json.dumps(state, indent=2))
    return state


def require_state(*keys: str) -> dict:
    state = load_state()
    missing = [k for k in keys if k not in state]
    require(
        not missing,
        f"Missing state key(s) {missing} in {STATE_FILE} — run the earlier "
        f"lifecycle_tests stage(s) that produce them first.",
    )
    return state


def score(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Accuracy + AUC for a binary classification result set."""
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    return {
        "n":        len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "auc":      round(roc_auc_score(y_true, y_prob), 4),
    }


def print_score(label: str, result: dict):
    print(f"  {label:<28} n={result['n']:<6} accuracy={result['accuracy']:.4f}  auc={result['auc']:.4f}")


def health() -> dict:
    return httpx.get(f"{BASE_URL}/health", timeout=10.0).json()


def wait_for_model_version(expected_version: str, timeout_s: int = 90, poll_s: int = 5) -> str:
    """Poll /health until model_version matches, or return the last-seen value on timeout."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = health().get("model_version")
        if str(last) == str(expected_version):
            return last
        time.sleep(poll_s)
    return last


def _airflow_exec(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", "exec", "-T", "airflow", "airflow", *args],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


_TERMINAL_STATES = {"success", "failed", "skipped", "upstream_failed"}


def _task_states(dag_id: str, run_id: str) -> list[str]:
    # `airflow dags state` wants an execution_date in this Airflow version,
    # not a run_id — `tasks states-for-dag-run` accepts the run_id directly
    # and gives per-task states, which is what we actually need to poll.
    result = _airflow_exec("tasks", "states-for-dag-run", dag_id, run_id)
    lines = [l for l in result.stdout.splitlines() if l.strip().startswith(dag_id)]
    return [line.split("|")[3].strip() for line in lines]


def trigger_dag_and_wait(dag_id: str, run_id: str, timeout_s: int = 240, poll_s: int = 5) -> str:
    """Unpauses, triggers (with a caller-chosen run_id so it can be polled
    and distinguished from the scheduler's own runs), and waits for every
    task to reach a terminal state. Returns 'success' or 'failed'."""
    _airflow_exec("dags", "unpause", dag_id)

    result = _airflow_exec("dags", "trigger", dag_id, "-r", run_id)
    require(result.returncode == 0, f"airflow dags trigger {dag_id} failed: {result.stderr}")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        states = _task_states(dag_id, run_id)
        if states and all(s in _TERMINAL_STATES for s in states):
            return "failed" if any(s in ("failed", "upstream_failed") for s in states) else "success"
        time.sleep(poll_s)
    require(False, f"DAG run {dag_id}/{run_id} did not reach a terminal state within {timeout_s}s")


def minio_client():
    from minio import Minio
    return Minio(
        "localhost:9000",
        access_key=os.environ.get("MINIO_ROOT_USER"),
        secret_key=os.environ.get("MINIO_ROOT_PASSWORD"),
        secure=False,
    )


def mlflow_client():
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    return mlflow.tracking.MlflowClient()


def banner(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
