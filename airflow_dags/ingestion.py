"""
airflow_dags/ingestion.py
─────────────────────────────────────────────────────────────────────────────
Daily DAG that runs at 01:00 UTC (one hour before the Monday retrain at 02:00).

Tasks:
  t1  pull_labeled_data   — downloads all CSVs from MinIO labeled-data bucket
                            uploaded by the business via POST /v1/labeled-data
  t2  check_psi_drift     — computes Population Stability Index against the
                            current DVC-tracked training baseline
  t3  commit_to_dvc       — only runs when drift >= 0.2 AND >= 100 labeled rows;
                            saves merged CSV to notebooks/archive/feedback_labeled.csv,
                            dvc add + dvc push, git commit the .dvc pointer

The weekly_retrain DAG (Monday 02:00) then picks up the updated DVC data.
"""

import sys, os, subprocess, json
import pandas as pd
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.drift import check_drift

# ── Constants ─────────────────────────────────────────────────────────────────
PSI_THRESHOLD  = 0.2
MIN_SAMPLES    = 100
FEEDBACK_CSV   = Path("notebooks/archive/feedback_labeled.csv")
BASELINE_CSV   = Path("notebooks/archive/Loan_Default.csv")
MINIO_BUCKET   = "labeled-data"

default_args = {
    "retries":     1,
    "retry_delay": timedelta(minutes=5),
}

# ── Helper: MinIO client ───────────────────────────────────────────────────────
def _get_minio():
    from minio import Minio
    return Minio(
        f"{os.environ.get('MINIO_HOST', 'localhost')}:{os.environ.get('MINIO_PORT', 9000)}",
        access_key=os.environ.get("MINIO_ROOT_USER"),
        secret_key=os.environ.get("MINIO_ROOT_PASSWORD"),
        secure=False,
    )


# ── Tasks ─────────────────────────────────────────────────────────────────────

def pull_labeled_data(**context):
    """
    Downloads every CSV from the MinIO labeled-data bucket, concatenates them
    into one DataFrame, and saves it to /tmp/labeled_pull.csv for the next task.
    Pushes n_samples (0 if nothing found) via XCom.
    """
    minio = _get_minio()

    if not minio.bucket_exists(MINIO_BUCKET):
        print("labeled-data bucket does not exist yet — skipping.")
        context["ti"].xcom_push(key="n_samples", value=0)
        return

    objects = list(minio.list_objects(MINIO_BUCKET, recursive=True))
    csv_objects = [o for o in objects if o.object_name.endswith(".csv")]

    if not csv_objects:
        print("No labeled CSVs found in MinIO — skipping.")
        context["ti"].xcom_push(key="n_samples", value=0)
        return

    frames = []
    for obj in csv_objects:
        response = minio.get_object(MINIO_BUCKET, obj.object_name)
        try:
            df = pd.read_csv(BytesIO(response.read()))
            frames.append(df)
        except Exception as e:
            print(f"  Skipping {obj.object_name}: {e}")

    if not frames:
        context["ti"].xcom_push(key="n_samples", value=0)
        return

    combined = pd.concat(frames, ignore_index=True)
    n = len(combined)
    print(f"Pulled {n} labeled rows from {len(frames)} CSVs")

    tmp_path = "/tmp/labeled_pull.csv"
    combined.to_csv(tmp_path, index=False)

    context["ti"].xcom_push(key="n_samples",    value=n)
    context["ti"].xcom_push(key="labeled_path", value=tmp_path)


def check_psi_drift(**context):
    """
    Loads the labeled pull and the DVC-tracked training CSV, then computes PSI
    on key numeric features.  Pushes drift_detected (bool) and drift_report
    (dict) via XCom.  Skips if fewer than MIN_SAMPLES rows are available.
    """
    ti        = context["ti"]
    n_samples = ti.xcom_pull(task_ids="pull_labeled_data", key="n_samples")

    if not n_samples or n_samples < MIN_SAMPLES:
        print(f"Only {n_samples} labeled rows — need {MIN_SAMPLES}. Skipping PSI check.")
        ti.xcom_push(key="drift_detected", value=False)
        return

    labeled_path = ti.xcom_pull(task_ids="pull_labeled_data", key="labeled_path")
    incoming_df  = pd.read_csv(labeled_path)

    if not BASELINE_CSV.exists():
        print(f"Baseline CSV not found at {BASELINE_CSV} — skipping PSI check.")
        ti.xcom_push(key="drift_detected", value=False)
        return

    baseline_df = pd.read_csv(BASELINE_CSV)
    result      = check_drift(baseline_df, incoming_df)

    print(f"PSI result: max_psi={result['max_psi']:.4f}, drifted={result['drifted']}")
    for col, psi in result["psi_per_col"].items():
        print(f"  {col:<25} PSI={psi:.4f}")

    ti.xcom_push(key="drift_detected", value=result["drifted"])
    ti.xcom_push(key="drift_report",   value=result)


def commit_to_dvc(**context):
    """
    Saves labeled data to a DVC-tracked CSV and commits the pointer file to git.
    Only executes when drift was detected in check_psi_drift.
    The weekly_retrain DAG picks up the updated DVC data on its next run.
    """
    ti             = context["ti"]
    drift_detected = ti.xcom_pull(task_ids="check_psi_drift", key="drift_detected")

    if not drift_detected:
        print("No drift detected — DVC commit skipped.")
        return

    labeled_path = ti.xcom_pull(task_ids="pull_labeled_data", key="labeled_path")
    drift_report = ti.xcom_pull(task_ids="check_psi_drift",   key="drift_report")
    labeled_df   = pd.read_csv(labeled_path)

    # Normalise target column name to match training CSV
    if "actual_outcome" in labeled_df.columns:
        labeled_df = labeled_df.rename(columns={"actual_outcome": "Status"})

    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(str(FEEDBACK_CSV), index=False)
    print(f"Saved {len(labeled_df)} labeled rows → {FEEDBACK_CSV}")

    # DVC track
    subprocess.run(["dvc", "add",  str(FEEDBACK_CSV)], check=True)
    subprocess.run(["dvc", "push"],                     check=True)

    # Commit the updated .dvc pointer file
    dvc_pointer = str(FEEDBACK_CSV) + ".dvc"
    subprocess.run(["git", "add", dvc_pointer], check=True)
    subprocess.run([
        "git", "commit", "-m",
        f"data: labeled feedback — PSI={drift_report['max_psi']:.3f} drift detected",
    ], check=True)

    print(f"DVC commit complete: {drift_report}")


# ── DAG definition ─────────────────────────────────────────────────────────────
with DAG(
    dag_id="selastone_daily_ingestion",
    schedule="0 1 * * *",        # daily 01:00 UTC — 1h before Monday retrain
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
) as dag:

    t1 = PythonOperator(task_id="pull_labeled_data", python_callable=pull_labeled_data)
    t2 = PythonOperator(task_id="check_psi_drift",   python_callable=check_psi_drift)
    t3 = PythonOperator(task_id="commit_to_dvc",     python_callable=commit_to_dvc)

    t1 >> t2 >> t3
