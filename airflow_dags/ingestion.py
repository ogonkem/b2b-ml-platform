"""
airflow_dags/ingestion.py
─────────────────────────────────────────────────────────────────────────────
Daily DAG that runs at 01:00 UTC (one hour before the Monday retrain at 02:00).

Tasks:
  t1  pull_labeled_data   — downloads all CSVs from MinIO labeled-data bucket
                            uploaded by the business via POST /v1/labeled-data
  t2  commit_to_dvc       — unconditionally saves the merged CSV to
                            notebooks/archive/feedback_labeled.csv whenever
                            >= MIN_SAMPLES labeled rows are available, then
                            dvc add + dvc push, git commit the .dvc pointer.
                            PSI drift is no longer computed here — the weekly
                            retrain DAG computes it against whatever this task
                            last committed and gates training on it.

The weekly_retrain DAG (Monday 02:00) then picks up the updated DVC data and
decides whether the drift it sees is worth retraining over.
"""

import sys, os, subprocess
import pandas as pd
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_SAMPLES  = 100
FEEDBACK_CSV = Path("notebooks/archive/feedback_labeled.csv")
MINIO_BUCKET = "labeled-data"

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


def commit_to_dvc(**context):
    """
    Saves labeled data to a DVC-tracked CSV and commits the pointer file to git.
    Runs unconditionally whenever >= MIN_SAMPLES rows were pulled — drift is no
    longer a gate here, it's computed downstream in selastone_weekly_retrain.
    """
    ti        = context["ti"]
    n_samples = ti.xcom_pull(task_ids="pull_labeled_data", key="n_samples")

    if not n_samples or n_samples < MIN_SAMPLES:
        print(f"Only {n_samples} labeled rows — need {MIN_SAMPLES}. DVC commit skipped.")
        return

    labeled_path = ti.xcom_pull(task_ids="pull_labeled_data", key="labeled_path")
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
        f"data: labeled feedback — {len(labeled_df)} rows collected",
    ], check=True)

    print("DVC commit complete.")


# ── DAG definition ─────────────────────────────────────────────────────────────
with DAG(
    dag_id="selastone_daily_ingestion",
    schedule="0 1 * * *",        # daily 01:00 UTC — 1h before Monday retrain
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
) as dag:

    t1 = PythonOperator(task_id="pull_labeled_data", python_callable=pull_labeled_data)
    t2 = PythonOperator(task_id="commit_to_dvc",     python_callable=commit_to_dvc)

    t1 >> t2
