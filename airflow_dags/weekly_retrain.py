from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import pandas as pd

# Make project root importable so promotion.py / shared/drift.py can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airflow_dags.promotion import promote_if_better
from shared.drift import check_drift

# ── Constants ─────────────────────────────────────────────────────────────────
FEEDBACK_CSV = Path("notebooks/archive/feedback_labeled.csv")
BASELINE_CSV = Path("notebooks/archive/Loan_Default.csv")

# define args for retries
default_args = {
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

# DAG (Directed Acyclic Graph) definition
with DAG(
    dag_id="selastone_weekly_retrain",
    schedule="0 2 * * 1",   # every Monday 02:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
) as dag:

    def sync_data():
        """
        Pull the latest git commits (including any .dvc pointer files committed
        unconditionally by the daily ingestion DAG whenever it collected enough
        labeled rows) then download the actual data files from DVC remote
        storage.

        dvc checkout only restores files to match the local git state — it does
        not fetch new .dvc pointers or download new data from remote.
        dvc pull = dvc fetch + dvc checkout, and works on the latest git HEAD.
        """
        # Explicit remote/branch — a bare `git pull` depends on the checkout
        # having upstream tracking configured for the current branch, which
        # isn't guaranteed on every machine that runs this DAG.
        subprocess.run(["git", "pull", "--ff-only", "origin", "main"], check=True)
        subprocess.run(["dvc", "pull"],                                check=True)

    def check_psi_drift(**context):
        """
        Gatekeeper: computes PSI between the DVC-tracked training baseline and
        whatever the daily ingestion DAG has most recently committed to
        notebooks/archive/feedback_labeled.csv, now that sync_data has pulled
        the latest git/DVC state.

        Returns True (proceed to train_model/promote_model) only when
        significant drift is detected — this is what a ShortCircuitOperator
        needs to decide whether to skip the rest of the DAG. Retraining on
        every unconditional daily commit regardless of drift would burn
        compute for no accuracy gain, so this is where that decision now
        lives (moved out of the daily ingestion DAG, which just collects data
        unconditionally).
        """
        if not FEEDBACK_CSV.exists():
            print(f"No feedback CSV at {FEEDBACK_CSV} — nothing new to check, skipping retrain.")
            return False

        if not BASELINE_CSV.exists():
            print(f"Baseline CSV not found at {BASELINE_CSV} — skipping PSI check, skipping retrain.")
            return False

        baseline_df = pd.read_csv(BASELINE_CSV)
        incoming_df = pd.read_csv(FEEDBACK_CSV)
        result      = check_drift(baseline_df, incoming_df)

        print(f"PSI result: max_psi={result['max_psi']:.4f}, drifted={result['drifted']}")
        for col, psi in result["psi_per_col"].items():
            print(f"  {col:<25} PSI={psi:.4f}")

        context["ti"].xcom_push(key="drift_report", value=result)

        if not result["drifted"]:
            print("No significant drift — skipping retrain this week.")
        return result["drifted"]

    def run_training():
        """
        Run the headless retraining script.  retrain.py checks for
        notebooks/archive/feedback_labeled.csv — written unconditionally by the
        ingestion DAG whenever it collects enough labeled rows — and merges it
        with the baseline training data before fitting the four candidate
        models. Only reached when check_psi_drift's gate passed.
        """
        subprocess.run(
            ["python", "notebooks/retrain.py"],
            check=True
        )

    # Define tasks
    t1 = PythonOperator(task_id="sync_data",        python_callable=sync_data)
    t2 = ShortCircuitOperator(task_id="check_psi_drift", python_callable=check_psi_drift)
    t3 = PythonOperator(task_id="train_model",      python_callable=run_training)
    t4 = PythonOperator(task_id="promote_model",    python_callable=promote_if_better)

    # Set task dependencies — check_psi_drift gates train_model/promote_model
    t1 >> t2 >> t3 >> t4
