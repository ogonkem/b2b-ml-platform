from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import sys
from pathlib import Path

# Make project root importable so promotion.py can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airflow_dags.promotion import promote_if_better

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
        by the daily ingestion DAG when drift was detected) then download the
        actual data files from DVC remote storage.

        dvc checkout only restores files to match the local git state — it does
        not fetch new .dvc pointers or download new data from remote.
        dvc pull = dvc fetch + dvc checkout, and works on the latest git HEAD.
        """
        subprocess.run(["git", "pull", "--ff-only"], check=True)
        subprocess.run(["dvc", "pull"],               check=True)

    def run_training():
        """
        Run the headless retraining script.  retrain.py checks for
        notebooks/archive/feedback_labeled.csv — written by the ingestion DAG
        when PSI drift >= 0.2 — and merges it with the baseline training data
        before fitting the four candidate models.
        """
        subprocess.run(
            ["python", "notebooks/retrain.py"],
            check=True
        )

    # Define tasks
    t1 = PythonOperator(task_id="sync_data",      python_callable=sync_data)
    t2 = PythonOperator(task_id="train_model",    python_callable=run_training)
    t3 = PythonOperator(task_id="promote_model",  python_callable=promote_if_better)

    # Set task dependencies
    t1 >> t2 >> t3