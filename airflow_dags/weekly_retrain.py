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

    def run_dvc_checkout():
        """Ensure we're on the latest DVC-tracked data and code."""
        subprocess.run(["dvc", "checkout"], check=True)

    def run_training():
        """Run the model training script."""
        subprocess.run(
            ["python", "notebooks/retrain.py"],   # headless version of notebook
            check=True
        )


    # Define tasks
    t1 = PythonOperator(task_id="dvc_checkout",   python_callable=run_dvc_checkout)
    t2 = PythonOperator(task_id="train_model",    python_callable=run_training)
    t3 = PythonOperator(task_id="promote_model",  python_callable=promote_if_better)

    # Set task dependencies
    t1 >> t2 >> t3