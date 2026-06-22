import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import importlib
import pytest
from unittest.mock import patch, MagicMock


def load_dag():
    """Load the DAG module fresh each time."""
    import airflow_dags.weekly_retrain as mod
    return mod


# ── DAG structure ─────────────────────────────────────────────────────────────

def test_dag_imports_cleanly():
    """DAG file must be importable without errors."""
    mod = load_dag()
    assert mod is not None

def test_dag_has_correct_id():
    mod = load_dag()
    assert mod.dag.dag_id == "selastone_weekly_retrain"

def test_dag_has_three_tasks():
    mod = load_dag()
    assert len(mod.dag.tasks) == 3

def test_dag_task_order():
    """dvc_checkout → train_model → promote_model."""
    mod = load_dag()
    task_ids = [t.task_id for t in mod.dag.topological_sort()]
    assert task_ids == ["dvc_checkout", "train_model", "promote_model"]

def test_dag_schedule():
    mod = load_dag()
    # Airflow 3.x stores schedule not schedule_interval
    assert mod.dag.schedule == "0 2 * * 1"

def test_dag_catchup_disabled():
    mod = load_dag()
    assert mod.dag.catchup is False

def test_dag_has_retry_config():
    mod = load_dag()
    assert mod.dag.default_args["retries"] == 1