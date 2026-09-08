import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
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

def test_dag_has_four_tasks():
    mod = load_dag()
    assert len(mod.dag.tasks) == 4

def test_dag_task_order():
    """sync_data → check_psi_drift → train_model → promote_model."""
    mod = load_dag()
    task_ids = [t.task_id for t in mod.dag.topological_sort()]
    assert task_ids == ["sync_data", "check_psi_drift", "train_model", "promote_model"]

def test_check_psi_drift_is_short_circuit():
    """The drift gate must actually be able to stop downstream tasks."""
    from airflow.operators.python import ShortCircuitOperator
    mod = load_dag()
    task = mod.dag.get_task("check_psi_drift")
    assert isinstance(task, ShortCircuitOperator)

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


# ── check_psi_drift (gatekeeper, moved here from the daily ingestion DAG) ──────

class TestCheckPSIDrift:

    def _make_df(self, rng_seed, mean_loan=200_000, n=500):
        rng = np.random.default_rng(rng_seed)
        return pd.DataFrame({
            "loan_amount":  rng.normal(mean_loan, 50_000, n),
            "income":       rng.normal(6_000, 2_000, n),
            "Credit_Score": rng.normal(700, 50, n),
        })

    def _run(self, feedback_df, baseline_df, tmp_path, feedback_exists=True, baseline_exists=True):
        feedback_path = tmp_path / "feedback_labeled.csv"
        baseline_path = tmp_path / "baseline.csv"
        if feedback_exists:
            feedback_df.to_csv(feedback_path, index=False)
        if baseline_exists:
            baseline_df.to_csv(baseline_path, index=False)

        mock_ti = MagicMock()
        mod = load_dag()
        with patch.object(mod, "FEEDBACK_CSV", feedback_path), \
             patch.object(mod, "BASELINE_CSV", baseline_path):
            result = mod.check_psi_drift(ti=mock_ti)
        return result, mock_ti

    def test_skips_when_feedback_missing(self, tmp_path):
        result, _ = self._run(self._make_df(0), self._make_df(1), tmp_path, feedback_exists=False)
        assert result is False

    def test_skips_when_baseline_missing(self, tmp_path):
        result, _ = self._run(self._make_df(0), self._make_df(1), tmp_path, baseline_exists=False)
        assert result is False

    def test_no_drift_on_same_distribution(self, tmp_path):
        df = self._make_df(42)
        result, _ = self._run(df.copy(), df, tmp_path)
        assert result is False

    def test_drift_detected_on_shifted_distribution(self, tmp_path):
        baseline = self._make_df(0, mean_loan=200_000)
        incoming = self._make_df(1, mean_loan=800_000)   # 12σ shift
        result, _ = self._run(incoming, baseline, tmp_path)
        assert result is True

    def test_pushes_drift_report(self, tmp_path):
        df = self._make_df(0)
        _, ti = self._run(df.copy(), df, tmp_path)
        report_push = [c for c in ti.xcom_push.call_args_list if c[1]["key"] == "drift_report"]
        assert len(report_push) == 1
        assert "max_psi" in report_push[0][1]["value"]
