"""
tests/unit/test_ingestion_dag.py
Tests for airflow_dags/ingestion.py — DAG structure and task logic.
All external services (MinIO, subprocess) are mocked.

PSI drift is no longer computed in this DAG — see test_weekly_retrain_dag.py
for TestCheckPSIDrift, which now lives in selastone_weekly_retrain as the
gatekeeper for training. This DAG only ever decides whether it collected
enough rows to be worth committing (MIN_SAMPLES), never whether they drifted.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, call


def load_dag():
    import airflow_dags.ingestion as mod
    return mod


# ── DAG structure ─────────────────────────────────────────────────────────────

def test_dag_imports_cleanly():
    assert load_dag() is not None

def test_dag_id():
    assert load_dag().dag.dag_id == "selastone_daily_ingestion"

def test_dag_has_two_tasks():
    assert len(load_dag().dag.tasks) == 2

def test_dag_task_order():
    task_ids = [t.task_id for t in load_dag().dag.topological_sort()]
    assert task_ids == ["pull_labeled_data", "commit_to_dvc"]

def test_dag_runs_daily_at_0100():
    assert load_dag().dag.schedule == "0 1 * * *"

def test_dag_catchup_disabled():
    assert load_dag().dag.catchup is False

def test_dag_has_retry_config():
    assert load_dag().dag.default_args["retries"] == 1


# ── pull_labeled_data ─────────────────────────────────────────────────────────

class TestPullLabeledData:

    CSV_BYTES = b"ID,loan_amount,actual_outcome\n1,200000,0\n2,300000,1\n"

    def _run(self, bucket_exists=True, csv_bytes=None, n_objects=1):
        csv = csv_bytes or self.CSV_BYTES
        mock_obj       = MagicMock()
        mock_obj.object_name = "tenant/20240101/data_labeled.csv"
        mock_minio     = MagicMock()
        mock_minio.bucket_exists.return_value = bucket_exists
        mock_minio.list_objects.return_value  = [mock_obj] * n_objects
        mock_minio.get_object.return_value    = MagicMock(read=lambda: csv)
        mock_ti = MagicMock()
        with patch("airflow_dags.ingestion._get_minio", return_value=mock_minio):
            from airflow_dags.ingestion import pull_labeled_data
            pull_labeled_data(ti=mock_ti)
        return mock_ti, mock_minio

    def test_pushes_zero_when_bucket_missing(self):
        ti, _ = self._run(bucket_exists=False)
        n_pushes = [c for c in ti.xcom_push.call_args_list if c[1]["key"] == "n_samples"]
        assert n_pushes[-1][1]["value"] == 0

    def test_pushes_zero_when_no_csvs(self):
        ti, _ = self._run(n_objects=0)
        n_pushes = [c for c in ti.xcom_push.call_args_list if c[1]["key"] == "n_samples"]
        assert n_pushes[-1][1]["value"] == 0

    def test_pushes_correct_sample_count(self):
        ti, _ = self._run(n_objects=1)    # CSV has 2 data rows
        n_pushes = [c for c in ti.xcom_push.call_args_list if c[1]["key"] == "n_samples"]
        assert n_pushes[-1][1]["value"] == 2

    def test_pushes_labeled_path_on_success(self):
        ti, _ = self._run(n_objects=1)
        path_pushes = [c for c in ti.xcom_push.call_args_list if c[1]["key"] == "labeled_path"]
        assert len(path_pushes) == 1

    def test_concatenates_multiple_files(self):
        ti, _ = self._run(n_objects=3)     # 3 CSVs × 2 rows = 6 rows
        n_pushes = [c for c in ti.xcom_push.call_args_list if c[1]["key"] == "n_samples"]
        assert n_pushes[-1][1]["value"] == 6

    def test_reads_from_labeled_data_bucket(self):
        _, mock_minio = self._run(n_objects=1)
        assert mock_minio.get_object.call_args[0][0] == "labeled-data"


# ── commit_to_dvc ─────────────────────────────────────────────────────────────
# Unconditional collection: this task no longer reads a drift_detected XCom at
# all — it commits whenever pull_labeled_data cleared MIN_SAMPLES, regardless
# of whether that data has drifted from baseline.

class TestCommitToDVC:

    def _run(self, n_samples, tmp_path):
        labeled_path = tmp_path / "labeled.csv"
        pd.DataFrame({
            "loan_amount":    [200_000] * max(n_samples, 1),
            "actual_outcome": [1] * max(n_samples, 1),
        }).to_csv(labeled_path, index=False)

        mock_ti = MagicMock()
        def xcom_side(task_ids, key):
            if key == "n_samples":    return n_samples
            if key == "labeled_path": return str(labeled_path)
        mock_ti.xcom_pull.side_effect = xcom_side

        feedback_path = tmp_path / "feedback_labeled.csv"
        with patch("subprocess.run") as mock_sub, \
             patch("airflow_dags.ingestion.FEEDBACK_CSV", feedback_path):
            from airflow_dags.ingestion import commit_to_dvc
            commit_to_dvc(ti=mock_ti)
        return mock_sub, feedback_path

    def test_skips_all_subprocess_when_below_min_samples(self, tmp_path):
        mock_sub, _ = self._run(n_samples=50, tmp_path=tmp_path)
        mock_sub.assert_not_called()

    def test_runs_dvc_push_when_above_min_samples(self, tmp_path):
        mock_sub, _ = self._run(n_samples=500, tmp_path=tmp_path)
        commands = [c[0][0] for c in mock_sub.call_args_list]
        assert ["dvc", "push"] in commands

    def test_runs_dvc_add_when_above_min_samples(self, tmp_path):
        mock_sub, feedback_path = self._run(n_samples=500, tmp_path=tmp_path)
        commands = [c[0][0] for c in mock_sub.call_args_list]
        assert any(c[:2] == ["dvc", "add"] for c in commands)

    def test_runs_git_commit_when_above_min_samples(self, tmp_path):
        mock_sub, _ = self._run(n_samples=500, tmp_path=tmp_path)
        commands = [c[0][0] for c in mock_sub.call_args_list]
        assert any(c[:2] == ["git", "commit"] for c in commands)

    def test_saves_feedback_csv_when_above_min_samples(self, tmp_path):
        _, feedback_path = self._run(n_samples=500, tmp_path=tmp_path)
        assert feedback_path.exists()

    def test_renames_actual_outcome_to_status(self, tmp_path):
        _, feedback_path = self._run(n_samples=500, tmp_path=tmp_path)
        saved = pd.read_csv(feedback_path)
        assert "Status" in saved.columns
        assert "actual_outcome" not in saved.columns

    def test_dvc_commands_run_before_git_commit(self, tmp_path):
        mock_sub, _ = self._run(n_samples=500, tmp_path=tmp_path)
        commands = [tuple(c[0][0]) for c in mock_sub.call_args_list]
        git_idx = next(i for i, c in enumerate(commands) if c[:2] == ("git", "commit"))
        dvc_idx = next(i for i, c in enumerate(commands) if c[:2] == ("dvc", "push"))
        assert dvc_idx < git_idx
