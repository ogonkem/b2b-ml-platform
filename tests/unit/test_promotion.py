import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from unittest.mock import patch, MagicMock


def make_run(auc: float, run_id: str):
    run = MagicMock()
    run.info.run_id = run_id
    run.data.metrics = {"test_auc": auc}
    return run


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    mock_mv         = MagicMock()
    mock_mv.version = "5"
    client.create_model_version.return_value = mock_mv
    return client


# ── Promotion logic ───────────────────────────────────────────────────────────

def test_promotes_when_challenger_is_better(mock_client):
    """0.88 vs 0.85 = 0.03 improvement >= 0.02 threshold → promote."""
    mock_client.search_runs.return_value = [
        make_run(auc=0.88, run_id="run_challenger"),
        make_run(auc=0.85, run_id="run_champion"),
    ]

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        from airflow_dags.promotion import promote_if_better
        promote_if_better()

    mock_client.create_model_version.assert_called_once()
    mock_client.transition_model_version_stage.assert_called_once_with(
        "selastone_credit_scorer", "5", "Production"
    )

def test_does_not_promote_when_improvement_too_small(mock_client):
    """0.88 vs 0.87 = 0.01 improvement < 0.02 threshold → do not promote."""
    mock_client.search_runs.return_value = [
        make_run(auc=0.88, run_id="run_challenger"),
        make_run(auc=0.87, run_id="run_champion"),
    ]

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        from airflow_dags.promotion import promote_if_better
        promote_if_better()

    mock_client.transition_model_version_stage.assert_not_called()

def test_does_not_promote_when_challenger_worse(mock_client):
    """0.88 vs 0.91 → challenger is worse → do not promote."""
    mock_client.search_runs.return_value = [
        make_run(auc=0.88, run_id="run_challenger"),
        make_run(auc=0.91, run_id="run_champion"),
    ]

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        from airflow_dags.promotion import promote_if_better
        promote_if_better()

    mock_client.transition_model_version_stage.assert_not_called()

def test_skips_when_only_one_run(mock_client):
    """Only one run — nothing to compare against → skip."""
    mock_client.search_runs.return_value = [
        make_run(auc=0.88, run_id="run_1")
    ]

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        from airflow_dags.promotion import promote_if_better
        promote_if_better()

    mock_client.transition_model_version_stage.assert_not_called()

def test_skips_when_no_runs(mock_client):
    """No runs at all → skip cleanly."""
    mock_client.search_runs.return_value = []

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        from airflow_dags.promotion import promote_if_better
        promote_if_better()

    mock_client.transition_model_version_stage.assert_not_called()

def test_promotes_at_exact_threshold(mock_client):
    """Exactly 0.02 improvement = exactly at threshold → should promote."""
    mock_client.search_runs.return_value = [
        make_run(auc=0.87, run_id="run_challenger"),
        make_run(auc=0.85, run_id="run_champion"),
    ]

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        from airflow_dags.promotion import promote_if_better
        promote_if_better()

    mock_client.transition_model_version_stage.assert_called_once()