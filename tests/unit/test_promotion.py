import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from unittest.mock import patch, MagicMock


def make_run(auc: float, run_id: str, start_time: int = 0):
    run = MagicMock()
    run.info.run_id = run_id
    run.info.start_time = start_time
    run.data.metrics = {"test_auc": auc}
    return run


def make_version(run_id: str, stage: str, version: str = "1"):
    v = MagicMock()
    v.run_id = run_id
    v.current_stage = stage
    v.version = version
    return v


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    return client


@pytest.fixture
def mock_register_model():
    mv = MagicMock()
    mv.version = "5"
    with patch("mlflow.register_model", return_value=mv) as m:
        yield m


def _set_champion(mock_client, auc: float, run_id: str = "run_champion"):
    """The champion is whatever run currently backs the deployed Production
    version — not simply 'the 2nd best run ever' (see promotion.py's
    docstring for why that comparison is wrong)."""
    mock_client.search_model_versions.return_value = [
        make_version(run_id, "Production", version="1"),
    ]
    mock_client.get_run.return_value = make_run(auc, run_id)


# ── Promotion logic ───────────────────────────────────────────────────────────

def test_promotes_when_challenger_is_better(mock_client, mock_register_model):
    """0.88 vs 0.85 = 0.03 improvement >= 0.02 threshold → promote."""
    mock_client.search_runs.return_value = [make_run(auc=0.88, run_id="run_challenger")]
    _set_champion(mock_client, auc=0.85)

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        from airflow_dags.promotion import promote_if_better
        promote_if_better()

    # Regression guard: register_model needs an actual model_uri pointing at
    # the challenger run — a bare run_id (or run_id/name swapped) is silently
    # accepted by a mock but fails against the real MLflow client.
    mock_register_model.assert_called_once_with(
        model_uri="runs:/run_challenger/model", name="selastone_credit_scorer",
    )
    mock_client.transition_model_version_stage.assert_called_once_with(
        "selastone_credit_scorer", "5", "Production", archive_existing_versions=True
    )

def test_does_not_promote_when_improvement_too_small(mock_client, mock_register_model):
    """0.86 vs 0.85 = 0.01 improvement < 0.02 threshold → do not promote."""
    mock_client.search_runs.return_value = [make_run(auc=0.86, run_id="run_challenger")]
    _set_champion(mock_client, auc=0.85)

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        from airflow_dags.promotion import promote_if_better
        promote_if_better()

    mock_register_model.assert_not_called()
    mock_client.transition_model_version_stage.assert_not_called()

def test_does_not_promote_when_challenger_worse(mock_client, mock_register_model):
    """0.80 vs 0.91 → challenger is worse → do not promote."""
    mock_client.search_runs.return_value = [make_run(auc=0.80, run_id="run_challenger")]
    _set_champion(mock_client, auc=0.91)

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        from airflow_dags.promotion import promote_if_better
        promote_if_better()

    mock_register_model.assert_not_called()
    mock_client.transition_model_version_stage.assert_not_called()

def test_skips_when_no_recent_runs(mock_client, mock_register_model):
    """No runs logged this retrain — nothing to compare against → skip."""
    mock_client.search_runs.return_value = []

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        from airflow_dags.promotion import promote_if_better
        promote_if_better()

    mock_register_model.assert_not_called()
    mock_client.transition_model_version_stage.assert_not_called()

def test_skips_when_no_production_version(mock_client, mock_register_model):
    """No champion deployed yet — first model must be bootstrapped outside
    this gate (it can't promote itself; see notebooks/retrain.py's caller)."""
    mock_client.search_runs.return_value = [make_run(auc=0.88, run_id="run_challenger")]
    mock_client.search_model_versions.return_value = []

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        from airflow_dags.promotion import promote_if_better
        promote_if_better()

    mock_register_model.assert_not_called()
    mock_client.transition_model_version_stage.assert_not_called()

def test_promotes_at_exact_threshold(mock_client, mock_register_model):
    """Exactly 0.02 improvement = exactly at threshold → should promote."""
    mock_client.search_runs.return_value = [make_run(auc=0.87, run_id="run_challenger")]
    _set_champion(mock_client, auc=0.85)

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        from airflow_dags.promotion import promote_if_better
        promote_if_better()

    mock_register_model.assert_called_once_with(
        model_uri="runs:/run_challenger/model", name="selastone_credit_scorer",
    )
    mock_client.transition_model_version_stage.assert_called_once()

def test_picks_best_among_recent_batch(mock_client, mock_register_model):
    """notebooks/retrain.py logs 4 candidate runs per invocation — the
    challenger must be the best of THOSE, not just the first one returned."""
    mock_client.search_runs.return_value = [
        make_run(auc=0.80, run_id="run_xgb"),
        make_run(auc=0.93, run_id="run_lgbm"),   # best of the batch
        make_run(auc=0.75, run_id="run_logreg"),
        make_run(auc=0.60, run_id="run_rf"),
    ]
    _set_champion(mock_client, auc=0.85)

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        from airflow_dags.promotion import promote_if_better
        promote_if_better()

    mock_register_model.assert_called_once_with(
        model_uri="runs:/run_lgbm/model", name="selastone_credit_scorer",
    )

def test_does_not_compare_two_runs_from_the_same_batch(mock_client, mock_register_model):
    """Regression guard for the original bug: ordering candidates by
    metrics.test_auc DESC across the whole experiment (instead of picking the
    champion from the deployed Production version) could pick the top-2
    AUC scores from the very same just-finished batch — comparing sibling
    candidates against each other and never testing against what's actually
    live. Two close-scoring runs from one batch must still promote when they
    both clear the deployed champion by >= 2pp.
    """
    mock_client.search_runs.return_value = [
        make_run(auc=0.8972, run_id="run_xgb"),
        make_run(auc=0.8962, run_id="run_lgbm"),   # 0.001 apart from each other
    ]
    _set_champion(mock_client, auc=0.8716, run_id="run_old_champion")  # >= 2pp behind both

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        from airflow_dags.promotion import promote_if_better
        promote_if_better()

    mock_register_model.assert_called_once_with(
        model_uri="runs:/run_xgb/model", name="selastone_credit_scorer",
    )
