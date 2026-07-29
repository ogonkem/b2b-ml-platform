"""
lifecycle_tests/01_train_champion.py
────────────────────────────────────────────────────────────────────────────
Stage 1 — TRAIN. Runs notebooks/retrain.py against train_slice.csv only, so
the resulting champion has never seen any row that appears in any other
holdout slice. Registers + promotes that run to Production directly via the
MLflow client, bypassing promote_if_better()'s normal 2-run/2pp-AUC gate:
that gate can't bootstrap a first model (needs 2 runs to compare — see
airflow_dags/promotion.py), and this run needs to become the live baseline
so stages 2-3 score against a holdout-honest model. Stage 6 later exercises
the real gate for its own promotion decision.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    REPO_ROOT, MLFLOW_URI, EXPERIMENT_NAME, MODEL_NAME,
    require, require_holdout_files, mlflow_client, save_state, banner,
)

TRAIN_SLICE = "train_slice.csv"


def run_retrain():
    train_path = REPO_ROOT / "notebooks" / "archive" / "holdout" / TRAIN_SLICE

    env = os.environ.copy()  # _common already points MLflow/S3 vars at localhost
    env["TRAINING_DATA_PATH"] = str(train_path)

    print(f"Running notebooks/retrain.py against {train_path}")
    result = subprocess.run(
        [sys.executable, "notebooks/retrain.py"],
        cwd=REPO_ROOT, env=env,
    )
    require(result.returncode == 0, "notebooks/retrain.py failed — see output above.")


def register_and_promote_baseline(train_started_ms: int) -> tuple[str, str, float]:
    import mlflow

    client = mlflow_client()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    require(exp is not None, f"MLflow experiment '{EXPERIMENT_NAME}' not found after training.")

    # retrain.py logs all 4 candidate models as separate runs — the most
    # *recently logged* one isn't necessarily the winner, so pick the best
    # test_auc among just the runs this invocation produced.
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.test_auc DESC"],
        max_results=20,
    )
    runs = [r for r in runs if r.info.start_time >= train_started_ms]
    require(len(runs) > 0, f"No runs found in experiment '{EXPERIMENT_NAME}' from this training run.")
    run = runs[0]
    run_id = run.info.run_id
    auc = run.data.metrics.get("test_auc")
    require(auc is not None, f"Run {run_id} has no test_auc metric.")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mv = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=MODEL_NAME)
    client.transition_model_version_stage(
        MODEL_NAME, mv.version, "Production", archive_existing_versions=True
    )

    return run_id, mv.version, auc


def main():
    banner("STAGE 1 — TRAIN CHAMPION on train_slice.csv (holdout-clean baseline)")
    require_holdout_files(TRAIN_SLICE)

    train_started_ms = int(time.time() * 1000)
    run_retrain()
    run_id, version, auc = register_and_promote_baseline(train_started_ms)

    print(f"\n  run_id  = {run_id}")
    print(f"  version = {version}  (Production)")
    print(f"  test_auc= {auc:.4f}")

    save_state(baseline_run_id=run_id, baseline_version=version, baseline_auc=auc)
    print("\n[OK] Stage 1 complete — baseline champion trained and promoted.")


if __name__ == "__main__":
    main()
