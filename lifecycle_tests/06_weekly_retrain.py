"""
lifecycle_tests/06_weekly_retrain.py
────────────────────────────────────────────────────────────────────────────
Stage 6 — WEEKLY RETRAIN. Triggers the real selastone_weekly_retrain DAG
(sync_data -> train_model -> promote_model) and lets it run to completion
inside the now-fixed Airflow container. train_model calls notebooks/retrain.py
with no TRAINING_DATA_PATH override, so it trains on the full baseline CSV
merged with the feedback_labeled.csv stage 5 just committed — matching real
weekly-DAG behavior. promote_model then runs the actual, governed
2-run/2pp-AUC promotion gate (airflow_dags/promotion.py) comparing this
challenger against stage 1's baseline run — for real, not simulated.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    EXPERIMENT_NAME, MODEL_NAME,
    require, require_state, mlflow_client, save_state, banner,
    trigger_dag_and_wait,
)

DAG_ID = "selastone_weekly_retrain"


def main():
    banner("STAGE 6 — WEEKLY RETRAIN (real sync_data -> train_model -> promote_model)")
    state = require_state("baseline_run_id", "baseline_version", "baseline_auc", "feedback_commit")

    run_id = f"lifecycle_weekly_retrain_{int(time.time())}"
    print(f"Triggering {DAG_ID} (run_id={run_id}) — training on the full baseline "
          f"+ feedback, this can take a few minutes...")
    dag_state = trigger_dag_and_wait(DAG_ID, run_id, timeout_s=900, poll_s=10)
    require(dag_state == "success", f"{DAG_ID}/{run_id} finished with state={dag_state}")

    client = mlflow_client()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.test_auc DESC"],
        max_results=20,
    )
    challenger_runs = [r for r in runs if r.info.run_id != state["baseline_run_id"]]
    require(len(challenger_runs) > 0, "no challenger runs found after weekly retrain")
    challenger = challenger_runs[0]
    challenger_auc = challenger.data.metrics.get("test_auc")

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    production = [v for v in versions if v.current_stage == "Production"]
    promoted = bool(production) and production[0].version != state["baseline_version"]

    print(f"\n  baseline   run_id={state['baseline_run_id']}  auc={state['baseline_auc']:.4f}")
    print(f"  challenger run_id={challenger.info.run_id}  auc={challenger_auc:.4f}")
    print(f"  delta      {challenger_auc - state['baseline_auc']:+.4f}  (>= 0.02 required to promote)")
    print(f"  Production version now: {production[0].version if production else 'none'}")
    print(f"  Promoted challenger: {promoted}")

    save_state(
        challenger_run_id=challenger.info.run_id,
        challenger_auc=challenger_auc,
        promoted=promoted,
        production_version=production[0].version if production else None,
    )
    print("\n[OK] Stage 6 complete.")


if __name__ == "__main__":
    main()
