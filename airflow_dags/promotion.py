import mlflow, json, os

MODEL_NAME = "selastone_credit_scorer"


# In promote_if_better():
def promote_if_better():
    """Check if the latest model is better than the current champion and promote if so."""
    import mlflow, json, os

    # Connect to MLflow and get the latest two runs for our experiment
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("selastone_loan_default")

    # Challenger: best test_auc among the batch of runs notebooks/retrain.py
    # just logged (it always logs exactly 4 candidates — XGBoost, LightGBM,
    # LogisticRegression, RandomForest — in one invocation, all within
    # seconds of each other). Ordering by metrics.test_auc DESC across the
    # *whole* experiment history instead would risk picking two sibling
    # candidates from that same run once enough retrains have piled up close
    # AUC scores — comparing a run against itself and never actually testing
    # against what's deployed.
    recent_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=4,
    )
    if not recent_runs:
        return  # Nothing to compare

    challenger = max(recent_runs, key=lambda r: r.data.metrics.get("test_auc", 0))
    challenger_auc = challenger.data.metrics.get("test_auc", 0)

    # Champion: whatever run currently backs the deployed Production version.
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    production = [v for v in versions if v.current_stage == "Production"]
    if not production:
        return  # No champion deployed yet — bootstrap promotion happens outside this gate

    champion_run = client.get_run(production[0].run_id)
    champion_auc = champion_run.data.metrics.get("test_auc", 0)

    # Only promote if challenger is meaningfully better (>= 2% improvement)
    if challenger_auc >= champion_auc + 0.02:
        # mlflow.register_model() (not client.create_model_version() directly)
        # — newer MLflow versions log xgboost/lightgbm/sklearn models as a
        # separate "Logged Model" rather than a classic run artifact, so
        # runs:/<id>/model resolves to nothing and ModelManager's
        # mlflow.pyfunc.load_model() fails with "Could not find an MLmodel
        # configuration file". register_model() detects that and
        # transparently falls back to the model's real models:/m-... URI;
        # create_model_version() alone does not.
        mv = mlflow.register_model(
            model_uri=f"runs:/{challenger.info.run_id}/model", name=MODEL_NAME
        )
        client.transition_model_version_stage(
            MODEL_NAME, mv.version, "Production", archive_existing_versions=True
        )
        print(f"Promoted model v{mv.version} — AUC {challenger_auc:.4f} vs {champion_auc:.4f}")
    else:
        print(f"Challenger AUC {challenger_auc:.4f} not better enough — keeping champion")
