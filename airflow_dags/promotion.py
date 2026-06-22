import mlflow, json, os

# In promote_if_better():
def promote_if_better():
    """Check if the latest model is better than the current champion and promote if so."""
    import mlflow, json, os

    # Connect to MLflow and get the latest two runs for our experiment
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("selastone_loan_default")

    # Get latest runs ordered by test AUC (descending)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_auc DESC"],
        max_results=2
    )
    if len(runs) < 2:
        return  # Nothing to compare

    challenger = runs[0] # best run from this week
    champion    = runs[1] # previous best run

    # Extract AUC metrics (default to 0 if missing)
    challenger_auc = challenger.data.metrics.get("test_auc", 0)
    champion_auc   = champion.data.metrics.get("test_auc", 0)

    # Only promote if challenger is meaningfully better (>= 2% improvement)
    if challenger_auc >= champion_auc + 0.02:
        mv = client.create_model_version(
            "selastone_credit_scorer",
            challenger.info.run_id,
            "model"
        )
        client.transition_model_version_stage(
            "selastone_credit_scorer", mv.version, "Production"
        )
        print(f"Promoted model v{mv.version} — AUC {challenger_auc:.4f} vs {champion_auc:.4f}")
    else:
        print(f"Challenger AUC {challenger_auc:.4f} not better enough — keeping champion")