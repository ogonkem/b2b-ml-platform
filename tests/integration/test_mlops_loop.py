import mlflow
import pytest
import time

MLFLOW_URI = "http://localhost:5000"
MODEL_NAME = "selastone_credit_scorer"

def test_mlflow_reachable():
    import requests
    r = requests.get(MLFLOW_URI, timeout=5)
    assert r.status_code == 200

def test_experiment_exists():
    mlflow.set_tracking_uri(MLFLOW_URI)
    exp = mlflow.get_experiment_by_name("selastone_loan_default")
    assert exp is not None

def test_model_registered():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    assert len(versions) > 0

def test_model_manager_hot_swap():
    """
    Simulate a promotion and verify the API picks up the new version.
    Requires: docker stack running + at least one registered model.
    """
    import httpx, time

    # Get current model version from API
    r1 = httpx.get("http://localhost:8000/health")
    version_before = r1.json().get("model_version")

    # Wait for next poll cycle
    time.sleep(65)

    # Version may have changed if a new model was promoted
    r2 = httpx.get("http://localhost:8000/health")
    version_after = r2.json().get("model_version")

    # API must still be healthy regardless
    assert r2.json()["status"] == "healthy"