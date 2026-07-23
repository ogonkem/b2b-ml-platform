"""
tests/unit/test_model_manager.py
Tests written to match the actual ModelManager implementation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import threading
import time
from unittest.mock import MagicMock, patch

# ── Patch MlflowClient at module level so __init__ never calls real MLflow ────
mlflow_patcher = patch("app.model_manager.mlflow.tracking.MlflowClient")
mock_client_cls = mlflow_patcher.start()
mock_client_cls.return_value = MagicMock()

from app.model_manager import ModelManager

# Stop patch after import if you want (optional — keeping it running is fine for tests)


@pytest.fixture
def manager():
    """Fresh ModelManager — _client is always a MagicMock."""
    mock_client = MagicMock()
    with patch("app.model_manager.mlflow.tracking.MlflowClient", return_value=mock_client):
        m = ModelManager("selastone_credit_scorer", poll_interval=1)
    # Replace client directly to control it in tests
    m._client = mock_client
    return m


# ── Initialisation ────────────────────────────────────────────────────────────

def test_manager_starts_with_no_model(manager):
    assert manager._model is None

def test_manager_starts_with_no_version(manager):
    assert manager._version is None

def test_manager_model_name_stored(manager):
    assert manager.model_name == "selastone_credit_scorer"

def test_manager_has_lock(manager):
    assert isinstance(manager._lock, threading.Lock().__class__)

def test_is_loaded_false_when_no_model(manager):
    assert manager.is_loaded is False


# ── load_latest ───────────────────────────────────────────────────────────────

def test_load_latest_no_production_version(manager):
    # search_model_versions returns versions but none in Production stage
    manager._client.search_model_versions.return_value = []
    manager.load_latest()
    assert manager._model is None

def test_load_latest_mlflow_unreachable(manager):
    manager._client.search_model_versions.side_effect = Exception("Connection refused")
    manager.load_latest()
    assert manager._model is None

def test_load_latest_loads_new_version(manager):
    mock_version = MagicMock()
    mock_version.version       = "3"
    mock_version.current_stage = "Production"   # required by the stage filter
    manager._client.search_model_versions.return_value = [mock_version]

    mock_model = MagicMock()
    with patch("app.model_manager.mlflow.pyfunc.load_model", return_value=mock_model):
        manager.load_latest()

    assert manager._model is mock_model
    assert manager._version == "3"

def test_load_latest_skips_same_version(manager):
    mock_version = MagicMock()
    mock_version.version       = "3"
    mock_version.current_stage = "Production"
    manager._client.search_model_versions.return_value = [mock_version]
    manager._version = "3"

    with patch("app.model_manager.mlflow.pyfunc.load_model") as mock_load:
        manager.load_latest()
        mock_load.assert_not_called()

def test_load_latest_swaps_to_newer_version(manager):
    mock_v4 = MagicMock()
    mock_v4.version       = "4"
    mock_v4.current_stage = "Production"
    manager._client.search_model_versions.return_value = [mock_v4]
    manager._version = "3"

    mock_model = MagicMock()
    with patch("app.model_manager.mlflow.pyfunc.load_model", return_value=mock_model):
        manager.load_latest()

    assert manager._version == "4"

def test_load_latest_download_fails_keeps_old_model(manager):
    old_model = MagicMock()
    manager._model   = old_model
    manager._version = "3"

    mock_v4 = MagicMock()
    mock_v4.version       = "4"
    mock_v4.current_stage = "Production"
    manager._client.search_model_versions.return_value = [mock_v4]

    with patch("app.model_manager.mlflow.pyfunc.load_model",
               side_effect=Exception("Download failed")):
        manager.load_latest()

    assert manager._model is old_model
    assert manager._version == "3"

def test_is_loaded_true_after_load(manager):
    mock_v = MagicMock()
    mock_v.version       = "1"
    mock_v.current_stage = "Production"
    manager._client.search_model_versions.return_value = [mock_v]
    with patch("app.model_manager.mlflow.pyfunc.load_model", return_value=MagicMock()):
        manager.load_latest()
    assert manager.is_loaded is True


# ── predict_proba ─────────────────────────────────────────────────────────────

def test_predict_proba_raises_if_no_model(manager):
    """No model loaded — must raise some exception."""
    with pytest.raises(Exception):   # catches AttributeError or RuntimeError
        manager.predict_proba([[1, 2, 3]])

def test_predict_proba_delegates_to_model(manager):
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.25, 0.75]]
    manager._model   = mock_model
    manager._version = "3"

    manager.predict_proba([[1, 2, 3]])
    mock_model.predict_proba.assert_called_once_with([[1, 2, 3]])

def test_predict_proba_returns_model_output(manager):
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.58, 0.42]]
    manager._model   = mock_model
    manager._version = "3"

    result = manager.predict_proba([[1, 2, 3]])
    assert result == [[0.58, 0.42]]


# ── Thread safety ─────────────────────────────────────────────────────────────

def test_concurrent_predictions_do_not_crash(manager):
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.5, 0.5]]
    manager._model   = mock_model
    manager._version = "1"

    errors = []
    def call_predict():
        try:
            manager.predict_proba([[1, 2, 3]])
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=call_predict) for _ in range(20)]
    [t.start() for t in threads]
    [t.join()  for t in threads]
    assert errors == []

def test_hot_swap_does_not_corrupt_predictions(manager):
    mock_v1 = MagicMock()
    mock_v1.predict_proba.return_value = [[0.7, 0.3]]
    mock_v2 = MagicMock()
    mock_v2.predict_proba.return_value = [[0.3, 0.7]]

    manager._model   = mock_v1
    manager._version = "1"

    results = []

    def predict_loop():
        for _ in range(50):
            try:
                r = manager.predict_proba([[1, 2, 3]])
                results.append(r[0][1])   # probability of default
            except Exception:
                results.append(None)

    def swap_model():
        time.sleep(0.01)
        with manager._lock:
            manager._model   = mock_v2
            manager._version = "2"

    t1 = threading.Thread(target=predict_loop)
    t2 = threading.Thread(target=swap_model)
    t1.start(); t2.start()
    t1.join();  t2.join()

    assert None not in results
    assert all(r in (0.3, 0.7) for r in results)


# ── Polling ───────────────────────────────────────────────────────────────────

def test_start_polling_does_not_block(manager):
    manager.load_latest = lambda: None
    start = time.time()
    manager.start_polling()
    assert time.time() - start < 1.0

def test_polling_calls_load_latest(manager):
    call_count = []
    manager.load_latest = lambda: call_count.append(1)
    manager.start_polling()
    time.sleep(0.15)
    assert len(call_count) >= 1