"""
ModelManager: A class to manage MLflow model versions and hot-swap the latest Production model for inference.
This class polls the MLflow model registry for new Production versions of a specified model and reloads
"""
import threading, time, pickle, mlflow, mlflow.pyfunc
import os

class ModelManager:
    """
    A class to manage MLflow model versions and hot-swap the latest Production model for inference.
    This class polls the MLflow model registry for new Production versions of a specified model and reloads
    the model in a thread-safe manner. It provides a predict_proba method for making predictions using the
    latest model. 
    """
    def __init__(self, model_name: str, poll_interval: int = 60):
        self._model_name    = model_name # Name of the MLflow registered model to manage
        self.poll_interval = poll_interval # Polling interval in seconds to check for new Production versions
        self._model        = None # The currently loaded model (thread-safe access)
        self._version      = None # The version of the currently loaded model
        self._lock         = threading.Lock() # A lock to ensure thread-safe access to the model and version
        self._client       = mlflow.tracking.MlflowClient(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        ) # MLflow client for interacting with the model registry

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def version(self):
        return self._version

    def load_latest(self):
        try:
            import mlflow.pyfunc
            # New API — replaces deprecated get_latest_versions
            versions = self._client.search_model_versions(
                f"name='{self.model_name}'"
            )
            production = [v for v in versions if v.current_stage == "Production"]
            
            if not production:
                return

            latest = max(production, key=lambda v: int(v.version))

            if latest.version == self._version:
                return

            new_model = mlflow.pyfunc.load_model(
                f"models:/{self.model_name}/Production"
            )
            with self._lock:
                self._model   = new_model
                self._version = latest.version
            print(f"[ModelManager] ✓ Hot-swapped to v{self._version}")

        except Exception as e:
            print(f"[ModelManager] load_latest error: {e}")
            
    def predict_proba(self, X):
        """Thread-safe prediction using the latest model."""
        with self._lock:
            return self._model.predict_proba(X)

    def start_polling(self):
        """Start the polling thread."""
        def _poll():
            while True:
                try:
                    self.load_latest()
                except Exception as e:
                    print(f"[ModelManager] Poll error: {e}")
                time.sleep(self.poll_interval)
        t = threading.Thread(target=_poll, daemon=True)
        t.start()