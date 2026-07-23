import sys, os, pickle
import pandas as pd
from io import BytesIO
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from celery_worker.celery_app import celery_app
from shared.features import FeaturePipeline

_redis_client = None
_minio_client = None
_model        = None
_pipeline     = None


def _redis():
    global _redis_client
    if _redis_client is None:
        import redis as redis_lib
        _redis_client = redis_lib.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            decode_responses=True,
        )
    return _redis_client


def _minio():
    global _minio_client
    if _minio_client is None:
        from minio import Minio
        _minio_client = Minio(
            f"{os.environ.get('MINIO_HOST', 'localhost')}:{os.environ.get('MINIO_PORT', 9000)}",
            access_key=os.environ.get("MINIO_ROOT_USER"),
            secret_key=os.environ.get("MINIO_ROOT_PASSWORD"),
            secure=False,
        )
    return _minio_client


def _ensure_bucket(client, bucket: str):
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def _load_model_and_pipeline():
    global _model, _pipeline
    if _model is None:
        model_path    = os.environ.get("MODEL_PATH",    "models/best_model.pkl")
        pipeline_path = os.environ.get("PIPELINE_PATH", "models/feature_pipeline.pkl")
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
        with open(pipeline_path, "rb") as f:
            _pipeline = pickle.load(f)
    return _model, _pipeline


@celery_app.task(name="process_batch", bind=True, max_retries=2)
def process_batch(self, job_id: str, object_name: str, tenant_id: str):
    """
    Reads a raw CSV from MinIO raw-landing, scores every row using the
    production model + feature pipeline, and writes a results CSV back to
    MinIO batch-results.  Job status is tracked in Redis throughout.
    """
    r     = _redis()
    minio = _minio()

    try:
        r.set(f"job:{job_id}:status", "processing", ex=86400)

        # ── Download input CSV ────────────────────────────────────────────────
        response = minio.get_object("raw-landing", object_name)
        df = pd.read_csv(BytesIO(response.read()))

        # ── Score ─────────────────────────────────────────────────────────────
        model, pipeline = _load_model_and_pipeline()
        X     = pipeline.transform(df)
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        results_df = pd.DataFrame({
            "application_id":      df["ID"] if "ID" in df.columns else range(len(df)),
            "default_prediction":  preds.tolist(),
            "default_probability": [round(float(p), 4) for p in probs],
        })

        # ── Write results to MinIO ────────────────────────────────────────────
        _ensure_bucket(minio, "batch-results")
        date_str      = datetime.utcnow().strftime("%Y%m%d")
        result_object = f"{tenant_id}/{date_str}/{job_id}_results.csv"
        csv_bytes     = results_df.to_csv(index=False).encode()

        minio.put_object(
            "batch-results", result_object,
            data=BytesIO(csv_bytes), length=len(csv_bytes),
            content_type="text/csv",
        )

        # ── Mark complete ─────────────────────────────────────────────────────
        r.set(f"job:{job_id}:status",        "complete",         ex=86400)
        r.set(f"job:{job_id}:result_object", result_object,      ex=86400)
        r.set(f"job:{job_id}:rows_scored",   str(len(results_df)), ex=86400)

    except Exception as exc:
        r.set(f"job:{job_id}:status", "failed", ex=86400)
        r.set(f"job:{job_id}:error",  str(exc), ex=86400)
        raise self.retry(exc=exc, countdown=30)
