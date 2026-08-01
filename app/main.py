import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import pickle, os
import hmac, hashlib
import redis
from datetime import datetime, timedelta
import clickhouse_connect
import uuid
from minio import Minio
from io import BytesIO
from prometheus_client import Counter, Histogram, make_asgi_app
import time
from app.model_manager import ModelManager
from app.plans import PLANS, DEFAULT_PLAN

try:
    from celery_worker.celery_app import celery_app as _celery_app
except ImportError:
    _celery_app = None


# Align local paths for imports
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from shared.features import FeaturePipeline

app = FastAPI(
    title="Loan Pipeline Gateway",
    version="1.3.0",
    description="Production API endpoint for processing dynamic loan configurations and calculating risk defaults."
)
security_scheme = HTTPBearer(auto_error=False)

# No browser client existed before the frontend/ SPA — CORS was never
# needed until now. FRONTEND_ORIGINS is a comma-separated list so both the
# Vite dev server and the built/nginx-served bundle can be allowed at once.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FRONTEND_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Every endpoint here returns data scoped to whichever bearer token made the
# request, on URLs that don't otherwise vary per tenant (e.g. GET
# /v1/usage). Without this, a browser (or any intermediary cache) is free to
# serve one tenant's cached response to a different tenant hitting the same
# path — the HTTP cache model keys on URL, not on Authorization header
# contents, unless told not to cache at all.
@app.middleware("http")
async def no_store_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response

from app.auth import router as auth_router, require_admin, get_current_user
app.include_router(auth_router)

# -------------------------------------------------------------------------
# prometheus metrics
# -------------------------------------------------------------------------
PREDICTIONS_TOTAL  = Counter(
    "predictions_total",
    "Total predictions made",
    ["tenant_id", "prediction"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency"
)
BATCH_UPLOADS_TOTAL = Counter(          
    "batch_uploads_total",
    "Total batch CSV uploads",
    ["tenant_id", "status"]            
)
BATCH_ROWS_TOTAL = Counter(             
    "batch_rows_total",
    "Total rows received via batch upload",
    ["tenant_id"]
)
BATCH_UPLOAD_LATENCY = Histogram(       
    "batch_upload_latency_seconds",
    "Batch upload request latency"
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# -------------------------------------------------------------------------
# AUTHENTICATION
# -------------------------------------------------------------------------

VALID_TOKENS = set(os.environ.get("API_TOKENS", "dev-token").split(","))

def _lookup_api_key_tenant(raw_key: str) -> Optional[str]:
    """Resolve a persistent, DB-issued API key (POST /auth/api-key) to its
    owner's tenant_id. Any failure (Postgres unreachable, etc.) is treated as
    'not a valid key' rather than a hard error — this is one of several
    things a bearer value could be, not the only one."""
    from app.auth import _hash_key
    from app.db import get_cursor
    try:
        with get_cursor() as cur:
            cur.execute(
                """SELECT u.tenant_id FROM app.api_keys k
                   JOIN app.users u ON u.id = k.user_id
                   WHERE k.key_hash = %s""",
                (_hash_key(raw_key),),
            )
            row = cur.fetchone()
        return row["tenant_id"] if row else None
    except Exception:
        return None


def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme)):
    """Accepts three kinds of bearer value, in order, all resolving to a
    tenant_id exactly as before — /v1/predict etc. don't need to know which
    kind was used:
      1. A static, pre-shared API_TOKENS entry (unchanged) — tenant_id is
         the token itself.
      2. A JWT issued by /auth/login or /auth/register (app/auth.py) —
         tenant_id comes from the JWT's own claim.
      3. A persistent, DB-issued API key (POST /auth/api-key) — only
         attempted for values shaped like one (see app.auth._generate_api_key)
         so a plain invalid token never triggers a Postgres round-trip.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Use Bearer token.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    token = credentials.credentials

    if token in VALID_TOKENS:
        return token

    from app.auth import decode_jwt
    payload = decode_jwt(token)
    if payload is not None:
        return payload["tenant_id"]

    if token.startswith("sk_"):
        tenant_id = _lookup_api_key_tenant(token)
        if tenant_id is not None:
            return tenant_id

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API token."
    )


# -------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -------------------------------------------------------------------------
class LoanApplication(BaseModel):
    ID: int = Field(..., example=24896)
    year: int = Field(..., example=2019)
    loan_limit: Optional[str] = "cf"
    Gender: Optional[str] = "Joint"
    approv_in_adv: Optional[str] = "pre"
    loan_type: Optional[str] = "type1"
    loan_purpose: Optional[str] = "p3"
    Credit_Worthiness: Optional[str] = "l1"
    open_credit: Optional[str] = "nopc"
    business_or_commercial: Optional[str] = "nob/c"
    loan_amount: float = Field(..., example=346500.0)
    rate_of_interest: Optional[float] = 4.5
    Interest_rate_spread: Optional[float] = 0.9998
    Upfront_charges: Optional[float] = 5120.0
    term: Optional[float] = 360.0
    Neg_ammortization: Optional[str] = "not_neg"
    interest_only: Optional[str] = "not_int"
    lump_sum_payment: Optional[str] = "not_lpsm"
    property_value: float = Field(..., example=438000.0)
    construction_type: Optional[str] = "sb"
    occupancy_type: Optional[str] = "pr"
    Secured_by: Optional[str] = "home"
    total_units: Optional[str] = "1U"
    income: float = Field(..., example=5040.0)
    credit_type: Optional[str] = "EXP"
    Credit_Score: float = Field(..., example=860.0)
    co_applicant_credit_type: Optional[str] = "EXP"
    age: Optional[str] = "55-64"
    submission_of_application: Optional[str] = "to_inst"
    LTV: Optional[float] = 79.10958904
    Region: Optional[str] = "North"
    Security_Type: Optional[str] = "direct"
    dtir1: Optional[float] = 44.0


class LabeledDataSchema(BaseModel):
    """Returned by POST /v1/labeled-data after the business uploads actuals."""
    object:        str
    tenant_id:     str
    rows_received: int
    status:        str

# -------------------------------------------------------------------------
# GLOBAL SERVICE INITIALIZATION
# -------------------------------------------------------------------------
# Set to True when real model artefacts exist (after running the notebook)
# Set to False during early development / CI without model files
USE_REAL_ARTEFACTS = os.environ.get("USE_REAL_ARTEFACTS", "false").lower() == "true"

if USE_REAL_ARTEFACTS:
    # ── Production mode: load real trained artefacts ──────────────────────
    MODEL_PATH    = os.environ.get("MODEL_PATH",    "models/best_model.pkl")
    PIPELINE_PATH = os.environ.get("PIPELINE_PATH", "models/feature_pipeline.pkl")

    with open(MODEL_PATH, 'rb') as f:
        pred_model = pickle.load(f)
    with open(PIPELINE_PATH, 'rb') as f:
        feature_pipeline = pickle.load(f)

    print(f"✓ Loaded real model    → {MODEL_PATH}")
    print(f"✓ Loaded real pipeline → {PIPELINE_PATH} ({len(feature_pipeline.feature_names)} features)")

else:
    # ── Development mode: dry-fit on baseline dummy data ──────────────────
    feature_pipeline = FeaturePipeline()

    baseline_historical_df = pd.DataFrame({
        "ID": [1, 2, 3],
        "income": [1740.0, 4980.0, 11880.0],
        "loan_amount": [116500.0, 206500.0, 456500.0],
        "property_value": [118000.0, 580000.0, 658000.0],
        "Credit_Score": [600.0, 720.0, 800.0],
        "rate_of_interest": [4.0, 4.5, 3.8],
        "Interest_rate_spread": [0.5, 0.9, 0.2],
        "Upfront_charges": [1200.0, 5000.0, 3100.0],
        "term": [360.0, 360.0, 180.0],
        "LTV": [70.0, 80.0, 65.0],
        "dtir1": [30.0, 40.0, 25.0]
    })
    feature_pipeline.fit(baseline_historical_df, target_col='Status')

    pred_model = xgb.XGBClassifier()
    mock_y = np.array([0, 1, 0])
    mock_X = feature_pipeline.transform(baseline_historical_df)
    pred_model.fit(mock_X, mock_y)

    print("⚠️  Running in development mode — using mock model and pipeline")
    print("   Set USE_REAL_ARTEFACTS=true (or in .env) to load real artefacts")

# ── ModelManager — start AFTER model is loaded ────────────────────────────
# Only polls MLflow when real artefacts are in use
# In dev mode it starts but load_latest() will find nothing and skip cleanly

try:
    model_manager = ModelManager(
        model_name="selastone_credit_scorer",
        poll_interval=int(os.environ.get("MODEL_POLL_INTERVAL", 60))
    )
    model_manager.start_polling()  # ← background thread only, don't call load_latest() here
    print("✓ ModelManager polling started")
except Exception as e:
    print(f"⚠️  ModelManager failed to start: {e}")
    model_manager = None

# ── Auth schema — idempotent, safe to run on every startup ────────────────
from app.db import init_schema

try:
    init_schema()
    print("✓ Auth schema ready (app.tenants / app.users / app.api_keys)")
except Exception as e:
    print(f"⚠️  Auth schema init failed: {e}")

# -------------------------------------------------------------------------
# Redis client initialization (for future stateful features like caching or rate limiting)
# -------------------------------------------------------------------------
redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    decode_responses=True
)

MONTHLY_QUOTA_LIMIT = 1000  # fallback default only — real callers pass monthly_limit=get_plan_quota(tenant)

def check_and_increment_quota(tenant_id: str, increment: int = 1, monthly_limit: int = MONTHLY_QUOTA_LIMIT):
    """
    Increments this tenant's monthly row counter by `increment`.
    For /v1/predict pass increment=1 (default).
    For /v1/batch/upload pass increment=row_count so bulk jobs consume quota fairly.
    Raises 429 if the running total exceeds monthly_limit.
    """
    key = f"quota:{tenant_id}:{datetime.utcnow().strftime('%Y_%m')}"
    # Check before incrementing — incrementing first (then rejecting on
    # overage) permanently charges the tenant for requests that never went
    # through, so a single oversized batch upload could burn through the
    # rest of the month's quota even though it was rejected outright.
    existing = int(redis_client.get(key) or 0)
    if existing + increment > monthly_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Monthly prediction quota exceeded"
        )
    current = redis_client.incrby(key, increment)
    if current <= increment:
        # Key was just created (or reset); set ~1-month TTL for automatic reset
        redis_client.expire(key, 60 * 60 * 24 * 32)
    return current


def get_plan_quota(tenant_id: str) -> int:
    """The real limit check_and_increment_quota should enforce for this
    tenant, based on its subscription plan — kept separate from
    check_and_increment_quota itself so that function stays pure-Redis
    with no Postgres dependency on its hot path's unit tests."""
    from app.db import get_tenant_plan
    plan = get_tenant_plan(tenant_id)
    return PLANS.get(plan, PLANS[DEFAULT_PLAN])["monthly_quota"]

# -------------------------------------------------------------------------
# ClickHouse client initialization (for future logging of predictions and usage)
# -------------------------------------------------------------------------
ch_client = None

def get_ch_client():
    """Lazy ClickHouse connection — only connects when first needed."""
    global ch_client
    if ch_client is None:
        try:
            import clickhouse_connect
            ch_client = clickhouse_connect.get_client(
                host=os.environ.get("CLICKHOUSE_HOST", "localhost"),
                port=int(os.environ.get("CLICKHOUSE_PORT", 8123)),
                username=os.environ.get("CLICKHOUSE_USER", "default"),
                password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
            )
            ch_client.command("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    tenant_id      String,
                    application_id Int64,
                    probability    Float32,
                    prediction     Int8,
                    model_version  String,
                    ts             DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                ORDER BY (tenant_id, ts)
            """)
            ch_client.command("""
                CREATE TABLE IF NOT EXISTS labeled_uploads (
                    tenant_id      String,
                    object_name    String,
                    rows_received  Int32,
                    ts             DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                ORDER BY (tenant_id, ts)
            """)
            print("✓ ClickHouse connected")
        except Exception as e:
            print(f"⚠️  ClickHouse unavailable: {e}")
            ch_client = None
    return ch_client


def log_prediction(tenant_id, app_id, prob, pred, model_version="v1"):
    client = get_ch_client()
    if client is None:
        print(f"[WARNING] ClickHouse not available — skipping log app_id={app_id}")
        return
    try:
        client.insert(
            "prediction_logs",
            [[tenant_id, app_id, prob, pred, model_version]],
            column_names=["tenant_id", "application_id", "probability",
                          "prediction", "model_version"]
        )
    except Exception as e:
        print(f"[WARNING] ClickHouse logging failed: {e}")


def log_labeled_upload(tenant_id: str, object_name: str, rows: int):
    """Audit trail: records every labeled CSV upload in ClickHouse."""
    client = get_ch_client()
    if client is None:
        return
    try:
        client.insert(
            "labeled_uploads",
            [[tenant_id, object_name, rows]],
            column_names=["tenant_id", "object_name", "rows_received"],
        )
    except Exception as e:
        print(f"[WARNING] ClickHouse labeled-upload log failed: {e}")

# --------------------------------------------------------------------------   
# MinIO client initialization (for future model artefact storage and retrieval)
# -------------------------------------------------------------------------
minio_client = Minio(
    f"{os.environ.get('MINIO_HOST', 'localhost')}:{os.environ.get('MINIO_PORT', 9000)}",
    access_key=os.environ.get("MINIO_ROOT_USER"),
    secret_key=os.environ.get("MINIO_ROOT_PASSWORD"),
    secure=False
)

# A presigned URL's host is baked into its SigV4 signature — MINIO_HOST is the
# Docker network hostname ("minio"), which only resolves inside the compose
# network, not in a browser. Rewriting the host in a returned URL after the
# fact breaks the signature (SignatureDoesNotMatch), so download links handed
# to the frontend must be signed against a browser-reachable host to begin
# with. This client exists only for that; every other MinIO call above still
# goes through minio_client on the internal network.
minio_public_client = Minio(
    os.environ.get("MINIO_PUBLIC_ENDPOINT", "localhost:9000"),
    access_key=os.environ.get("MINIO_ROOT_USER"),
    secret_key=os.environ.get("MINIO_ROOT_PASSWORD"),
    secure=False,
    # Without an explicit region, minio-py's presigned_get_object() calls
    # GetBucketLocation against the configured host to discover it — which
    # would try to reach "localhost:9000" from inside this container (where
    # localhost isn't MinIO) and fail. SigV4 signing itself needs no
    # connectivity; setting the region short-circuits that lookup so this
    # client never needs to actually be reachable, only correct in the URL
    # it signs.
    region="us-east-1",
)


# -------------------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------------------
@app.post(
    "/v1/predict",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Evaluate Default Risk Score"
)
async def predict_default(
    payload: LoanApplication,
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme)
):
    tenant = verify_token(token)
    check_and_increment_quota(tenant, monthly_limit=get_plan_quota(tenant))

    start = time.time() # start timer
    try:
        # The training data's raw column is `co-applicant_credit_type` (hyphen,
        # straight from the source CSV) — Python identifiers can't contain
        # hyphens, so the pydantic field is named with an underscore instead.
        # Without this rename, FeaturePipeline never finds the column under its
        # trained name and silently falls back to the training-mode value on
        # every request, ignoring whatever the caller actually sent.
        payload_dict = payload.model_dump()
        payload_dict["co-applicant_credit_type"] = payload_dict.pop("co_applicant_credit_type")
        transformed_features = feature_pipeline.transform(pd.DataFrame([payload_dict]))

        # Prefer the hot-swapped Production model once ModelManager has loaded
        # one; fall back to the model loaded at startup (dev mode, or before any
        # model has ever been promoted to Production). Without this branch,
        # /v1/predict silently keeps using the startup model forever even after
        # promotion — ModelManager's poll loop would update model_manager.version
        # (visible on /health) but never actually reach a live request.
        if model_manager is not None and model_manager.is_loaded:
            probability   = float(model_manager.predict_proba(transformed_features)[0, 1])
            prediction    = int(model_manager.predict(transformed_features)[0])
            model_version = str(model_manager.version)
        else:
            probability   = float(pred_model.predict_proba(transformed_features)[0, 1])
            prediction    = int(pred_model.predict(transformed_features)[0])
            model_version = "dev"

        # RECORD LATENCY and INCREMENT COUNTER
        PREDICTION_LATENCY.observe(time.time() - start) 
        PREDICTIONS_TOTAL.labels(
            tenant_id=tenant,
            prediction=str(prediction)
        ).inc()

        # Log the prediction event in ClickHouse for future analysis
        try:
            log_prediction(
                tenant_id=tenant,
                app_id=payload.ID,
                prob=probability,
                pred=prediction,
                model_version=model_version,
            )
        except Exception as log_err:
            print(f"[WARNING] Logging failed: {log_err}")

        return {
            "application_id": payload.ID,
            "default_prediction": prediction,
            "default_probability": round(probability, 4),
            "status": "success"
        }
        
    except Exception as e:
        PREDICTION_LATENCY.observe(time.time() - start)  # RECORD LATENCY even on failure
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failure: {str(e)}"
        )
    
@app.post("/v1/batch/upload")
async def batch_upload(
    file: UploadFile = File(...),
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme)
):
    tenant = verify_token(token)

    start = time.time()

    content = await file.read()

    # Validate file content checks
    if len(content) == 0:
        BATCH_UPLOADS_TOTAL.labels(tenant_id=tenant, status="invalid").inc()   # ← track
        raise HTTPException(status_code=400, detail="CSV file is empty")

    try:
        df = pd.read_csv(BytesIO(content))
        row_count = len(df)
    except Exception:
        BATCH_UPLOADS_TOTAL.labels(tenant_id=tenant, status="invalid").inc()   # ← track
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    # empty file check
    if row_count == 0:
        BATCH_UPLOADS_TOTAL.labels(tenant_id=tenant, status="invalid").inc()   # ← track
        raise HTTPException(status_code=400, detail="CSV file has no rows")

    # Required-column check — mirrors LoanApplication's required fields. Without
    # this, a CSV missing e.g. loan_amount would still get queued, and
    # FeaturePipeline.transform()'s reindex() silently fills the whole absent
    # column with the training-set median rather than failing loudly.
    required_columns = {"ID", "year", "loan_amount", "property_value", "income", "Credit_Score"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        BATCH_UPLOADS_TOTAL.labels(tenant_id=tenant, status="invalid").inc()   # ← track
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required column(s): {', '.join(sorted(missing_columns))}",
        )

    # Check and increment tenant's monthly quota
    try:
        check_and_increment_quota(tenant, row_count, monthly_limit=get_plan_quota(tenant))
    except Exception:
        BATCH_UPLOADS_TOTAL.labels(tenant_id=tenant, status="quota_exceeded").inc()  # ← track
        raise HTTPException(status_code=429, detail="Quota exceeded")

    # Write to MinIO
    job_id = str(uuid.uuid4())
    object_name = f"{tenant}/{datetime.utcnow().strftime('%Y%m%d')}/{job_id}.csv"

    try:
        if not minio_client.bucket_exists("raw-landing"):
            minio_client.make_bucket("raw-landing")
        minio_client.put_object(
            "raw-landing", object_name,
            data=BytesIO(content),
            length=len(content),
            content_type="text/csv"
        )
    except Exception as e:
        BATCH_UPLOADS_TOTAL.labels(tenant_id=tenant, status="minio_error").inc()
        BATCH_UPLOAD_LATENCY.observe(time.time() - start)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload to MinIO: {str(e)}"
        )

    # ── Track job status in Redis so /v1/batch/results/{job_id} can poll it ──
    redis_client.set(f"job:{job_id}:status", "queued", ex=86400)
    redis_client.set(f"job:{job_id}:tenant", tenant,   ex=86400)

    # ── Index this job under the tenant so GET /v1/batch/jobs can list it ────
    redis_client.lpush(f"jobs:{tenant}", job_id)
    redis_client.expire(f"jobs:{tenant}", 60 * 60 * 24 * 30)

    # ── Enqueue Celery scoring task ───────────────────────────────────────────
    if _celery_app:
        try:
            _celery_app.send_task("process_batch", kwargs={
                "job_id":      job_id,
                "object_name": object_name,
                "tenant_id":   tenant,
            })
        except Exception as e:
            print(f"[WARNING] Celery enqueue failed for job {job_id}: {e}")

    # ── Record success metrics ────────────────────────────────────────────────
    BATCH_UPLOADS_TOTAL.labels(tenant_id=tenant, status="success").inc()
    BATCH_ROWS_TOTAL.labels(tenant_id=tenant).inc(row_count)
    BATCH_UPLOAD_LATENCY.observe(time.time() - start)

    return {
        "job_id":        job_id,
        "tenant_id":     tenant,
        "rows_received": row_count,
        "object":        object_name,
        "status":        "queued"
    }

@app.get("/health", status_code=status.HTTP_200_OK, summary="Health Check API")
async def health_check():
    return {
        "status":           "healthy",
        "pipeline_fitted":  feature_pipeline.is_fitted,
        "model_version":    str(model_manager.version or "dev"),
    }


@app.get("/v1/batch/results/{job_id}", summary="Poll batch job status and retrieve results")
async def batch_results(
    job_id: str,
    token:  Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    """
    Returns the current status of a batch scoring job.
    When status is 'complete', also returns a presigned MinIO download URL
    (valid for 1 hour) pointing to the results CSV.
    """
    tenant = verify_token(token)

    # Ownership check — tenants can only see their own jobs
    job_tenant = redis_client.get(f"job:{job_id}:tenant")
    if job_tenant is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_tenant != tenant:
        raise HTTPException(status_code=403, detail="Access denied")

    job_status = redis_client.get(f"job:{job_id}:status")
    response   = {"job_id": job_id, "status": job_status}

    if job_status == "complete":
        result_object = redis_client.get(f"job:{job_id}:result_object")
        rows_scored   = redis_client.get(f"job:{job_id}:rows_scored")
        download_url  = minio_public_client.presigned_get_object(
            "batch-results", result_object, expires=timedelta(hours=1)
        )
        response.update({
            "rows_scored":   int(rows_scored or 0),
            "download_url":  download_url,
        })

    elif job_status == "failed":
        response["error"] = redis_client.get(f"job:{job_id}:error")

    return response


@app.get("/v1/batch/jobs", summary="List this tenant's batch job history")
async def batch_jobs(
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    """Most recent 50 batch jobs for the caller's tenant — job_id + status
    only; call GET /v1/batch/results/{job_id} for the full detail (download
    URL, rows scored, error) of any one of them."""
    tenant = verify_token(token)
    job_ids = redis_client.lrange(f"jobs:{tenant}", 0, 49)
    jobs = [
        {"job_id": job_id, "status": redis_client.get(f"job:{job_id}:status")}
        for job_id in job_ids
    ]
    return {"jobs": jobs}


@app.get("/v1/usage", summary="This tenant's current-month prediction quota usage")
async def usage(
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    tenant = verify_token(token)
    from app.db import get_tenant_plan
    plan   = get_tenant_plan(tenant)
    month  = datetime.utcnow().strftime("%Y_%m")
    used   = int(redis_client.get(f"quota:{tenant}:{month}") or 0)
    return {
        "tenant_id": tenant,
        "month":     datetime.utcnow().strftime("%Y-%m"),
        "used":      used,
        "limit":     PLANS.get(plan, PLANS[DEFAULT_PLAN])["monthly_quota"],
        "plan":      plan,
    }


@app.get("/v1/plans", summary="Available subscription tiers")
async def list_plans(
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    verify_token(token)
    return {"plans": [{"id": plan_id, **details} for plan_id, details in PLANS.items()]}


class PlanChangeRequest(BaseModel):
    plan: str


@app.post("/v1/tenant/plan", summary="Change this tenant's subscription plan")
async def change_plan(
    payload: PlanChangeRequest,
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    tenant = verify_token(token)
    if payload.plan not in PLANS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown plan '{payload.plan}'. Valid plans: {', '.join(PLANS)}",
        )
    from app.db import get_tenant_row, set_tenant_plan
    if payload.plan == "free":
        # Switching straight to free while Paystack still has an active
        # subscription would leave the tenant billed for a plan they no
        # longer have entitlement to — cancellation has to go through the
        # billing portal (GET /v1/tenant/billing-portal) first; the webhook
        # is what actually flips plan+subscription_status once that
        # cancellation lands, not this endpoint.
        row = get_tenant_row(tenant)
        if row and row.get("subscription_status") == "active":
            raise HTTPException(
                status_code=400,
                detail="You have an active paid subscription — cancel it from the "
                        "billing portal (GET /v1/tenant/billing-portal) rather than "
                        "switching plans directly, so billing and quota stay in sync.",
            )
    updated = set_tenant_plan(tenant, payload.plan)
    if not updated:
        raise HTTPException(
            status_code=400,
            detail="Plan management requires a registered account — static API "
                    "tokens aren't tied to a tenant row. Register via /auth/register.",
        )
    return {
        "tenant_id": tenant,
        "plan": payload.plan,
        "monthly_quota": PLANS[payload.plan]["monthly_quota"],
    }


@app.post("/v1/tenant/checkout", summary="Start a Paystack checkout for a paid plan")
async def start_checkout(payload: PlanChangeRequest, user: dict = Depends(get_current_user)):
    """Requires a real JWT, not a static token or API key — Paystack needs
    an email address for the checkout, which only a logged-in user has.
    Enterprise has no fixed price and is sales-assisted, so it's never in
    CHECKOUT_PLANS and can't be started here."""
    from app.plans import CHECKOUT_PLANS
    if payload.plan not in CHECKOUT_PLANS:
        raise HTTPException(
            status_code=400,
            detail=f"'{payload.plan}' isn't sold through checkout. Available: {', '.join(CHECKOUT_PLANS)}",
        )
    from app.payments import PAYSTACK_PLAN_CODES, initialize_transaction, usd_to_paystack_subunits
    plan_code = PAYSTACK_PLAN_CODES.get(payload.plan)
    if not plan_code:
        raise HTTPException(
            status_code=500,
            detail=f"No Paystack plan code configured for '{payload.plan}' — run "
                    "scripts/setup_paystack_plans.py and set PAYSTACK_PLAN_"
                    f"{payload.plan.upper()} in .env.",
        )
    from app.db import get_cursor
    with get_cursor() as cur:
        cur.execute("SELECT email FROM app.users WHERE id = %s", (user["sub"],))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    # Paystack overwrites callback_url's query string with its own
    # trxref/reference params on redirect — any query string we pass here
    # doesn't survive the round trip, so the frontend detects "returned
    # from checkout" from those Paystack-added params instead.
    frontend_origin = os.environ.get("FRONTEND_ORIGINS", "http://localhost:5173").split(",")[0]
    try:
        data = initialize_transaction(
            email=row["email"],
            plan_code=plan_code,
            amount_subunits=usd_to_paystack_subunits(PLANS[payload.plan]["price_amount"]),
            callback_url=f"{frontend_origin}/plans",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Paystack checkout initialization failed: {e}")
    return {"authorization_url": data["authorization_url"]}


@app.post("/webhooks/paystack", summary="Paystack subscription event webhook")
async def paystack_webhook(request: Request):
    from app.payments import verify_webhook_signature
    raw_body = await request.body()
    signature = request.headers.get("x-paystack-signature", "")
    if not verify_webhook_signature(raw_body, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    event = await request.json()
    event_type = event.get("event")
    data = event.get("data", {})

    try:
        from app.db import get_tenant_id_by_email, get_tenant_row, set_tenant_subscription
        from app.payments import PAYSTACK_PLAN_CODES
        if event_type == "subscription.create":
            # Fires with a real subscription_code already in its payload —
            # e.g. echoing back the subscription this same handler creates
            # below in response to charge.success. Idempotent either way.
            email = data.get("customer", {}).get("email")
            plan_code = data.get("plan", {}).get("plan_code")
            tenant_id = get_tenant_id_by_email(email) if email else None
            plan_id = next((p for p, code in PAYSTACK_PLAN_CODES.items() if code == plan_code), None)
            if tenant_id and plan_id:
                set_tenant_subscription(
                    tenant_id, plan_id,
                    customer_code=data.get("customer", {}).get("customer_code"),
                    subscription_code=data.get("subscription_code", ""),
                    status="active",
                )
        elif event_type == "charge.success":
            # Attaching a `plan` to /transaction/initialize only charges
            # once for that plan's amount — confirmed against a real
            # Paystack account, which had no subscription record at all
            # after a successful plan-attached charge. The recurring
            # subscription has to be created explicitly using the card
            # authorization from this first successful charge.
            email = data.get("customer", {}).get("email")
            plan_code = data.get("plan", {}).get("plan_code")
            tenant_id = get_tenant_id_by_email(email) if email else None
            plan_id = next((p for p, code in PAYSTACK_PLAN_CODES.items() if code == plan_code), None)
            if tenant_id and plan_id:
                existing = get_tenant_row(tenant_id) or {}
                if existing.get("subscription_status") == "active" and existing.get("paystack_subscription_code"):
                    # A renewal charge (or a redelivered webhook) against a
                    # subscription that already exists — every charge after
                    # the first is Paystack's own recurring billing, not a
                    # new signup, so there's nothing new to create.
                    pass
                else:
                    customer_code = data.get("customer", {}).get("customer_code")
                    auth_code = data.get("authorization", {}).get("authorization_code")
                    subscription_code = ""
                    if customer_code and auth_code:
                        from app.payments import create_subscription
                        try:
                            sub = create_subscription(customer_code, plan_code, auth_code)
                            subscription_code = sub.get("subscription_code", "")
                        except Exception as e:
                            print(f"[WARNING] Failed to create Paystack subscription for tenant={tenant_id}: {e}")
                    set_tenant_subscription(
                        tenant_id, plan_id,
                        customer_code=customer_code,
                        subscription_code=subscription_code,
                        status="active",
                    )
        elif event_type in ("subscription.disable", "subscription.not_renew"):
            email = data.get("customer", {}).get("email")
            tenant_id = get_tenant_id_by_email(email) if email else None
            if tenant_id:
                # Keep the existing customer/subscription codes on file (for
                # the billing portal / re-subscribe history) rather than
                # overwriting them with whatever this event happens to
                # carry — only the entitlement and status actually change.
                existing = get_tenant_row(tenant_id) or {}
                set_tenant_subscription(
                    tenant_id, DEFAULT_PLAN,
                    customer_code=existing.get("paystack_customer_code"),
                    subscription_code=existing.get("paystack_subscription_code"),
                    status="canceled",
                )
    except Exception as e:
        # Signature already verified — this is a genuine Paystack event.
        # Log and still return 200 so Paystack doesn't retry-storm us over
        # a transient DB hiccup; same graceful-degradation style as
        # ClickHouse logging elsewhere in this file.
        print(f"[WARNING] Paystack webhook processing failed for event={event_type}: {e}")

    return {"status": "received"}


@app.get("/v1/tenant/billing-portal", summary="Get this tenant's Paystack subscription-management link")
async def billing_portal(user: dict = Depends(get_current_user)):
    from app.db import get_tenant_row
    tenant_row = get_tenant_row(user["tenant_id"])
    subscription_code = tenant_row.get("paystack_subscription_code") if tenant_row else None
    if not subscription_code:
        raise HTTPException(
            status_code=400,
            detail="No active subscription to manage — switch to a paid plan first.",
        )
    from app.payments import get_manage_link
    try:
        link = get_manage_link(subscription_code)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Paystack billing-portal request failed: {e}")
    return {"link": link}


@app.get("/v1/predictions/history", summary="This tenant's recent predictions")
async def predictions_history(
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    tenant = verify_token(token)
    client = get_ch_client()
    if client is None:
        return {"predictions": []}
    try:
        result = client.query(
            "SELECT application_id, probability, prediction, model_version, ts "
            "FROM prediction_logs WHERE tenant_id = {tenant_id:String} "
            "ORDER BY ts DESC LIMIT 50",
            parameters={"tenant_id": tenant},
        )
        predictions = [
            {
                "application_id": row[0],
                "probability":    row[1],
                "prediction":     row[2],
                "model_version":  row[3],
                "timestamp":      row[4].isoformat(),
            }
            for row in result.result_rows
        ]
    except Exception as e:
        print(f"[WARNING] predictions history query failed: {e}")
        predictions = []
    return {"predictions": predictions}


@app.get("/admin/tenants", summary="Cross-tenant usage summary (admin only)")
async def admin_tenants(admin_user: dict = Depends(require_admin)):
    from app.db import get_cursor

    with get_cursor() as cur:
        cur.execute(
            "SELECT tenant_id, name, plan, invite_code, created_at FROM app.tenants "
            "ORDER BY created_at DESC"
        )
        tenants = cur.fetchall()

    month = datetime.utcnow().strftime("%Y_%m")
    result = []
    for t in tenants:
        with get_cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS n FROM app.users WHERE tenant_id = %s",
                (t["tenant_id"],),
            )
            user_count = cur.fetchone()["n"]
        used = int(redis_client.get(f"quota:{t['tenant_id']}:{month}") or 0)
        plan = t.get("plan") or DEFAULT_PLAN
        result.append({
            "tenant_id":   t["tenant_id"],
            "name":        t["name"],
            "plan":        plan,
            "created_at":  t["created_at"].isoformat(),
            "user_count":  user_count,
            "quota_used":  used,
            "quota_limit": PLANS.get(plan, PLANS[DEFAULT_PLAN])["monthly_quota"],
        })
    return {"tenants": result}


@app.get("/v1/labeled-data/history", summary="This tenant's recent labeled-data uploads")
async def labeled_data_history(
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    tenant = verify_token(token)
    client = get_ch_client()
    if client is None:
        return {"uploads": []}
    try:
        result = client.query(
            "SELECT object_name, rows_received, ts "
            "FROM labeled_uploads WHERE tenant_id = {tenant_id:String} "
            "ORDER BY ts DESC LIMIT 50",
            parameters={"tenant_id": tenant},
        )
        uploads = [
            {
                "object_name":   row[0],
                "rows_received": row[1],
                "timestamp":     row[2].isoformat(),
            }
            for row in result.result_rows
        ]
    except Exception as e:
        print(f"[WARNING] labeled-data history query failed: {e}")
        uploads = []
    return {"uploads": uploads}


@app.post(
    "/v1/labeled-data",
    response_model=LabeledDataSchema,
    status_code=status.HTTP_200_OK,
    summary="Upload labeled actuals CSV for drift detection and retraining",
)
async def upload_labeled_data(
    file:  UploadFile = File(...),
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    """
    The business uploads a CSV containing actual loan outcomes (ground truth)
    alongside the original feature columns.  The CSV must include either an
    'actual_outcome' column (0 = no default, 1 = default) or a 'Status' column.

    The file is stored in MinIO labeled-data bucket.  The daily ingestion DAG
    pulls these CSVs, computes PSI drift against the training baseline, and —
    if drift >= 0.2 — commits the labeled data to DVC so the weekly retrain
    picks it up.
    """
    tenant = verify_token(token)

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    try:
        df = pd.read_csv(BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV has no rows")

    if "actual_outcome" not in df.columns and "Status" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain an 'actual_outcome' or 'Status' column",
        )

    # Write to MinIO labeled-data bucket
    upload_id   = str(uuid.uuid4())
    object_name = f"{tenant}/{datetime.utcnow().strftime('%Y%m%d')}/{upload_id}_labeled.csv"

    try:
        if not minio_client.bucket_exists("labeled-data"):
            minio_client.make_bucket("labeled-data")
        minio_client.put_object(
            "labeled-data", object_name,
            data=BytesIO(content), length=len(content),
            content_type="text/csv",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MinIO write failed: {str(e)}")

    log_labeled_upload(tenant, object_name, len(df))

    return LabeledDataSchema(
        object=object_name,
        tenant_id=tenant,
        rows_received=len(df),
        status="stored",
    )


