import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Security, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import pickle, json, os
import hmac, hashlib
import redis
from datetime import datetime
import clickhouse_connect
import uuid
from minio import Minio
from io import BytesIO
from app.model_manager import ModelManager

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

# -------------------------------------------------------------------------
# AUTHENTICATION
# -------------------------------------------------------------------------

VALID_TOKENS = set(os.environ.get("API_TOKENS", "dev-token").split(","))

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme)):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Use Bearer token.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    token = credentials.credentials
    if token not in VALID_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API token."
        )
    
    return token


# -------------------------------------------------------------------------
# PYDANTIC INPUT DATA SCHEMA
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
        xgb_model = pickle.load(f)

    with open(PIPELINE_PATH, 'rb') as f:
        feature_pipeline = pickle.load(f)

    print(f"✓ Loaded real model    → {MODEL_PATH}")
    print(f"✓ Loaded real pipeline → {PIPELINE_PATH}")

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

    xgb_model = xgb.XGBClassifier()
    mock_y = np.array([0, 1, 0])
    mock_X = feature_pipeline.transform(baseline_historical_df)
    xgb_model.fit(mock_X, mock_y)

    print("⚠️  Running in development mode — using mock model and pipeline")
    print("   Set USE_REAL_ARTEFACTS=true (or in .env) to load real artefacts")

# ── ModelManager — start AFTER model is loaded ────────────────────────────
# Only polls MLflow when real artefacts are in use
# In dev mode it starts but load_latest() will find nothing and skip cleanly

model_manager = ModelManager(
    model_name="selastone_credit_scorer",
    poll_interval=int(os.environ.get("MODEL_POLL_INTERVAL", 60))
)
model_manager.load_latest()    # try immediately on startup
model_manager.start_polling()  # background thread checks every 60s

# -------------------------------------------------------------------------
# Redis client initialization (for future stateful features like caching or rate limiting)
# -------------------------------------------------------------------------
redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    decode_responses=True
)

def check_and_increment_quota(tenant_id: str, monthly_limit: int = 1000):
    """
    Increments this tenant's monthly prediction counter in Redis.
    (fixed window approach for simplicity: counts per calendar month, resets automatically with TTL)
    Key resets automatically after ~1 month via TTL.
    Raises 429 if limit exceeded.
    """
    key = f"quota:{tenant_id}:{datetime.utcnow().strftime('%Y_%m')}"
    current = redis_client.incr(key)
    if current == 1:
        # New key created, set TTL to expire after ~1 month to reset quota automatically
        redis_client.expire(key, 60 * 60 * 24 * 32)  # ~1 month TTL
    if current > monthly_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, 
            detail="Monthly prediction quota exceeded"
        )
    return current

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

# --------------------------------------------------------------------------   
# MinIO client initialization (for future model artefact storage and retrieval)
# -------------------------------------------------------------------------
minio_client = Minio(
    f"{os.environ.get('MINIO_HOST', 'localhost')}:{os.environ.get('MINIO_PORT', 9000)}",
    access_key=os.environ.get("MINIO_ROOT_USER"),
    secret_key=os.environ.get("MINIO_ROOT_PASSWORD"),
    secure=False
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
    check_and_increment_quota(tenant)
    try:
        # Convert incoming JSON payload to a pandas DataFrame
        input_data = pd.DataFrame([payload.model_dump()])
        
        # Extract features through processing pipeline
        transformed_features = feature_pipeline.transform(input_data)
        
        # Calculate raw probability score and concrete decisions
        probability = float(xgb_model.predict_proba(transformed_features)[0, 1])
        prediction = int(xgb_model.predict(transformed_features)[0])

        # Log the prediction event in ClickHouse for future analysis
        log_prediction(
            tenant_id=tenant, 
            app_id=payload.ID, 
            prob=probability, 
            pred=prediction
        )
        
        return {
            "application_id": payload.ID,
            "default_prediction": prediction,
            "default_probability": round(probability, 4),
            "status": "success"
        }
        
    except Exception as e:
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

    content = await file.read()

    # Validate file content checks
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    try:
        df = pd.read_csv(BytesIO(content))
        row_count = len(df)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file")
    
    # empty file check
    if row_count == 0:
        raise HTTPException(status_code=400, detail="CSV file has no rows")

    # Check and increment tenant's monthly quota
    check_and_increment_quota(tenant, row_count)

    # Write to MinIO
    job_id = str(uuid.uuid4())
    object_name = f"{tenant}/{datetime.utcnow().strftime('%Y%m%d')}/{job_id}.csv"

    try:
        minio_client.put_object(
            "raw-landing", object_name,
            data=BytesIO(content),
            length=len(content),
            content_type="text/csv"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to upload to MinIO: {str(e)}"
        )

    return {
        "job_id":         job_id,
        "tenant_id":      tenant,
        "rows_received":  row_count,
        "object":         object_name,
        "status":         "queued"
    }

@app.get("/health", status_code=status.HTTP_200_OK, summary="Health Check API")
async def health_check():
    return {"status": "healthy", "pipeline_fitted": feature_pipeline.is_fitted}


