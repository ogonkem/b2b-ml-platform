import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import pickle, json, os

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
    try:
        # Convert incoming JSON payload to a pandas DataFrame
        input_data = pd.DataFrame([payload.model_dump()])
        
        # Extract features through processing pipeline
        transformed_features = feature_pipeline.transform(input_data)
        
        # Calculate raw probability score and concrete decisions
        probability = float(xgb_model.predict_proba(transformed_features)[0, 1])
        prediction = int(xgb_model.predict(transformed_features)[0])
        
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

@app.get("/health", status_code=status.HTTP_200_OK, summary="Health Check API")
async def health_check():
    return {"status": "healthy", "pipeline_fitted": feature_pipeline.is_fitted}


