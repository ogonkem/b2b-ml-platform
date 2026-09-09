"""
rule_engine/main.py
Thin FastAPI wrapper around rule_engine/engine.py's pure decision logic.

Deliberately its own service rather than an endpoint inside rag_harness's
FastAPI app, even though the two would share nothing but the auth import:
this is an audit-critical, must-be-deterministic code path with genuinely
zero I/O dependencies of its own (no DB, no object storage, no queue, no
LLM/embedding calls) — see rule_engine/engine.py's module docstring.
Bundling it into rag_harness would make its availability depend on
Postgres/MinIO/Redis/Celery/OpenAI all being healthy, none of which this
endpoint actually needs. The cost of that isolation is operational, not
architectural: one more container, port, and Dockerfile to run
(Dockerfile.rule_engine, requirements-rule-engine.txt) versus zero extra
infrastructure if it had lived inside rag_harness.

Auth reuses app.auth.verify_token directly, same as rag_harness — a bearer
value that already authenticates against Selastone's API resolves to the
same tenant_id here with zero extra setup.
"""
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from app.auth import security_scheme, verify_token
from rule_engine.engine import UnknownTenantError, decide
from rule_engine.thresholds import RULE_ENGINE_VERSION

app = FastAPI(
    title="Selastone Rule Engine",
    version=RULE_ENGINE_VERSION,
    description="Deterministic, code-only loan decisioning — no LLM, no external API calls.",
)


class DecideRequest(BaseModel):
    tenant_id: str
    risk_score: float
    shap_factors: List[Dict[str, Any]] = Field(default_factory=list)
    loan_type: str
    applicant_profile: Dict[str, Any] = Field(default_factory=dict)
    use_fallback: bool = Field(
        default=False,
        description="Bypass tenant-specific thresholds and use the generic, "
                    "risk-score-only fallback bands (see rule_engine/thresholds.py's "
                    "FALLBACK_RISK_BANDS) — set by agent/graph.py when rag_harness "
                    "found no policy documents for this tenant.",
    )


class TriggeredThreshold(BaseModel):
    rule: str
    applicant_value: Any
    threshold: Any
    loan_type: str


class DecideResponse(BaseModel):
    decision: str
    triggered_thresholds: List[TriggeredThreshold]
    rule_engine_version: str


@app.post("/v1/decide", response_model=DecideResponse, summary="Deterministic rule-based loan decision")
async def decide_endpoint(
    payload: DecideRequest,
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    tenant = verify_token(token)
    # Same convention as rag_harness's /v1/retrieve: tenant_id comes from
    # the authenticated token, the body's copy is only ever checked against
    # it, never used as the actual identity for the decision.
    if payload.tenant_id != tenant:
        raise HTTPException(status_code=403, detail="tenant_id does not match authenticated token")

    try:
        return decide(
            tenant_id=payload.tenant_id,
            risk_score=payload.risk_score,
            shap_factors=payload.shap_factors,
            loan_type=payload.loan_type,
            applicant_profile=payload.applicant_profile,
            use_fallback=payload.use_fallback,
        )
    except UnknownTenantError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health", status_code=status.HTTP_200_OK, summary="Health Check")
async def health_check():
    return {"status": "healthy"}
