"""
agent/main.py
FastAPI wrapper exposing POST /v1/agent/assess — orchestrates Selastone's
/v1/predict, rag_service's /v1/retrieve, and rule_engine's /v1/decide via
the fixed LangGraph pipeline in agent/graph.py, then an LLM synthesis step
and an audit write (agent.agent_decisions).

Its own service, like rag_service and rule_engine — depends on Postgres
(for the audit trail) and the three services it calls over HTTP, plus
app.auth for auth reuse. All three downstream calls are real HTTP calls
(not in-process function calls), since each lives in its own deployment.
"""
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from agent.clients import UpstreamServiceError
from agent.db import get_cursor, init_schema
from agent.graph import GRAPH
from app.auth import security_scheme, verify_token

app = FastAPI(
    title="Selastone Decision Agent",
    version="0.1.0",
    description="LangGraph orchestrator: predict -> retrieve policy -> rule-based decide -> LLM narrative -> audit.",
)

# The frontend SPA calls this service directly (pages/AgentAssess.tsx) —
# same CORS setup as app/main.py and rag_service/main.py.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FRONTEND_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    init_schema()
    print("[OK] agent schema ready")
except Exception as e:
    print(f"[WARN] agent schema init failed: {e}")


class AssessRequest(BaseModel):
    tenant_id: str
    loan_type: str
    applicant: Dict[str, Any] = Field(..., description="LoanApplication-shaped payload forwarded to /v1/predict")
    applicant_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra fields rule_engine reads (dti, is_first_time_borrower, requested_amount)",
    )


class PolicyChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_version: str
    section_ref: Optional[str] = None
    chunk_text: str
    score: float


class AssessResponse(BaseModel):
    decision: str
    risk_score: float
    statistical_factors: List[Dict[str, Any]]
    policy_basis: List[Dict[str, Any]]
    # The actual retrieved policy text (with section_ref) this decision
    # cites — not just which thresholds fired (policy_basis). Empty when
    # policy_aligned is false (the fallback path retrieved nothing).
    policy_chunks: List[PolicyChunk] = Field(default_factory=list)
    narrative: str
    policy_aligned: bool


@app.post("/v1/agent/assess", response_model=AssessResponse, summary="Full agentic loan assessment")
async def assess(
    payload: AssessRequest,
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    tenant = verify_token(token)
    # Same convention as rag_service/rule_engine: tenant_id comes from the
    # authenticated token, the body's copy is only ever checked against it.
    if payload.tenant_id != tenant:
        raise HTTPException(status_code=403, detail="tenant_id does not match authenticated token")

    initial_state = {
        "tenant_id": tenant,
        "token": token.credentials,
        "applicant": payload.applicant,
        "loan_type": payload.loan_type,
        "applicant_profile": payload.applicant_profile,
    }

    try:
        final_state = GRAPH.invoke(initial_state)
    except UpstreamServiceError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return AssessResponse(
        decision=final_state["decision"],
        risk_score=final_state["risk_score"],
        statistical_factors=final_state.get("shap_factors", []),
        policy_basis=final_state.get("triggered_thresholds", []),
        policy_chunks=final_state.get("retrieved_chunks", []),
        narrative=final_state["narrative"],
        policy_aligned=final_state.get("policy_aligned", False),
    )


class DecisionRecord(BaseModel):
    id: str
    tenant_id: str
    application_ref: str
    risk_score: Optional[float] = None
    model_version: Optional[str] = None
    rule_engine_version: Optional[str] = None
    decision: Optional[str] = None
    triggered_thresholds: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    policy_doc_version: Optional[str] = None
    policy_aligned: Optional[bool] = None
    llm_narrative: Optional[str] = None
    created_at: datetime


def _row_to_record(row: dict) -> DecisionRecord:
    return DecisionRecord(
        id=str(row["id"]),
        tenant_id=row["tenant_id"],
        application_ref=row["application_ref"],
        risk_score=row["risk_score"],
        model_version=row["model_version"],
        rule_engine_version=row["rule_engine_version"],
        decision=row["decision"],
        triggered_thresholds=row["triggered_thresholds"] or [],
        retrieved_chunk_ids=[str(cid) for cid in (row["retrieved_chunk_ids"] or [])],
        policy_doc_version=row["policy_doc_version"],
        policy_aligned=row["policy_aligned"],
        llm_narrative=row["llm_narrative"],
        created_at=row["created_at"],
    )


@app.get(
    "/v1/agent/decisions/{application_ref}",
    response_model=List[DecisionRecord],
    summary="Full audit trace for a past decision",
)
async def get_decision_by_application_ref(
    application_ref: str,
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    """A given application_ref can have more than one row if it was
    reassessed — returns every decision ever made for it, newest first,
    rather than silently picking just the latest and hiding the history."""
    tenant = verify_token(token)
    with get_cursor() as cur:
        cur.execute(
            """SELECT * FROM agent.agent_decisions
               WHERE tenant_id = %s AND application_ref = %s
               ORDER BY created_at DESC""",
            (tenant, application_ref),
        )
        rows = cur.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No decision found for this application_ref")
    return [_row_to_record(row) for row in rows]


@app.get(
    "/v1/agent/decisions",
    response_model=List[DecisionRecord],
    summary="List this tenant's decisions, optionally filtered by policy_doc_version",
)
async def list_decisions(
    tenant_id: Optional[str] = Query(
        default=None, description="Must match the authenticated token if provided — never used as the actual filter on its own."
    ),
    policy_doc_version: Optional[str] = Query(
        default=None, description="e.g. \"show every decision made under policy version Y\""
    ),
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    tenant = verify_token(token)
    # Same convention as the rest of this session: tenant_id always comes
    # from the authenticated token. A query-string copy is only ever
    # checked against it, never used as the actual WHERE-clause value.
    if tenant_id is not None and tenant_id != tenant:
        raise HTTPException(status_code=403, detail="tenant_id does not match authenticated token")

    query  = "SELECT * FROM agent.agent_decisions WHERE tenant_id = %s"
    params: List[Any] = [tenant]
    if policy_doc_version is not None:
        query += " AND policy_doc_version = %s"
        params.append(policy_doc_version)
    query += " ORDER BY created_at DESC"

    with get_cursor() as cur:
        cur.execute(query, tuple(params))
        rows = cur.fetchall()
    return [_row_to_record(row) for row in rows]


@app.get("/health", status_code=status.HTTP_200_OK, summary="Health Check")
async def health_check():
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1")
        db_connected = True
    except Exception:
        db_connected = False
    return {"status": "healthy" if db_connected else "degraded", "db_connected": db_connected}
