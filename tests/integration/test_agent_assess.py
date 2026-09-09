# Requires the full docker stack running: docker compose up -d
#
# Additionally requires a real OPENAI_API_KEY in .env — POST /v1/agent/assess
# always calls rag_service's /v1/retrieve (embeds the query) and always ends
# with an LLM synthesis call (agent/graph.py's node_synthesize). Without a
# real key, every test here is skipped rather than failing.
#
# Also requires the three demo tenant docs already ingested — this file
# assumes tests/integration/test_rag_service.py has run first in the same
# session (pytest collects/runs files in a session, and ingestion is left
# in place afterward; see that file's own ingested_docs fixture). If run in
# isolation with nothing ingested yet, these tests still pass: policy_aligned
# is simply False and rule_engine falls back to FALLBACK_RISK_BANDS, which
# is itself one of the scenarios worth covering (see the standalone
# fallback test at the bottom of this file).
import time
from pathlib import Path

import httpx
import psycopg2
import pytest
from dotenv import dotenv_values
from psycopg2.extras import RealDictCursor

_env = dotenv_values(Path(__file__).resolve().parent.parent.parent / ".env")

OPENAI_API_KEY = _env.get("OPENAI_API_KEY", "")
pytestmark = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="requires a real OPENAI_API_KEY in .env — /v1/agent/assess always "
           "retrieves (embeds the query) and always synthesizes a narrative via LLM",
)

AGENT_URL = "http://localhost:8003"

# Same three demo tenants as test_rag_service.py, seeded in
# rule_engine/thresholds.py. Distinct fake applicant IDs per tenant so each
# tenant's decision row is independently addressable via
# GET /v1/agent/decisions/{application_ref} without colliding with another
# tenant's row (application_ref is unique per tenant, not globally, but
# using distinct IDs anyway keeps this file's assertions unambiguous).
TENANTS = ["commercial_bank", "microfinance_sacco", "informal_digital_lender"]
APPLICANT_ID = {"commercial_bank": 900001, "microfinance_sacco": 900002, "informal_digital_lender": 900003}

# A single deliberately extreme profile: 650k loan against a 700k property,
# 3k monthly income, 580 credit score. Empirically produces risk_score≈98
# against the real trained model (verified earlier in this session) — high
# enough to breach every tenant's hard cap/highest risk band in
# rule_engine/thresholds.py AND FALLBACK_RISK_BANDS' >70 reject boundary.
# Using one profile for all three tenants means the "reject" outcome is
# deterministic regardless of whether real per-tenant policy retrieval
# succeeds or the generic fallback fires — this test is about the pipeline
# wiring end-to-end, not about exercising every decision band.
HIGH_RISK_APPLICANT = {
    "ID": None,     # filled in per tenant below
    "year": 2019,
    "loan_amount": 650000.0,
    "property_value": 700000.0,
    "income": 3000.0,
    "Credit_Score": 580.0,
}
HIGH_RISK_PROFILE = {"dti": 0.99, "requested_amount": 650000.0, "is_first_time_borrower": True}


def _pg_conn():
    return psycopg2.connect(
        host="localhost",
        port=int(_env.get("POSTGRES_PORT", 5432)),
        user=_env.get("POSTGRES_USER"),
        password=_env.get("POSTGRES_PASSWORD"),
        dbname=_env.get("POSTGRES_DB"),
    )


def _assess(tenant: str) -> dict:
    applicant = dict(HIGH_RISK_APPLICANT, ID=APPLICANT_ID[tenant])
    resp = httpx.post(
        f"{AGENT_URL}/v1/agent/assess",
        headers={"Authorization": f"Bearer {tenant}"},
        json={
            "tenant_id": tenant,
            "loan_type": "real_estate_payment_plan",
            "applicant": applicant,
            "applicant_profile": HIGH_RISK_PROFILE,
        },
        timeout=60.0,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.parametrize("tenant", TENANTS)
def test_assess_end_to_end_rejects_and_cites_policy(tenant):
    result = _assess(tenant)

    assert result["decision"] == "reject"
    assert result["risk_score"] > 70
    assert len(result["statistical_factors"]) > 0
    assert len(result["policy_basis"]) >= 1
    assert result["narrative"].strip() != ""

    # Whether or not this tenant's policy doc had been ingested when this
    # test ran, the response must always say so honestly via policy_aligned
    # — never claim alignment while presenting a fallback-only decision.
    if result["policy_aligned"]:
        assert len(result["policy_chunks"]) > 0
        for chunk in result["policy_chunks"]:
            assert chunk["doc_id"]
            assert chunk["chunk_text"].strip() != ""
    else:
        assert result["policy_chunks"] == []

    application_ref = str(APPLICANT_ID[tenant])

    # GET /v1/agent/decisions/{application_ref} reflects the same decision
    fetch = httpx.get(
        f"{AGENT_URL}/v1/agent/decisions/{application_ref}",
        headers={"Authorization": f"Bearer {tenant}"},
        timeout=10.0,
    )
    assert fetch.status_code == 200, fetch.text
    records = fetch.json()
    assert len(records) >= 1
    latest = records[0]
    assert latest["tenant_id"] == tenant
    assert latest["decision"] == "reject"
    assert latest["llm_narrative"].strip() != ""

    # And the audit row landed directly in agent.agent_decisions with the
    # fields the graph's node_audit is responsible for writing.
    conn = _pg_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT * FROM agent.agent_decisions
                   WHERE tenant_id = %s AND application_ref = %s
                   ORDER BY created_at DESC LIMIT 1""",
                (tenant, application_ref),
            )
            row = cur.fetchone()
        assert row is not None
        assert row["decision"] == "reject"
        assert row["risk_score"] is not None and row["risk_score"] > 70
        assert row["model_version"] is not None
        assert row["rule_engine_version"] is not None
        assert row["llm_narrative"].strip() != ""
        assert isinstance(row["triggered_thresholds"], list) and len(row["triggered_thresholds"]) >= 1
        assert row["policy_aligned"] == result["policy_aligned"]
        if result["policy_aligned"]:
            assert row["policy_doc_version"] is not None
            assert len(row["retrieved_chunk_ids"]) > 0
        else:
            assert row["policy_doc_version"] is None
            assert row["retrieved_chunk_ids"] == []
    finally:
        conn.close()
