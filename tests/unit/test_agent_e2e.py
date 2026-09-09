"""
tests/unit/test_agent_e2e.py
End-to-end tests for agent/graph.py's full six-node pipeline, run against
the three seeded rule_engine tenants with synthetic predict/retrieve
responses crafted to clearly land in each decision band.

Fully hermetic — no live services, no network:
  - call_predict / call_retrieve / call_llm_synthesis are stubbed with
    crafted stand-ins (predict's own correctness is covered by
    tests/unit/test_predict.py; retrieval's hybrid-search correctness by
    tests/unit/test_rag_retrieve.py — this file isn't re-testing those).
  - call_decide is stubbed to call the REAL rule_engine.engine.decide()
    directly (in-process, no HTTP) — so every decision and threshold value
    asserted on below is the actual rule_engine output for that tenant and
    input, not a hand-picked fake pretending to be one.

informal_digital_lender's own policy has no "refer" outcome at all (fully
automated, no manual-review step — see rule_engine/thresholds.py's
comment). Its refer coverage below comes from the retrieval-fallback path
instead (zero retrieved chunks -> rule_engine's generic FALLBACK_RISK_BANDS,
which does have a refer tier), which is itself a real, separate code path
worth covering — not a workaround.
"""
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from agent.clients import UpstreamServiceError
from agent.graph import GRAPH
from rule_engine.engine import decide as real_decide


def _fake_call_decide(tenant_id, risk_score, shap_factors, loan_type, applicant_profile, use_fallback, token):
    return real_decide(
        tenant_id=tenant_id, risk_score=risk_score, shap_factors=shap_factors,
        loan_type=loan_type, applicant_profile=applicant_profile, use_fallback=use_fallback,
    )


class FakeAuditCursor:
    def __init__(self):
        self.inserted_rows = []

    def execute(self, query, params=None):
        self.inserted_rows.append(params)

    def fetchone(self):
        return None

    def fetchall(self):
        return []


def _fake_get_cursor(cursor):
    @contextmanager
    def _get_cursor(commit=False):
        yield cursor
    return _get_cursor


DEFAULT_CHUNK = {
    "chunk_id": "c1", "doc_id": "d1", "doc_version": "v1",
    "section_ref": "4.2", "chunk_text": "Relevant policy text.", "score": 0.9,
}


def run_assessment(
    tenant_id, risk_score, shap_factors, loan_type, applicant_profile,
    retrieved_chunks=None, narrative="Test narrative.", predict_error=None,
):
    retrieved_chunks = [DEFAULT_CHUNK] if retrieved_chunks is None else retrieved_chunks
    audit_cursor = FakeAuditCursor()
    captured = {}

    def fake_predict(applicant, token):
        if predict_error:
            raise predict_error
        return {
            "application_id": applicant.get("ID"),
            "model_version": "test-model-v1",
            "default_prediction": 1 if risk_score >= 50 else 0,
            "default_probability": risk_score / 100,
            "risk_score": risk_score,
            "shap_factors": shap_factors,
        }

    def fake_llm(prompt):
        captured["prompt"] = prompt
        return narrative

    with patch("agent.graph.call_predict", side_effect=fake_predict), \
         patch("agent.graph.call_retrieve", return_value=retrieved_chunks), \
         patch("agent.graph.call_decide", side_effect=_fake_call_decide), \
         patch("agent.graph.call_llm_synthesis", side_effect=fake_llm), \
         patch("agent.db.get_cursor", _fake_get_cursor(audit_cursor)):
        state = GRAPH.invoke({
            "tenant_id": tenant_id,
            "token": "test-token",
            "applicant": {"ID": 1, "loan_amount": 250000},
            "loan_type": loan_type,
            "applicant_profile": applicant_profile,
        })

    state["_prompt"] = captured.get("prompt", "")
    state["_audit_rows"] = audit_cursor.inserted_rows
    return state


# ── commercial_bank: real policy drives all three bands ────────────────────────

COMMERCIAL_BANK_SCENARIOS = [
    ("approve", 10, {"dti": 0.10}),
    ("refer",   45, {"dti": 0.10}),
    ("reject",  90, {"dti": 0.10}),
]


@pytest.mark.parametrize("expected_decision,risk_score,profile", COMMERCIAL_BANK_SCENARIOS)
def test_commercial_bank_lands_in_expected_band(expected_decision, risk_score, profile):
    shap_factors = [{"feature": "debt_to_income", "value": 0.31}]
    result = run_assessment("commercial_bank", risk_score, shap_factors, "real_estate_payment_plan", profile)
    assert result["decision"] == expected_decision
    assert result["policy_aligned"] is True
    assert result["retrieval_query"] == "debt to income ratio threshold real estate payment plan"


def test_commercial_bank_dti_hard_cap_forces_reject_even_at_low_risk_score():
    result = run_assessment(
        "commercial_bank", 5, [{"feature": "debt_to_income", "value": 0.55}],
        "mortgage", {"dti": 0.55},
    )
    assert result["decision"] == "reject"
    assert any(t["rule"] == "dti_hard_cap" for t in result["triggered_thresholds"])


# ── microfinance_sacco: real policy drives all three bands ──────────────────────

SACCO_SCENARIOS = [
    ("approve", 10, {"dti": 0.10}),
    ("refer",   70, {"dti": 0.10}),
    ("reject",  10, {"dti": 0.50}),   # DTI hard cap, independent of risk score
]


@pytest.mark.parametrize("expected_decision,risk_score,profile", SACCO_SCENARIOS)
def test_sacco_lands_in_expected_band(expected_decision, risk_score, profile):
    shap_factors = [{"feature": "debt_to_income", "value": 0.20}]
    result = run_assessment("microfinance_sacco", risk_score, shap_factors, "group_loan", profile)
    assert result["decision"] == expected_decision
    assert result["policy_aligned"] is True


# ── informal_digital_lender: approve/reject from its real policy; refer via
#    the retrieval-fallback path (see module docstring) ─────────────────────────

INFORMAL_LENDER_SCENARIOS = [
    ("approve", 10),
    ("reject",  90),
]


@pytest.mark.parametrize("expected_decision,risk_score", INFORMAL_LENDER_SCENARIOS)
def test_informal_lender_lands_in_expected_band(expected_decision, risk_score):
    shap_factors = [{"feature": "transaction_frequency", "value": 0.2}]
    result = run_assessment("informal_digital_lender", risk_score, shap_factors, "cash_advance", {})
    assert result["decision"] == expected_decision
    assert result["policy_aligned"] is True


def test_informal_lender_never_reaches_refer_via_its_own_policy():
    for score in [0, 20, 40, 55, 70, 90, 100]:
        result = run_assessment("informal_digital_lender", score, [], "cash_advance", {})
        assert result["decision"] != "refer"


def test_informal_lender_reaches_refer_via_fallback_when_no_policy_ingested():
    result = run_assessment(
        "informal_digital_lender", 55, [{"feature": "transaction_frequency", "value": 0.2}],
        "cash_advance", {}, retrieved_chunks=[],
    )
    assert result["policy_aligned"] is False
    assert result["decision"] == "refer"
    assert any(t["rule"] == "fallback_mid" for t in result["triggered_thresholds"])


# ── Zero-chunk fallback behavior (explicit requirement) ─────────────────────────

class TestRetrievalFallback:

    def test_zero_chunks_sets_policy_aligned_false(self):
        result = run_assessment("commercial_bank", 10, [], "mortgage", {}, retrieved_chunks=[])
        assert result["policy_aligned"] is False

    def test_nonzero_chunks_sets_policy_aligned_true(self):
        result = run_assessment("commercial_bank", 10, [], "mortgage", {}, retrieved_chunks=[DEFAULT_CHUNK])
        assert result["policy_aligned"] is True

    def test_zero_chunks_uses_fallback_bands_not_tenant_policy(self):
        """A DTI that would hard-reject under commercial_bank's real policy
        must have no effect once fallback kicks in — fallback is
        risk-score-only, regardless of tenant."""
        result = run_assessment(
            "commercial_bank", 10, [], "mortgage", {"dti": 0.99}, retrieved_chunks=[],
        )
        assert result["decision"] == "approve"   # fallback_low, DTI ignored
        assert result["triggered_thresholds"] == []

    def test_zero_chunks_still_works_for_a_tenant_with_no_rule_engine_config_at_all(self):
        """The retrieval-fallback path must not require the tenant to be
        onboarded into rule_engine's TENANT_THRESHOLDS either — it's a
        generic, tenant-agnostic decision."""
        result = run_assessment("brand-new-tenant", 55, [], "mortgage", {}, retrieved_chunks=[])
        assert result["policy_aligned"] is False
        assert result["decision"] == "refer"


# ── Audit trail ──────────────────────────────────────────────────────────────

def test_audit_row_is_written_with_the_full_trace():
    result = run_assessment(
        "commercial_bank", 90, [{"feature": "debt_to_income", "value": 0.5}],
        "mortgage", {"dti": 0.5}, narrative="Because of high DTI.",
        retrieved_chunks=[DEFAULT_CHUNK],
    )
    assert len(result["_audit_rows"]) == 1
    params = result["_audit_rows"][0]
    (tenant_id, application_ref, risk_score, model_version, rule_engine_version,
     decision, thresholds_json, chunk_ids, policy_doc_version, policy_aligned,
     narrative) = params

    assert tenant_id == "commercial_bank"
    assert application_ref == "1"   # str(applicant["ID"]) as echoed back by /v1/predict
    assert risk_score == 90
    assert decision == "reject"
    assert json.loads(thresholds_json)
    assert chunk_ids == [DEFAULT_CHUNK["chunk_id"]]
    assert policy_doc_version == DEFAULT_CHUNK["doc_version"]
    assert policy_aligned is True
    assert narrative == "Because of high DTI."


def test_audit_row_reflects_fallback_state():
    result = run_assessment("commercial_bank", 10, [], "mortgage", {}, retrieved_chunks=[])
    params = result["_audit_rows"][0]
    chunk_ids, policy_doc_version, policy_aligned = params[7], params[8], params[9]
    assert chunk_ids == []
    assert policy_doc_version is None
    assert policy_aligned is False


# ── Response field mapping ───────────────────────────────────────────────────

def test_full_state_has_every_documented_response_field():
    result = run_assessment(
        "commercial_bank", 45, [{"feature": "debt_to_income", "value": 0.31}],
        "mortgage", {"dti": 0.10}, narrative="A narrative explanation.",
    )
    assert result["decision"] in {"approve", "reject", "refer"}
    assert result["risk_score"] == 45
    assert result["shap_factors"] == [{"feature": "debt_to_income", "value": 0.31}]
    assert isinstance(result["triggered_thresholds"], list)
    assert result["narrative"] == "A narrative explanation."
    assert result["policy_aligned"] is True


# ── Upstream failure propagation ─────────────────────────────────────────────

def test_predict_failure_propagates_and_never_reaches_audit():
    with pytest.raises(UpstreamServiceError):
        run_assessment(
            "commercial_bank", 10, [], "mortgage", {},
            predict_error=UpstreamServiceError("Selastone /v1/predict returned 500: boom"),
        )
