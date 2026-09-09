"""
tests/unit/test_rule_engine_api.py
Wiring tests for rule_engine/main.py's HTTP layer — auth, tenant-mismatch
rejection, and error-code mapping. The actual decision logic is covered
exhaustively in tests/unit/test_rule_engine.py; this file only confirms the
FastAPI wrapper calls it correctly and translates its errors properly. No
external services to mock — rule_engine has none — beyond auth, which is
exercised for real via app.auth.verify_token (same approach as
tests/unit/test_rag_retrieve.py for rag_harness).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi.testclient import TestClient

from rule_engine.main import app

client = TestClient(app)

TENANT = "token-rule-engine-tenant"

from app.auth import VALID_TOKENS
VALID_TOKENS.update({TENANT, "commercial_bank", "ghost-tenant"})


def _payload(**overrides):
    base = {
        "tenant_id": TENANT,
        "risk_score": 10,
        "loan_type": "mortgage",
        "applicant_profile": {},
    }
    base.update(overrides)
    return base


class TestAuth:

    def test_requires_auth(self):
        resp = client.post("/v1/decide", json=_payload())
        assert resp.status_code == 401

    def test_rejects_invalid_token(self):
        resp = client.post("/v1/decide", json=_payload(), headers={"Authorization": "Bearer not-a-real-token"})
        assert resp.status_code == 403

    def test_body_tenant_id_mismatched_with_token_is_rejected(self):
        resp = client.post(
            "/v1/decide",
            json=_payload(tenant_id="some-other-tenant"),
            headers={"Authorization": f"Bearer {TENANT}"},
        )
        assert resp.status_code == 403


class TestDecideEndpoint:

    def test_happy_path_returns_full_shape(self):
        resp = client.post(
            "/v1/decide",
            json=_payload(tenant_id="commercial_bank"),
            headers={"Authorization": "Bearer commercial_bank"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] == "approve"
        assert body["triggered_thresholds"] == []
        assert "rule_engine_version" in body

    def test_unknown_tenant_returns_400_not_500(self):
        resp = client.post(
            "/v1/decide",
            json=_payload(tenant_id="ghost-tenant"),
            headers={"Authorization": "Bearer ghost-tenant"},
        )
        assert resp.status_code == 400
        assert "ghost-tenant" in resp.json()["detail"]

    def test_triggered_thresholds_reach_the_caller_unmodified(self):
        resp = client.post(
            "/v1/decide",
            json=_payload(
                tenant_id="commercial_bank",
                risk_score=80,
                loan_type="real_estate_payment_plan",
                applicant_profile={"dti": 0.50},
            ),
            headers={"Authorization": "Bearer commercial_bank"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] == "reject"
        rules = {t["rule"] for t in body["triggered_thresholds"]}
        assert rules == {"dti_hard_cap", "risk_band_high"}
        for t in body["triggered_thresholds"]:
            assert t["loan_type"] == "real_estate_payment_plan"

    def test_missing_required_fields_returns_422(self):
        resp = client.post(
            "/v1/decide",
            json={"tenant_id": TENANT},   # missing risk_score, loan_type
            headers={"Authorization": f"Bearer {TENANT}"},
        )
        assert resp.status_code == 422

    def test_shap_factors_and_applicant_profile_are_optional(self):
        resp = client.post(
            "/v1/decide",
            json={"tenant_id": "commercial_bank", "risk_score": 10, "loan_type": "x"},
            headers={"Authorization": "Bearer commercial_bank"},
        )
        assert resp.status_code == 200
