"""
tests/unit/test_agent_api.py
Wiring tests for agent/main.py's HTTP layer — auth, tenant-mismatch
rejection, and upstream-error-to-502 mapping. The graph's actual behavior
is covered in tests/unit/test_agent_e2e.py; this file only confirms the
FastAPI wrapper invokes it correctly. psycopg2.connect is mocked before
import since agent.main runs init_schema() at import time (same
requirement CLAUDE.md documents for app.main).
"""
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

with patch("psycopg2.connect") as _mock_connect:
    _mock_connect.return_value = MagicMock()
    from fastapi.testclient import TestClient
    from agent.main import app

from agent.clients import UpstreamServiceError

client = TestClient(app)

TENANT = "token-agent-tenant"

from app.auth import VALID_TOKENS
VALID_TOKENS.add(TENANT)


def _payload(**overrides):
    base = {
        "tenant_id": TENANT,
        "loan_type": "mortgage",
        "applicant": {"ID": 1},
        "applicant_profile": {},
    }
    base.update(overrides)
    return base


GRAPH_RESULT = {
    "decision": "approve",
    "risk_score": 10.0,
    "shap_factors": [{"feature": "income", "value": 0.1}],
    "triggered_thresholds": [],
    "retrieved_chunks": [
        {"chunk_id": "c1", "doc_id": "d1", "doc_version": "v1",
         "section_ref": "4.2", "chunk_text": "Policy text.", "score": 0.9},
    ],
    "narrative": "All clear.",
    "policy_aligned": True,
}


class TestAuth:

    def test_requires_auth(self):
        resp = client.post("/v1/agent/assess", json=_payload())
        assert resp.status_code == 401

    def test_body_tenant_id_mismatched_with_token_is_rejected(self):
        resp = client.post(
            "/v1/agent/assess",
            json=_payload(tenant_id="some-other-tenant"),
            headers={"Authorization": f"Bearer {TENANT}"},
        )
        assert resp.status_code == 403


class TestAssessEndpoint:

    def test_happy_path_maps_graph_result_to_response_shape(self):
        with patch("agent.main.GRAPH") as fake_graph:
            fake_graph.invoke.return_value = GRAPH_RESULT
            resp = client.post(
                "/v1/agent/assess", json=_payload(), headers={"Authorization": f"Bearer {TENANT}"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] == "approve"
        assert body["risk_score"] == 10.0
        assert body["statistical_factors"] == [{"feature": "income", "value": 0.1}]
        assert body["policy_basis"] == []
        assert body["policy_chunks"][0]["section_ref"] == "4.2"
        assert body["policy_chunks"][0]["chunk_text"] == "Policy text."
        assert body["narrative"] == "All clear."
        assert body["policy_aligned"] is True

    def test_policy_chunks_empty_when_fallback_used(self):
        result = {**GRAPH_RESULT, "retrieved_chunks": [], "policy_aligned": False}
        with patch("agent.main.GRAPH") as fake_graph:
            fake_graph.invoke.return_value = result
            resp = client.post(
                "/v1/agent/assess", json=_payload(), headers={"Authorization": f"Bearer {TENANT}"},
            )
        assert resp.json()["policy_chunks"] == []
        assert resp.json()["policy_aligned"] is False

    def test_upstream_failure_returns_502_not_500(self):
        with patch("agent.main.GRAPH") as fake_graph:
            fake_graph.invoke.side_effect = UpstreamServiceError("rule_engine /v1/decide returned 500: boom")
            resp = client.post(
                "/v1/agent/assess", json=_payload(), headers={"Authorization": f"Bearer {TENANT}"},
            )
        assert resp.status_code == 502
        assert "rule_engine" in resp.json()["detail"]

    def test_forwards_the_callers_bearer_token_to_the_graph(self):
        with patch("agent.main.GRAPH") as fake_graph:
            fake_graph.invoke.return_value = GRAPH_RESULT
            client.post("/v1/agent/assess", json=_payload(), headers={"Authorization": f"Bearer {TENANT}"})
        initial_state = fake_graph.invoke.call_args[0][0]
        assert initial_state["token"] == TENANT
        assert initial_state["tenant_id"] == TENANT

    def test_missing_required_fields_returns_422(self):
        resp = client.post(
            "/v1/agent/assess",
            json={"tenant_id": TENANT},   # missing loan_type, applicant
            headers={"Authorization": f"Bearer {TENANT}"},
        )
        assert resp.status_code == 422

    def test_applicant_profile_is_optional(self):
        with patch("agent.main.GRAPH") as fake_graph:
            fake_graph.invoke.return_value = GRAPH_RESULT
            resp = client.post(
                "/v1/agent/assess",
                json={"tenant_id": TENANT, "loan_type": "mortgage", "applicant": {"ID": 1}},
                headers={"Authorization": f"Bearer {TENANT}"},
            )
        assert resp.status_code == 200


# ── GET /v1/agent/decisions* ─────────────────────────────────────────────────

class FakeDecisionsCursor:
    """Enough of the query surface for get_decision_by_application_ref and
    list_decisions — matches on the distinguishing WHERE clause, not a full
    SQL parser."""

    def __init__(self, rows):
        self.rows = rows
        self._result = []

    def execute(self, query, params=None):
        params = params or ()
        q = " ".join(query.split()).upper()
        if "APPLICATION_REF = %S" in q:
            tenant_id, application_ref = params
            matched = [r for r in self.rows if r["tenant_id"] == tenant_id and r["application_ref"] == application_ref]
        else:
            tenant_id = params[0]
            matched = [r for r in self.rows if r["tenant_id"] == tenant_id]
            if len(params) > 1:
                policy_doc_version = params[1]
                matched = [r for r in matched if r["policy_doc_version"] == policy_doc_version]
        self._result = sorted(matched, key=lambda r: r["created_at"], reverse=True)

    def fetchall(self):
        return self._result

    def fetchone(self):
        return self._result[0] if self._result else None


def _fake_get_cursor(cursor):
    @contextmanager
    def _get_cursor(commit=False):
        yield cursor
    return _get_cursor


def _row(tenant_id, application_ref, policy_doc_version="v1", created_at=None, **overrides):
    base = dict(
        id=str(uuid.uuid4()), tenant_id=tenant_id, application_ref=application_ref,
        risk_score=50.0, model_version="m1", rule_engine_version="1.0.0",
        decision="approve", triggered_thresholds=[], retrieved_chunk_ids=[],
        policy_doc_version=policy_doc_version, policy_aligned=True,
        llm_narrative="Explanation.", created_at=created_at or datetime.now(timezone.utc),
    )
    base.update(overrides)
    return base


class TestGetDecisionByApplicationRef:

    def test_requires_auth(self):
        resp = client.get("/v1/agent/decisions/app-1")
        assert resp.status_code == 401

    def test_returns_matching_rows_for_this_tenant_only(self):
        rows = [_row(TENANT, "app-1"), _row("other-tenant", "app-1")]
        with patch("agent.main.get_cursor", _fake_get_cursor(FakeDecisionsCursor(rows))):
            resp = client.get("/v1/agent/decisions/app-1", headers={"Authorization": f"Bearer {TENANT}"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["tenant_id"] == TENANT

    def test_404_when_nothing_found(self):
        with patch("agent.main.get_cursor", _fake_get_cursor(FakeDecisionsCursor([]))):
            resp = client.get("/v1/agent/decisions/does-not-exist", headers={"Authorization": f"Bearer {TENANT}"})
        assert resp.status_code == 404

    def test_multiple_reassessments_all_returned_newest_first(self):
        older = _row(TENANT, "app-1", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), decision="refer")
        newer = _row(TENANT, "app-1", created_at=datetime(2026, 2, 1, tzinfo=timezone.utc), decision="approve")
        with patch("agent.main.get_cursor", _fake_get_cursor(FakeDecisionsCursor([older, newer]))):
            resp = client.get("/v1/agent/decisions/app-1", headers={"Authorization": f"Bearer {TENANT}"})
        body = resp.json()
        assert len(body) == 2
        assert body[0]["decision"] == "approve"   # newest first
        assert body[1]["decision"] == "refer"

    def test_full_trace_fields_present(self):
        rows = [_row(
            TENANT, "app-1", risk_score=77.5, policy_doc_version="v3",
            retrieved_chunk_ids=[str(uuid.uuid4())], triggered_thresholds=[{"rule": "x"}],
        )]
        with patch("agent.main.get_cursor", _fake_get_cursor(FakeDecisionsCursor(rows))):
            resp = client.get("/v1/agent/decisions/app-1", headers={"Authorization": f"Bearer {TENANT}"})
        record = resp.json()[0]
        assert record["risk_score"] == 77.5
        assert record["policy_doc_version"] == "v3"
        assert len(record["retrieved_chunk_ids"]) == 1
        assert record["triggered_thresholds"] == [{"rule": "x"}]


class TestListDecisions:

    def test_requires_auth(self):
        resp = client.get("/v1/agent/decisions")
        assert resp.status_code == 401

    def test_lists_only_this_tenants_decisions(self):
        rows = [_row(TENANT, "app-1"), _row("other-tenant", "app-2")]
        with patch("agent.main.get_cursor", _fake_get_cursor(FakeDecisionsCursor(rows))):
            resp = client.get("/v1/agent/decisions", headers={"Authorization": f"Bearer {TENANT}"})
        body = resp.json()
        assert len(body) == 1
        assert body[0]["application_ref"] == "app-1"

    def test_filters_by_policy_doc_version(self):
        """The "show every decision made under policy version Y" query."""
        rows = [
            _row(TENANT, "app-1", policy_doc_version="v1"),
            _row(TENANT, "app-2", policy_doc_version="v2"),
        ]
        with patch("agent.main.get_cursor", _fake_get_cursor(FakeDecisionsCursor(rows))):
            resp = client.get(
                "/v1/agent/decisions?policy_doc_version=v2", headers={"Authorization": f"Bearer {TENANT}"},
            )
        body = resp.json()
        assert len(body) == 1
        assert body[0]["application_ref"] == "app-2"

    def test_matching_tenant_id_query_param_is_allowed(self):
        rows = [_row(TENANT, "app-1")]
        with patch("agent.main.get_cursor", _fake_get_cursor(FakeDecisionsCursor(rows))):
            resp = client.get(
                f"/v1/agent/decisions?tenant_id={TENANT}", headers={"Authorization": f"Bearer {TENANT}"},
            )
        assert resp.status_code == 200

    def test_mismatched_tenant_id_query_param_is_rejected(self):
        with patch("agent.main.get_cursor", _fake_get_cursor(FakeDecisionsCursor([]))):
            resp = client.get(
                "/v1/agent/decisions?tenant_id=some-other-tenant", headers={"Authorization": f"Bearer {TENANT}"},
            )
        assert resp.status_code == 403

    def test_empty_list_when_no_decisions_yet(self):
        with patch("agent.main.get_cursor", _fake_get_cursor(FakeDecisionsCursor([]))):
            resp = client.get("/v1/agent/decisions", headers={"Authorization": f"Bearer {TENANT}"})
        assert resp.status_code == 200
        assert resp.json() == []
