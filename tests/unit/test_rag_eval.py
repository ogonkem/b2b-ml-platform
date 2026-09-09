"""
tests/unit/test_rag_eval.py
Tests for the post-ingestion RAG eval feature: rag_service/ingest_task.py's
_score_eval_query/_compute_eval_metrics/run_eval/reevaluate_document, and
rag_service/main.py's eval_set upload validation/diffing and the two new
/v1/documents/{doc_id}/eval endpoints. Everything external (Postgres,
Redis, MinIO, Celery, OpenAI via _execute_retrieval) is mocked, matching
this project's existing tests/unit/ convention.

Set env vars and mock external services BEFORE importing rag_service.main,
same requirement tests/unit/test_rag_retrieve.py documents: the module runs
init_schema() and MinIO bucket setup at import time. Auth uses the same
VALID_TOKENS-mutation pattern as that file (not an API_TOKENS env var) for
the same import-order reason documented there.
"""
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

with patch("psycopg2.connect") as _mock_connect, \
     patch("minio.Minio") as _mock_minio_cls:
    _mock_connect.return_value = MagicMock()
    _mock_minio_instance = MagicMock()
    _mock_minio_instance.bucket_exists.return_value = True
    _mock_minio_cls.return_value = _mock_minio_instance

    from fastapi.testclient import TestClient

    from rag_service.ingest_task import _compute_eval_metrics, _score_eval_query
    from rag_service.main import _store_eval_set_if_changed, _validate_eval_set, app

client = TestClient(app)

TEST_TENANT = "token-rag-eval-tenant"
from app.auth import VALID_TOKENS  # noqa: E402
VALID_TOKENS.update({TEST_TENANT})
AUTH = {"Authorization": f"Bearer {TEST_TENANT}"}


# ── _score_eval_query ────────────────────────────────────────────────────────

class TestScoreEvalQuery:

    def test_hits_on_substring_match_returns_1_indexed_rank(self):
        results = [
            {"chunk_text": "irrelevant text", "section_ref": None},
            {"chunk_text": "Loan-to-Value must not exceed 80%.", "section_ref": "4.2"},
        ]
        rank = _score_eval_query(results, "loan-to-value", None)
        assert rank == 2

    def test_substring_match_is_case_insensitive(self):
        results = [{"chunk_text": "DEBT SERVICE COVERAGE RATIO", "section_ref": None}]
        assert _score_eval_query(results, "debt service coverage", None) == 1

    def test_falls_back_to_section_ref_match(self):
        results = [{"chunk_text": "some unrelated body text", "section_ref": "3.1"}]
        assert _score_eval_query(results, "text that never appears", "3.1") == 1

    def test_no_match_returns_none(self):
        results = [{"chunk_text": "nothing relevant here", "section_ref": "9.9"}]
        assert _score_eval_query(results, "collateral valuation", "1.1") is None

    def test_empty_results_returns_none(self):
        assert _score_eval_query([], "anything", None) is None


# ── _compute_eval_metrics ─────────────────────────────────────────────────────

class TestComputeEvalMetrics:

    def test_all_hits_at_rank_1_gives_perfect_scores(self):
        per_query = [{"query": "q1", "hit": True, "rank": 1}, {"query": "q2", "hit": True, "rank": 1}]
        metrics = _compute_eval_metrics(per_query)
        assert metrics["recall_at_k"] == 1.0
        assert metrics["mrr"] == 1.0
        assert metrics["total_queries"] == 2

    def test_mixed_hits_and_misses(self):
        per_query = [
            {"query": "q1", "hit": True, "rank": 1},
            {"query": "q2", "hit": True, "rank": 4},
            {"query": "q3", "hit": False, "rank": None},
        ]
        metrics = _compute_eval_metrics(per_query)
        assert metrics["recall_at_k"] == pytest.approx(2 / 3)
        assert metrics["mrr"] == pytest.approx((1 / 1 + 1 / 4) / 3)

    def test_all_misses_gives_zero_scores(self):
        per_query = [{"query": "q1", "hit": False, "rank": None}]
        metrics = _compute_eval_metrics(per_query)
        assert metrics["recall_at_k"] == 0.0
        assert metrics["mrr"] == 0.0

    def test_no_queries_does_not_divide_by_zero(self):
        metrics = _compute_eval_metrics([])
        assert metrics["recall_at_k"] == 0.0
        assert metrics["mrr"] == 0.0
        assert metrics["total_queries"] == 0

    def test_per_query_detail_is_preserved_in_output(self):
        per_query = [{"query": "q1", "hit": True, "rank": 2}]
        metrics = _compute_eval_metrics(per_query)
        assert metrics["per_query"] == per_query


# ── _validate_eval_set ────────────────────────────────────────────────────────

class TestValidateEvalSet:

    def test_accepts_well_formed_array(self):
        raw = json.dumps([{"query": "q", "expected_text_substring": "x"}])
        assert _validate_eval_set(raw) == [{"query": "q", "expected_text_substring": "x"}]

    def test_accepts_optional_section_ref(self):
        raw = json.dumps([{"query": "q", "expected_text_substring": "x", "expected_section_ref": "1.1"}])
        parsed = _validate_eval_set(raw)
        assert parsed[0]["expected_section_ref"] == "1.1"

    def test_rejects_invalid_json(self):
        with pytest.raises(Exception) as exc_info:
            _validate_eval_set("{not valid json")
        assert exc_info.value.status_code == 400

    def test_rejects_non_list(self):
        with pytest.raises(Exception) as exc_info:
            _validate_eval_set(json.dumps({"query": "q"}))
        assert exc_info.value.status_code == 400

    def test_rejects_empty_list(self):
        with pytest.raises(Exception) as exc_info:
            _validate_eval_set("[]")
        assert exc_info.value.status_code == 400

    def test_rejects_entry_missing_query(self):
        with pytest.raises(Exception) as exc_info:
            _validate_eval_set(json.dumps([{"expected_text_substring": "x"}]))
        assert exc_info.value.status_code == 400

    def test_rejects_entry_missing_expected_text_substring(self):
        with pytest.raises(Exception) as exc_info:
            _validate_eval_set(json.dumps([{"query": "q"}]))
        assert exc_info.value.status_code == 400


# ── _store_eval_set_if_changed — hash diffing, versioned like doc_chunks ─────

class FakeEvalSetCursor:
    """Enough of rag.eval_sets to exercise the hash-diff/version-bump logic
    without a real Postgres connection."""

    def __init__(self):
        self.rows = []  # list of dicts, insertion order == upload order

    def execute(self, query, params=None):
        q = " ".join(query.split()).upper()
        params = params or ()
        if q.startswith("SELECT EVAL_HASH, VERSION FROM RAG.EVAL_SETS"):
            doc_id = params[0]
            matches = [r for r in self.rows if r["doc_id"] == doc_id]
            self._result = [matches[-1]] if matches else []
        elif q.startswith("INSERT INTO RAG.EVAL_SETS"):
            tenant_id, doc_id, version, eval_hash, queries_json = params
            self.rows.append({
                "tenant_id": tenant_id, "doc_id": doc_id, "version": version,
                "eval_hash": eval_hash, "queries": json.loads(queries_json),
            })
        else:
            raise AssertionError(f"Unexpected query in FakeEvalSetCursor: {query!r}")

    def fetchone(self):
        return self._result[0] if getattr(self, "_result", None) else None

    def fetchall(self):
        return getattr(self, "_result", [])


@contextmanager
def _fake_get_cursor(cursor):
    yield cursor


class TestStoreEvalSetIfChanged:

    def test_first_upload_creates_v1(self):
        cur = FakeEvalSetCursor()
        with patch("rag_service.main.get_cursor", lambda commit=False: _fake_get_cursor(cur)):
            _store_eval_set_if_changed("tenant-a", "doc-1", [{"query": "q", "expected_text_substring": "x"}])
        assert len(cur.rows) == 1
        assert cur.rows[0]["version"] == "v1"

    def test_identical_reupload_does_not_create_new_version(self):
        cur = FakeEvalSetCursor()
        queries = [{"query": "q", "expected_text_substring": "x"}]
        with patch("rag_service.main.get_cursor", lambda commit=False: _fake_get_cursor(cur)):
            _store_eval_set_if_changed("tenant-a", "doc-1", queries)
            _store_eval_set_if_changed("tenant-a", "doc-1", queries)
        assert len(cur.rows) == 1

    def test_changed_content_bumps_to_v2(self):
        cur = FakeEvalSetCursor()
        with patch("rag_service.main.get_cursor", lambda commit=False: _fake_get_cursor(cur)):
            _store_eval_set_if_changed("tenant-a", "doc-1", [{"query": "q1", "expected_text_substring": "x"}])
            _store_eval_set_if_changed("tenant-a", "doc-1", [{"query": "q2", "expected_text_substring": "y"}])
        assert len(cur.rows) == 2
        assert cur.rows[1]["version"] == "v2"

    def test_key_order_does_not_affect_hash(self):
        """json.dumps(..., sort_keys=True) means {"a":1,"b":2} and
        {"b":2,"a":1} must hash identically — otherwise re-uploading the
        exact same eval_set with keys in a different order would wrongly
        bump the version."""
        cur = FakeEvalSetCursor()
        with patch("rag_service.main.get_cursor", lambda commit=False: _fake_get_cursor(cur)):
            _store_eval_set_if_changed("tenant-a", "doc-1", [{"query": "q", "expected_text_substring": "x"}])
            _store_eval_set_if_changed("tenant-a", "doc-1", [{"expected_text_substring": "x", "query": "q"}])
        assert len(cur.rows) == 1


# ── rag_service/db.py: tenant pipeline config ────────────────────────────────

class TestTenantPipelineConfig:

    def test_returns_defaults_when_no_row_exists(self):
        from rag_service.db import get_tenant_pipeline_config
        fake_cur = MagicMock()
        fake_cur.fetchone.return_value = None
        with patch("rag_service.db.get_cursor", lambda commit=False: _fake_get_cursor(fake_cur)):
            config = get_tenant_pipeline_config("tenant-a", {"rrf_k": 60, "candidate_limit": 20})
        assert config == {"rrf_k": 60, "candidate_limit": 20}

    def test_overrides_only_non_null_fields(self):
        from rag_service.db import get_tenant_pipeline_config
        fake_cur = MagicMock()
        fake_cur.fetchone.return_value = {"rrf_k": 30, "candidate_limit": None}
        with patch("rag_service.db.get_cursor", lambda commit=False: _fake_get_cursor(fake_cur)):
            config = get_tenant_pipeline_config("tenant-a", {"rrf_k": 60, "candidate_limit": 20})
        assert config == {"rrf_k": 30, "candidate_limit": 20}

    def test_degrades_to_defaults_on_any_exception(self):
        """A tuning-knob lookup failing (missing table, DB hiccup, whatever)
        must never break retrieval — see the function's own docstring."""
        from rag_service.db import get_tenant_pipeline_config

        def _raises(commit=False):
            raise RuntimeError("connection refused")

        with patch("rag_service.db.get_cursor", _raises):
            config = get_tenant_pipeline_config("tenant-a", {"rrf_k": 60, "candidate_limit": 20})
        assert config == {"rrf_k": 60, "candidate_limit": 20}


# ── POST/GET /v1/documents/{doc_id}/eval ─────────────────────────────────────

DOC_ID = "11111111-1111-1111-1111-111111111111"


class TestReevaluateEndpoint:

    def test_404_when_document_not_found(self):
        with patch("rag_service.main.get_cursor") as mock_cur:
            mock_cur.return_value.__enter__.return_value.fetchone.return_value = None
            resp = client.post(f"/v1/documents/{DOC_ID}/eval", json={"rrf_k": 40}, headers=AUTH)
        assert resp.status_code == 404

    def test_404_on_malformed_doc_id(self):
        resp = client.post("/v1/documents/not-a-uuid/eval", json={"rrf_k": 40}, headers=AUTH)
        assert resp.status_code == 404

    def test_query_time_only_lever_does_not_trigger_rechunk(self):
        with patch("rag_service.main.get_cursor") as mock_cur, \
             patch("rag_service.main.upsert_tenant_pipeline_config") as mock_upsert, \
             patch("rag_service.main._celery_app") as mock_celery:
            mock_cur.return_value.__enter__.return_value.fetchone.return_value = {"id": DOC_ID}
            resp = client.post(f"/v1/documents/{DOC_ID}/eval", json={"rrf_k": 40, "candidate_limit": 15}, headers=AUTH)

        assert resp.status_code == 202
        assert resp.json()["rechunk"] is False
        mock_upsert.assert_called_once()
        assert mock_upsert.call_args.kwargs["rrf_k"] == 40
        assert mock_upsert.call_args.kwargs["candidate_limit"] == 15
        mock_celery.send_task.assert_called_once_with(
            "reevaluate_document",
            kwargs={"tenant_id": TEST_TENANT, "doc_id": DOC_ID, "rechunk": False},
        )

    def test_chunking_lever_triggers_rechunk(self):
        with patch("rag_service.main.get_cursor") as mock_cur, \
             patch("rag_service.main.upsert_tenant_pipeline_config"), \
             patch("rag_service.main._celery_app") as mock_celery:
            mock_cur.return_value.__enter__.return_value.fetchone.return_value = {"id": DOC_ID}
            resp = client.post(f"/v1/documents/{DOC_ID}/eval", json={"max_chunk_tokens": 300}, headers=AUTH)

        assert resp.status_code == 202
        assert resp.json()["rechunk"] is True
        mock_celery.send_task.assert_called_once_with(
            "reevaluate_document",
            kwargs={"tenant_id": TEST_TENANT, "doc_id": DOC_ID, "rechunk": True},
        )

    def test_empty_body_still_enqueues_without_touching_config(self):
        with patch("rag_service.main.get_cursor") as mock_cur, \
             patch("rag_service.main.upsert_tenant_pipeline_config") as mock_upsert, \
             patch("rag_service.main._celery_app"):
            mock_cur.return_value.__enter__.return_value.fetchone.return_value = {"id": DOC_ID}
            resp = client.post(f"/v1/documents/{DOC_ID}/eval", json={}, headers=AUTH)

        assert resp.status_code == 202
        mock_upsert.assert_not_called()


class TestEvalReportEndpoint:

    def test_404_when_document_not_found(self):
        with patch("rag_service.main.get_cursor") as mock_cur:
            mock_cur.return_value.__enter__.return_value.fetchone.return_value = None
            resp = client.get(f"/v1/documents/{DOC_ID}/eval", headers=AUTH)
        assert resp.status_code == 404

    def test_returns_latest_run_and_status(self):
        latest_run = {
            "id": "run-1", "doc_version": "v1", "eval_set_version": "v1", "status": "complete",
            "metrics": {"recall_at_k": 0.8, "mrr": 0.75, "total_queries": 5},
            "lever_snapshot": {"rrf_k": 60}, "error": None, "created_at": "2026-01-01T00:00:00Z",
        }
        fetchone_results = iter([{"id": DOC_ID}, latest_run])
        with patch("rag_service.main.get_cursor") as mock_cur, \
             patch("rag_service.main.get_eval_status", return_value={"status": "complete", "error": None}):
            mock_cur.return_value.__enter__.return_value.fetchone.side_effect = lambda: next(fetchone_results)
            resp = client.get(f"/v1/documents/{DOC_ID}/eval", headers=AUTH)

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "complete"
        assert body["latest_run"]["metrics"]["recall_at_k"] == 0.8

    def test_no_run_yet_returns_null_latest_run(self):
        fetchone_results = iter([{"id": DOC_ID}, None])
        with patch("rag_service.main.get_cursor") as mock_cur, \
             patch("rag_service.main.get_eval_status", return_value={"status": "unknown", "error": None}):
            mock_cur.return_value.__enter__.return_value.fetchone.side_effect = lambda: next(fetchone_results)
            resp = client.get(f"/v1/documents/{DOC_ID}/eval", headers=AUTH)

        assert resp.status_code == 200
        assert resp.json()["latest_run"] is None


# ── run_eval — end to end with retrieval mocked ──────────────────────────────

class TestRunEval:

    def test_no_eval_set_sets_status_and_writes_no_run_row(self):
        from rag_service.ingest_task import run_eval

        fake_cur = MagicMock()
        fake_cur.fetchone.return_value = None  # no eval_sets row
        with patch("rag_service.ingest_task.get_cursor", lambda commit=False: _fake_get_cursor(fake_cur)), \
             patch("rag_service.ingest_task.set_eval_status") as mock_set_status:
            result = run_eval("tenant-a", "doc-1")

        assert result["status"] == "no_eval_set"
        mock_set_status.assert_called_once_with("doc-1", "no_eval_set")

    def test_complete_run_scores_hits_and_writes_metrics(self):
        from rag_service.ingest_task import run_eval

        eval_set_row = {
            "version": "v1",
            "queries": [
                {"query": "collateral requirements", "expected_text_substring": "collateral"},
                {"query": "unrelated question", "expected_text_substring": "will never match"},
            ],
        }
        doc_row = {"current_version": "v2"}

        call_sequence = []

        class Cur:
            def execute(self, query, params=None):
                q = " ".join(query.split()).upper()
                call_sequence.append(q)
                if q.startswith("SELECT VERSION, QUERIES FROM RAG.EVAL_SETS"):
                    self._next = eval_set_row
                elif q.startswith("SELECT CURRENT_VERSION FROM RAG.DOCUMENTS"):
                    self._next = doc_row
                elif q.startswith("SELECT * FROM RAG.TENANT_PIPELINE_CONFIG"):
                    self._next = None
                elif q.startswith("INSERT INTO RAG.EVAL_RUNS"):
                    self._next = None
                    self.inserted_params = params
                else:
                    raise AssertionError(f"Unexpected query: {query!r}")

            def fetchone(self):
                return self._next

        cur = Cur()

        def fake_execute_retrieval(tenant_id, query, top_k):
            if "collateral" in query:
                return [{"chunk_text": "Section on collateral valuation standards.", "section_ref": "2.1"}]
            return [{"chunk_text": "totally different content", "section_ref": "9.9"}]

        with patch("rag_service.ingest_task.get_cursor", lambda commit=False: _fake_get_cursor(cur)), \
             patch("rag_service.ingest_task.set_eval_status") as mock_set_status, \
             patch("rag_service.main._execute_retrieval", side_effect=fake_execute_retrieval):
            result = run_eval("tenant-a", "doc-1")

        assert result["status"] == "complete"
        assert result["metrics"]["total_queries"] == 2
        assert result["metrics"]["recall_at_k"] == pytest.approx(0.5)
        mock_set_status.assert_called_once_with("doc-1", "complete")
        # tenant_id, doc_id, doc_version, eval_set_version, metrics_json, lever_json
        assert cur.inserted_params[0] == "tenant-a"
        assert cur.inserted_params[2] == "v2"
        assert cur.inserted_params[3] == "v1"

    def test_retrieval_failure_writes_failed_run_and_does_not_raise(self):
        from rag_service.ingest_task import run_eval

        eval_set_row = {"version": "v1", "queries": [{"query": "q", "expected_text_substring": "x"}]}
        doc_row = {"current_version": "v1"}

        class Cur:
            def __init__(self):
                self.failed_insert_params = None

            def execute(self, query, params=None):
                q = " ".join(query.split()).upper()
                if q.startswith("SELECT VERSION, QUERIES FROM RAG.EVAL_SETS"):
                    self._next = eval_set_row
                elif q.startswith("SELECT CURRENT_VERSION FROM RAG.DOCUMENTS"):
                    self._next = doc_row
                elif q.startswith("SELECT * FROM RAG.TENANT_PIPELINE_CONFIG"):
                    self._next = None
                elif q.startswith("INSERT INTO RAG.EVAL_RUNS"):
                    self.failed_insert_params = params
                    self._next = None
                else:
                    raise AssertionError(f"Unexpected query: {query!r}")

            def fetchone(self):
                return self._next

        cur = Cur()

        with patch("rag_service.ingest_task.get_cursor", lambda commit=False: _fake_get_cursor(cur)), \
             patch("rag_service.ingest_task.set_eval_status") as mock_set_status, \
             patch("rag_service.main._execute_retrieval", side_effect=RuntimeError("no OPENAI_API_KEY")):
            result = run_eval("tenant-a", "doc-1")

        assert result["status"] == "failed"
        assert "OPENAI_API_KEY" in result["error"]
        mock_set_status.assert_called_once()
        assert mock_set_status.call_args.args[1] == "failed"
        assert cur.failed_insert_params is not None


# ── reevaluate_document — rechunk decision ───────────────────────────────────

class TestReevaluateDocumentTask:

    def test_rechunk_false_skips_ingestion_and_runs_eval_only(self):
        from rag_service.ingest_task import reevaluate_document

        with patch("rag_service.ingest_task._redis", return_value=MagicMock()), \
             patch("rag_service.ingest_task.run_eval", return_value={"status": "complete"}) as mock_run_eval, \
             patch("rag_service.ingest_task._run_ingestion") as mock_run_ingestion, \
             patch("rag_service.ingest_task._minio") as mock_minio:
            reevaluate_document.run("tenant-a", "doc-1", rechunk=False)

        mock_run_ingestion.assert_not_called()
        mock_minio.return_value.get_object.assert_not_called()
        mock_run_eval.assert_called_once_with("tenant-a", "doc-1")

    def test_rechunk_true_fetches_original_from_minio_and_reingests(self):
        from rag_service.ingest_task import reevaluate_document

        fake_cur = MagicMock()
        fake_cur.fetchone.return_value = {"filename": "policy.md", "current_version": "v1"}
        fake_response = MagicMock()
        fake_response.read.return_value = b"original file bytes"
        fake_minio = MagicMock()
        fake_minio.get_object.return_value = fake_response

        with patch("rag_service.ingest_task.get_cursor", lambda commit=False: _fake_get_cursor(fake_cur)), \
             patch("rag_service.ingest_task._redis", return_value=MagicMock()), \
             patch("rag_service.ingest_task._minio", return_value=fake_minio), \
             patch("rag_service.ingest_task._run_ingestion") as mock_run_ingestion, \
             patch("rag_service.ingest_task.run_eval", return_value={"status": "complete"}) as mock_run_eval:
            reevaluate_document.run("tenant-a", "doc-1", rechunk=True)

        fake_minio.get_object.assert_called_once_with("doc-chunks-raw", "tenant-a/doc-1/v1/original.md")
        mock_run_ingestion.assert_called_once_with("tenant-a", "doc-1", "policy.md", b"original file bytes", ingested_by="tenant-a")
        mock_run_eval.assert_called_once_with("tenant-a", "doc-1")

    def test_document_not_found_sets_failed_status_and_raises(self):
        from rag_service.ingest_task import reevaluate_document

        fake_cur = MagicMock()
        fake_cur.fetchone.return_value = None

        with patch("rag_service.ingest_task.get_cursor", lambda commit=False: _fake_get_cursor(fake_cur)), \
             patch("rag_service.ingest_task._redis", return_value=MagicMock()), \
             patch("rag_service.ingest_task.set_eval_status") as mock_set_status:
            with pytest.raises(ValueError):
                reevaluate_document.run("tenant-a", "doc-1", rechunk=True)

        assert mock_set_status.call_args.args[1] == "failed"
