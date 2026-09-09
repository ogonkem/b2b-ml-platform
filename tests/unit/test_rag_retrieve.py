"""
tests/unit/test_rag_retrieve.py
Tests for rag_service/main.py's retrieval and document-management endpoints
(POST /v1/retrieve, POST/GET/DELETE /v1/documents).

Set env vars and mock external services BEFORE importing rag_service.main,
same requirement CLAUDE.md documents for app.main: the module runs
init_schema() and MinIO bucket setup at import time.

Auth is exercised for real (not mocked) via app.auth.verify_token — two
static tokens stand in for two different tenants, exactly like
tests/unit/test_auth.py does for app.main. VALID_TOKENS is a module-level
set computed once at app.auth's *first* import anywhere in the test
session — whichever test file happens to import it first (collection order,
not this file's) wins, so setting API_TOKENS via os.environ here would be
silently ignored if some other test file imported app.auth first. Mutating
the already-imported set directly sidesteps that ordering dependency.
"""
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
    from rag_service.main import app

client = TestClient(app)

TENANT_A = "token-tenant-a"
TENANT_B = "token-tenant-b"

# eval_set is a required upload field (rag_service/main.py) — a placeholder
# value here for every test that isn't specifically exercising eval_set
# validation itself (see TestUploadDocument's dedicated eval_set tests).
DEFAULT_EVAL_SET = '[{"query": "x", "expected_text_substring": "x"}]'

from app.auth import VALID_TOKENS
VALID_TOKENS.update({TENANT_A, TENANT_B})


@pytest.fixture(autouse=True)
def _generous_quota():
    """Default for every test in this file: an effectively-unlimited RAG
    plan quota and a Redis stand-in that always reports zero prior usage —
    so tests that predate quota enforcement, or don't care about it, don't
    need to know it exists. Tests that specifically exercise quota
    enforcement (TestIngestionQuota, TestRetrievalQuota, TestUsageEndpoint)
    override these within their own `with patch(...)` block."""
    fake_redis = MagicMock()
    fake_redis.get.return_value = None
    fake_redis.incrby.side_effect = lambda key, n: n
    with patch("rag_service.main._get_rag_plan_quotas", return_value=(1_000_000, 1_000_000)), \
         patch("rag_service.main._redis", return_value=fake_redis), \
         patch("rag_service.ingest_task._redis", return_value=MagicMock()):
        # ingest_task._redis is a separate lazy singleton from main._redis
        # (set_ingest_status, called by upload_document, uses its own) —
        # patched here too so pre-existing upload tests that don't care
        # about ingestion status don't attempt a real Redis connection.
        yield

# rag.documents.id / rag.doc_chunks.doc_id are UUID columns — doc_id values
# used in tests that exercise real doc_id validation must be well-formed.
DOC_A_ID      = "11111111-1111-1111-1111-111111111111"
DOC_B_ID      = "22222222-2222-2222-2222-222222222222"
DOC_SHARED_ID = "33333333-3333-3333-3333-333333333333"


# ── Fake Postgres layer ────────────────────────────────────────────────────────

class FakeCursor:
    """Simulates just enough of rag.documents / rag.doc_chunks to exercise
    retrieve/list/delete without a real Postgres. Vector similarity and
    full-text rank are stood in for by explicit `distance` (lower = closer)
    and `rank` (higher = better) fields on each seeded chunk row — the
    ordering behavior these drive is what real Postgres would compute from
    `embedding <=> ...` / `ts_rank(...)`, but the tenant_id WHERE-clause
    filtering below is exercised exactly as real SQL would apply it: on the
    candidate set, before any ordering."""

    def __init__(self, chunks=None, documents=None):
        self.chunks = chunks or []
        self.documents = documents or {}
        self.eval_sets = []  # every upload writes one (eval_set is a required field)
        self.rowcount = 0
        self._result = []

    def execute(self, query, params=None):
        q = " ".join(query.split()).upper()
        params = params or ()

        if "ORDER BY EMBEDDING" in q:
            tenant_id, _vector, limit = params
            rows = [c for c in self.chunks if c["tenant_id"] == tenant_id and c["effective_to"] is None]
            rows.sort(key=lambda c: c["distance"])
            self._result = rows[:limit]

        elif "PLAINTO_TSQUERY" in q and "SELECT ID, DOC_ID" in q:
            tenant_id, _q1, _q2, limit = params
            rows = [c for c in self.chunks if c["tenant_id"] == tenant_id and c["effective_to"] is None]
            rows.sort(key=lambda c: -c["rank"])
            self._result = rows[:limit]

        elif q.startswith("SELECT TENANT_ID FROM RAG.DOCUMENTS WHERE ID"):
            (doc_id,) = params
            doc = self.documents.get(doc_id)
            self._result = [{"tenant_id": doc["tenant_id"]}] if doc else []

        elif q.startswith("SELECT ID, FILENAME, CURRENT_VERSION, UPLOADED_AT"):
            (tenant_id,) = params
            self._result = [
                {"id": doc_id, **{k: v for k, v in doc.items() if k != "tenant_id"}}
                for doc_id, doc in self.documents.items() if doc["tenant_id"] == tenant_id
            ]

        elif q.startswith("SELECT ID FROM RAG.DOCUMENTS WHERE ID"):
            doc_id, tenant_id = params
            doc = self.documents.get(doc_id)
            self._result = [{"id": doc_id}] if doc and doc["tenant_id"] == tenant_id else []

        elif q.startswith("UPDATE RAG.DOC_CHUNKS SET EFFECTIVE_TO"):
            doc_id, tenant_id = params
            matched = [c for c in self.chunks
                       if c["doc_id"] == doc_id and c["tenant_id"] == tenant_id and c["effective_to"] is None]
            for c in matched:
                c["effective_to"] = "now"
            self.rowcount = len(matched)

        elif q.startswith("SELECT EVAL_HASH, VERSION FROM RAG.EVAL_SETS"):
            (doc_id,) = params
            matches = [e for e in self.eval_sets if e["doc_id"] == doc_id]
            self._result = [matches[-1]] if matches else []

        elif q.startswith("INSERT INTO RAG.EVAL_SETS"):
            tenant_id, doc_id, version, eval_hash, _queries_json = params
            self.eval_sets.append({"tenant_id": tenant_id, "doc_id": doc_id, "version": version, "eval_hash": eval_hash})

        else:
            raise AssertionError(f"Unexpected query in FakeCursor: {query!r}")

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return self._result


def _fake_get_cursor(cursor):
    @contextmanager
    def _get_cursor(commit=False):
        yield cursor
    return _get_cursor


def _chunk(id, tenant_id, doc_id="doc-1", doc_version="v1", section_ref=None,
           chunk_text="text", distance=0.5, rank=0.5, effective_to=None):
    return dict(id=id, tenant_id=tenant_id, doc_id=doc_id, doc_version=doc_version,
                section_ref=section_ref, chunk_text=chunk_text, distance=distance,
                rank=rank, effective_to=effective_to)


def _patched(cursor, embed_return=None):
    return (
        patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)),
        patch("rag_service.main._embed_batch", return_value=embed_return or [[0.0] * 1536]),
    )


# ── /v1/retrieve — tenant isolation (the critical requirement) ────────────────

class TestRetrieveTenantIsolation:

    def test_tenant_a_never_sees_tenant_bs_chunk_even_when_it_is_the_closer_match(self):
        chunks = [
            _chunk("a1", TENANT_A, chunk_text="tenant A content", distance=0.90, rank=0.10),
            _chunk("b1", TENANT_B, chunk_text="tenant B content", distance=0.01, rank=0.99),
        ]
        cursor = FakeCursor(chunks=chunks)
        p1, p2 = _patched(cursor)
        with p1, p2:
            resp = client.post(
                "/v1/retrieve",
                json={"tenant_id": TENANT_A, "query": "anything", "top_k": 5},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert results[0]["chunk_id"] == "a1"
        assert all(r["chunk_id"] != "b1" for r in results)

    def test_tenant_b_sees_only_its_own_chunk(self):
        chunks = [
            _chunk("a1", TENANT_A, chunk_text="tenant A content", distance=0.01, rank=0.99),
            _chunk("b1", TENANT_B, chunk_text="tenant B content", distance=0.90, rank=0.10),
        ]
        cursor = FakeCursor(chunks=chunks)
        p1, p2 = _patched(cursor)
        with p1, p2:
            resp = client.post(
                "/v1/retrieve",
                json={"tenant_id": TENANT_B, "query": "anything", "top_k": 5},
                headers={"Authorization": f"Bearer {TENANT_B}"},
            )
        results = resp.json()
        assert [r["chunk_id"] for r in results] == ["b1"]

    def test_superseded_chunk_is_excluded_even_for_the_right_tenant(self):
        chunks = [
            _chunk("a1", TENANT_A, effective_to="past"),
            _chunk("a2", TENANT_A, effective_to=None),
        ]
        cursor = FakeCursor(chunks=chunks)
        p1, p2 = _patched(cursor)
        with p1, p2:
            resp = client.post(
                "/v1/retrieve",
                json={"tenant_id": TENANT_A, "query": "anything"},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        results = resp.json()
        assert [r["chunk_id"] for r in results] == ["a2"]

    def test_body_tenant_id_mismatched_with_token_is_rejected(self):
        cursor = FakeCursor(chunks=[])
        p1, p2 = _patched(cursor)
        with p1, p2:
            resp = client.post(
                "/v1/retrieve",
                json={"tenant_id": TENANT_B, "query": "anything"},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 403

    def test_body_tenant_id_cannot_be_used_to_read_another_tenants_data(self):
        """Even if a caller holding tenant A's token claims tenant B's
        tenant_id in the body, the query must still run (and be rejected)
        rather than ever using the body's value as the actual filter."""
        chunks = [_chunk("b1", TENANT_B, distance=0.01, rank=0.99)]
        cursor = FakeCursor(chunks=chunks)
        p1, p2 = _patched(cursor)
        with p1, p2:
            resp = client.post(
                "/v1/retrieve",
                json={"tenant_id": TENANT_B, "query": "anything"},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 403
        # Confirm no query against doc_chunks was even attempted.
        assert cursor._result == []


class RecordingCursor:
    """Captures every raw (query, params) pair passed to execute() —
    unlike FakeCursor, this doesn't simulate row filtering at all. Used to
    inspect the actual SQL/parameter structure the endpoint builds, not
    just its behavior: proving tenant_id is a genuine bound WHERE-clause
    parameter on both underlying queries, positioned before ORDER BY/LIMIT,
    never a post-filter and never interpolated into the query text."""

    def __init__(self):
        self.calls = []

    def execute(self, query, params=None):
        self.calls.append((" ".join(query.split()), params or ()))

    def fetchall(self):
        return []

    def fetchone(self):
        return None


class TestRetrieveQueryStructure:

    def test_tenant_id_is_a_bound_where_clause_parameter_on_both_queries(self):
        cursor = RecordingCursor()
        p1, p2 = _patched(cursor)
        with p1, p2:
            resp = client.post(
                "/v1/retrieve",
                json={"tenant_id": TENANT_A, "query": "debt to income ratio"},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 200

        # Exactly two queries: the vector-similarity query and the
        # full-text query — both against rag.doc_chunks.
        assert len(cursor.calls) == 2

        for query, params in cursor.calls:
            upper = query.upper()
            assert "WHERE" in upper and "ORDER BY" in upper
            where_idx = upper.index("WHERE")
            order_idx = upper.index("ORDER BY")
            tenant_idx = upper.index("TENANT_ID")
            # tenant_id must be part of the WHERE clause, evaluated before
            # any ranking/ordering — never a post-filter on already-ranked
            # results.
            assert where_idx < tenant_idx < order_idx

            # The actual value must arrive as a bound parameter, not
            # string-interpolated into the SQL text — %s placeholders stay
            # literal in the query string, and the real tenant_id shows up
            # only in the params tuple.
            assert "%s" in query
            assert TENANT_A not in query
            assert params[0] == TENANT_A

    def test_effective_to_is_null_is_also_in_both_where_clauses(self):
        """The other half of tenant isolation: a superseded chunk must be
        excluded at the same WHERE-clause stage, not filtered out after
        the fact either."""
        cursor = RecordingCursor()
        p1, p2 = _patched(cursor)
        with p1, p2:
            client.post(
                "/v1/retrieve",
                json={"tenant_id": TENANT_A, "query": "anything"},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        for query, _params in cursor.calls:
            upper = query.upper()
            assert "EFFECTIVE_TO IS NULL" in upper
            assert upper.index("WHERE") < upper.index("EFFECTIVE_TO IS NULL") < upper.index("ORDER BY")


class TestRetrieveResponseShape:

    def test_response_includes_expected_fields(self):
        chunks = [_chunk("a1", TENANT_A, doc_id="doc-9", doc_version="v3",
                          section_ref="2.1", chunk_text="hello world")]
        cursor = FakeCursor(chunks=chunks)
        p1, p2 = _patched(cursor)
        with p1, p2:
            resp = client.post(
                "/v1/retrieve",
                json={"tenant_id": TENANT_A, "query": "hello"},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        row = resp.json()[0]
        assert row["doc_id"] == "doc-9"
        assert row["doc_version"] == "v3"
        assert row["section_ref"] == "2.1"
        assert row["chunk_text"] == "hello world"
        assert isinstance(row["score"], float) and row["score"] > 0

    def test_top_k_limits_results(self):
        chunks = [_chunk(f"c{i}", TENANT_A, distance=i / 10, rank=1 - i / 10) for i in range(10)]
        cursor = FakeCursor(chunks=chunks)
        p1, p2 = _patched(cursor)
        with p1, p2:
            resp = client.post(
                "/v1/retrieve",
                json={"tenant_id": TENANT_A, "query": "anything", "top_k": 3},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert len(resp.json()) == 3

    def test_chunk_ranked_first_in_both_lists_outranks_one_ranked_first_in_only_one(self):
        chunks = [
            _chunk("both",     TENANT_A, distance=0.1, rank=0.9),   # #1 vector, #1 text
            _chunk("vec_only", TENANT_A, distance=0.2, rank=0.1),   # #2 vector, last text
        ]
        cursor = FakeCursor(chunks=chunks)
        p1, p2 = _patched(cursor)
        with p1, p2:
            resp = client.post(
                "/v1/retrieve",
                json={"tenant_id": TENANT_A, "query": "anything", "top_k": 2},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        results = resp.json()
        assert results[0]["chunk_id"] == "both"
        assert results[0]["score"] > results[1]["score"]

    def test_requires_auth(self):
        resp = client.post("/v1/retrieve", json={"tenant_id": TENANT_A, "query": "x"})
        assert resp.status_code == 401


# ── /v1/documents ──────────────────────────────────────────────────────────────

class TestUploadDocument:

    def test_upload_generates_doc_id_and_enqueues_task(self):
        cursor = FakeCursor()
        fake_task = MagicMock(id="task-123")
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.main._celery_app") as fake_celery:
            fake_celery.send_task.return_value = fake_task
            resp = client.post(
                "/v1/documents",
                data={"eval_set": DEFAULT_EVAL_SET},
                files={"file": ("policy.md", b"# Title\n\nBody text.", "text/markdown")},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 202
        body = resp.json()
        assert body["task_id"] == "task-123"
        assert body["filename"] == "policy.md"
        assert body["status"] == "queued"
        assert body["doc_id"]

        kwargs = fake_celery.send_task.call_args.kwargs["kwargs"]
        assert kwargs["tenant_id"] == TENANT_A
        assert kwargs["doc_id"] == body["doc_id"]
        assert kwargs["filename"] == "policy.md"

    def test_rejects_unsupported_extension(self):
        cursor = FakeCursor()
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.main._celery_app") as fake_celery:
            resp = client.post(
                "/v1/documents",
                data={"eval_set": DEFAULT_EVAL_SET},
                files={"file": ("data.csv", b"a,b,c", "text/csv")},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 400
        fake_celery.send_task.assert_not_called()

    def test_rejects_empty_file(self):
        cursor = FakeCursor()
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.main._celery_app") as fake_celery:
            resp = client.post(
                "/v1/documents",
                data={"eval_set": DEFAULT_EVAL_SET},
                files={"file": ("policy.md", b"", "text/markdown")},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 400
        fake_celery.send_task.assert_not_called()

    def test_cannot_version_another_tenants_doc_id(self):
        cursor = FakeCursor(documents={DOC_B_ID: {"tenant_id": TENANT_B, "filename": "x", "current_version": "v1"}})
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.main._celery_app") as fake_celery:
            resp = client.post(
                "/v1/documents",
                data={"doc_id": DOC_B_ID, "eval_set": DEFAULT_EVAL_SET},
                files={"file": ("policy.md", b"content", "text/markdown")},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 403
        fake_celery.send_task.assert_not_called()

    def test_can_version_own_existing_doc_id(self):
        cursor = FakeCursor(documents={DOC_A_ID: {"tenant_id": TENANT_A, "filename": "x", "current_version": "v1"}})
        fake_task = MagicMock(id="task-9")
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.main._celery_app") as fake_celery:
            fake_celery.send_task.return_value = fake_task
            resp = client.post(
                "/v1/documents",
                data={"doc_id": DOC_A_ID, "eval_set": DEFAULT_EVAL_SET},
                files={"file": ("policy.md", b"content", "text/markdown")},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 202
        assert resp.json()["doc_id"] == DOC_A_ID

    def test_requires_auth(self):
        resp = client.post(
            "/v1/documents",
            data={"eval_set": DEFAULT_EVAL_SET},
            files={"file": ("policy.md", b"x", "text/markdown")},
        )
        assert resp.status_code == 401

    def test_malformed_doc_id_is_rejected_cleanly_not_500(self):
        """rag.documents.id is a UUID column — a non-UUID doc_id must not
        reach Postgres and blow up as an unhandled 500."""
        cursor = FakeCursor()
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.main._celery_app") as fake_celery:
            resp = client.post(
                "/v1/documents",
                data={"doc_id": "not-a-uuid", "eval_set": DEFAULT_EVAL_SET},
                files={"file": ("policy.md", b"content", "text/markdown")},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 400
        fake_celery.send_task.assert_not_called()

    def test_rejects_upload_without_eval_set(self):
        """eval_set is a required Form field now, not optional — FastAPI's
        own request-validation rejects a missing one with a 422 before the
        endpoint body ever runs."""
        resp = client.post(
            "/v1/documents",
            files={"file": ("policy.md", b"content", "text/markdown")},
            headers={"Authorization": f"Bearer {TENANT_A}"},
        )
        assert resp.status_code == 422


class TestDocumentStatus:

    def test_requires_auth(self):
        resp = client.get(f"/v1/documents/status/{DOC_A_ID}")
        assert resp.status_code == 401

    def test_returns_recorded_status(self):
        cursor = FakeCursor(documents={DOC_A_ID: {"tenant_id": TENANT_A, "filename": "a.md", "current_version": "v1"}})
        fake_redis = MagicMock()
        fake_redis.get.side_effect = lambda key: (
            "processing" if key == f"ingest_status:{DOC_A_ID}" else None
        )
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.ingest_task._redis", return_value=fake_redis):
            resp = client.get(f"/v1/documents/status/{DOC_A_ID}", headers={"Authorization": f"Bearer {TENANT_A}"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_id"] == DOC_A_ID
        assert body["status"] == "processing"
        assert body["error"] is None

    def test_returns_failed_status_with_error_message(self):
        cursor = FakeCursor(documents={DOC_A_ID: {"tenant_id": TENANT_A, "filename": "a.md", "current_version": "v1"}})
        fake_redis = MagicMock()
        fake_redis.get.side_effect = lambda key: (
            "failed" if key == f"ingest_status:{DOC_A_ID}" else
            "OpenAI error" if key == f"ingest_status:{DOC_A_ID}:error" else None
        )
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.ingest_task._redis", return_value=fake_redis):
            resp = client.get(f"/v1/documents/status/{DOC_A_ID}", headers={"Authorization": f"Bearer {TENANT_A}"})
        body = resp.json()
        assert body["status"] == "failed"
        assert body["error"] == "OpenAI error"

    def test_404_for_document_belonging_to_another_tenant(self):
        cursor = FakeCursor(documents={DOC_B_ID: {"tenant_id": TENANT_B, "filename": "b.md", "current_version": "v1"}})
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)):
            resp = client.get(f"/v1/documents/status/{DOC_B_ID}", headers={"Authorization": f"Bearer {TENANT_A}"})
        assert resp.status_code == 404

    def test_404_for_nonexistent_document(self):
        cursor = FakeCursor()
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)):
            resp = client.get(f"/v1/documents/status/{DOC_A_ID}", headers={"Authorization": f"Bearer {TENANT_A}"})
        assert resp.status_code == 404

    def test_malformed_doc_id_returns_404_not_500(self):
        cursor = FakeCursor()
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)):
            resp = client.get("/v1/documents/status/not-a-uuid", headers={"Authorization": f"Bearer {TENANT_A}"})
        assert resp.status_code == 404


class TestListDocuments:

    def test_lists_only_this_tenants_documents(self):
        cursor = FakeCursor(documents={
            "doc-a": {"tenant_id": TENANT_A, "filename": "a.md", "current_version": "v1", "uploaded_at": "t1"},
            "doc-b": {"tenant_id": TENANT_B, "filename": "b.md", "current_version": "v1", "uploaded_at": "t2"},
        })
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)):
            resp = client.get("/v1/documents", headers={"Authorization": f"Bearer {TENANT_A}"})
        assert resp.status_code == 200
        ids = [d["id"] for d in resp.json()]
        assert ids == ["doc-a"]

    def test_requires_auth(self):
        resp = client.get("/v1/documents")
        assert resp.status_code == 401


class TestDeleteDocument:

    def test_supersedes_all_current_chunks(self):
        chunks = [_chunk("c1", TENANT_A, doc_id=DOC_A_ID), _chunk("c2", TENANT_A, doc_id=DOC_A_ID)]
        cursor = FakeCursor(
            chunks=chunks,
            documents={DOC_A_ID: {"tenant_id": TENANT_A, "filename": "a.md", "current_version": "v1"}},
        )
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)):
            resp = client.delete(f"/v1/documents/{DOC_A_ID}", headers={"Authorization": f"Bearer {TENANT_A}"})
        assert resp.status_code == 200
        assert resp.json()["chunks_deleted"] == 2
        assert all(c["effective_to"] is not None for c in chunks)

    def test_does_not_touch_another_tenants_chunks_sharing_the_same_doc_id(self):
        """Defense in depth: even if a doc_id collided across tenants, the
        UPDATE is still tenant-scoped, not just doc_id-scoped."""
        chunks = [_chunk("c1", TENANT_A, doc_id=DOC_SHARED_ID), _chunk("c2", TENANT_B, doc_id=DOC_SHARED_ID)]
        cursor = FakeCursor(
            chunks=chunks,
            documents={DOC_SHARED_ID: {"tenant_id": TENANT_A, "filename": "a.md", "current_version": "v1"}},
        )
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)):
            resp = client.delete(f"/v1/documents/{DOC_SHARED_ID}", headers={"Authorization": f"Bearer {TENANT_A}"})
        assert resp.status_code == 200
        assert resp.json()["chunks_deleted"] == 1
        assert chunks[0]["effective_to"] is not None    # tenant A's row: superseded
        assert chunks[1]["effective_to"] is None         # tenant B's row: untouched

    def test_404_for_document_belonging_to_another_tenant(self):
        cursor = FakeCursor(documents={DOC_B_ID: {"tenant_id": TENANT_B, "filename": "b.md", "current_version": "v1"}})
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)):
            resp = client.delete(f"/v1/documents/{DOC_B_ID}", headers={"Authorization": f"Bearer {TENANT_A}"})
        assert resp.status_code == 404

    def test_404_for_nonexistent_document(self):
        cursor = FakeCursor()
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)):
            resp = client.delete("/v1/documents/does-not-exist", headers={"Authorization": f"Bearer {TENANT_A}"})
        assert resp.status_code == 404

    def test_requires_auth(self):
        resp = client.delete("/v1/documents/doc-a")
        assert resp.status_code == 401

    def test_malformed_doc_id_returns_404_not_500(self):
        """rag.documents.id is a UUID column — a non-UUID path segment must
        not reach Postgres and blow up as an unhandled 500."""
        cursor = FakeCursor()
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)):
            resp = client.delete("/v1/documents/not-a-uuid", headers={"Authorization": f"Bearer {TENANT_A}"})
        assert resp.status_code == 404


# ── Quota enforcement ───────────────────────────────────────────────────────────

def _quota_bypass(ingestion_limit=1_000_000, retrieval_limit=1_000_000, existing_usage=None):
    """Overrides this file's default _generous_quota fixture for a specific
    test: a chosen (ingestion_limit, retrieval_limit) plan and a Redis
    stand-in reporting `existing_usage` (or zero) as the prior count —
    exercises the real _check_and_increment_quota logic, not a mock of it."""
    fake_redis = MagicMock()
    fake_redis.get.return_value = None if existing_usage is None else str(existing_usage)
    fake_redis.incrby.side_effect = lambda key, n: (existing_usage or 0) + n
    return (
        patch("rag_service.main._get_rag_plan_quotas", return_value=(ingestion_limit, retrieval_limit)),
        patch("rag_service.main._redis", return_value=fake_redis),
    )


class TestIngestionQuota:

    def test_returns_429_when_ingestion_quota_exceeded(self):
        cursor = FakeCursor()
        p1, p2 = _quota_bypass(ingestion_limit=0)
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.main._celery_app") as fake_celery, \
             p1, p2:
            resp = client.post(
                "/v1/documents",
                data={"eval_set": DEFAULT_EVAL_SET},
                files={"file": ("policy.md", b"content", "text/markdown")},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 429
        assert "ingestion" in resp.json()["detail"].lower()
        fake_celery.send_task.assert_not_called()   # rejected before ever enqueueing

    def test_succeeds_at_exactly_the_limit(self):
        cursor = FakeCursor()
        fake_task = MagicMock(id="t1")
        p1, p2 = _quota_bypass(ingestion_limit=5, existing_usage=4)
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.main._celery_app") as fake_celery, \
             p1, p2:
            fake_celery.send_task.return_value = fake_task
            resp = client.post(
                "/v1/documents",
                data={"eval_set": DEFAULT_EVAL_SET},
                files={"file": ("policy.md", b"content", "text/markdown")},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 202

    def test_rejects_the_request_that_would_exceed_the_limit(self):
        cursor = FakeCursor()
        p1, p2 = _quota_bypass(ingestion_limit=5, existing_usage=5)
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.main._celery_app") as fake_celery, \
             p1, p2:
            resp = client.post(
                "/v1/documents",
                data={"eval_set": DEFAULT_EVAL_SET},
                files={"file": ("policy.md", b"content", "text/markdown")},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 429
        fake_celery.send_task.assert_not_called()

    def test_uses_the_rag_ingest_prefix_not_the_prediction_quota_prefix(self):
        """Defense against the exact collision the shared Redis instance
        makes possible: this must never touch app.main's "quota:" key."""
        cursor = FakeCursor()
        fake_task = MagicMock(id="t1")
        fake_redis = MagicMock()
        fake_redis.get.return_value = None
        fake_redis.incrby.side_effect = lambda key, n: n
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.main._celery_app") as fake_celery, \
             patch("rag_service.main._get_rag_plan_quotas", return_value=(100, 100)), \
             patch("rag_service.main._redis", return_value=fake_redis):
            fake_celery.send_task.return_value = fake_task
            client.post(
                "/v1/documents",
                data={"eval_set": DEFAULT_EVAL_SET},
                files={"file": ("policy.md", b"content", "text/markdown")},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        incr_key = fake_redis.incrby.call_args[0][0]
        assert incr_key.startswith("rag_ingest_quota:")
        assert not incr_key.startswith("quota:")


class TestRetrievalQuota:

    def test_returns_429_when_retrieval_quota_exceeded(self):
        cursor = FakeCursor(chunks=[_chunk("a1", TENANT_A)])
        p1, p2 = _quota_bypass(retrieval_limit=0)
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.main._embed_batch") as fake_embed, \
             p1, p2:
            resp = client.post(
                "/v1/retrieve",
                json={"tenant_id": TENANT_A, "query": "anything"},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 429
        assert "retrieval" in resp.json()["detail"].lower()
        fake_embed.assert_not_called()   # rejected before spending an embedding call

    def test_succeeds_under_the_limit(self):
        cursor = FakeCursor(chunks=[_chunk("a1", TENANT_A)])
        p1, p2 = _quota_bypass(retrieval_limit=10, existing_usage=3)
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.main._embed_batch", return_value=[[0.0] * 1536]), \
             p1, p2:
            resp = client.post(
                "/v1/retrieve",
                json={"tenant_id": TENANT_A, "query": "anything"},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        assert resp.status_code == 200

    def test_uses_the_rag_retrieve_prefix_not_the_prediction_quota_prefix(self):
        cursor = FakeCursor(chunks=[_chunk("a1", TENANT_A)])
        fake_redis = MagicMock()
        fake_redis.get.return_value = None
        fake_redis.incrby.side_effect = lambda key, n: n
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.main._embed_batch", return_value=[[0.0] * 1536]), \
             patch("rag_service.main._get_rag_plan_quotas", return_value=(100, 100)), \
             patch("rag_service.main._redis", return_value=fake_redis):
            client.post(
                "/v1/retrieve",
                json={"tenant_id": TENANT_A, "query": "anything"},
                headers={"Authorization": f"Bearer {TENANT_A}"},
            )
        incr_key = fake_redis.incrby.call_args[0][0]
        assert incr_key.startswith("rag_retrieve_quota:")
        assert not incr_key.startswith("quota:")

    def test_ingestion_and_retrieval_quotas_are_independent(self):
        """Uploading up to the ingestion limit must not affect how many
        retrievals are still allowed, and vice versa — they're separate
        counters under separate prefixes on the same Redis instance."""
        store = {}

        def fake_get(key):
            return str(store[key]) if key in store else None

        def fake_incrby(key, n):
            store[key] = store.get(key, 0) + n
            return store[key]

        fake_redis = MagicMock()
        fake_redis.get.side_effect = fake_get
        fake_redis.incrby.side_effect = fake_incrby

        cursor = FakeCursor(chunks=[_chunk("a1", TENANT_A)])
        fake_task = MagicMock(id="t1")
        with patch("rag_service.main.get_cursor", _fake_get_cursor(cursor)), \
             patch("rag_service.main._celery_app") as fake_celery, \
             patch("rag_service.main._embed_batch", return_value=[[0.0] * 1536]), \
             patch("rag_service.main._get_rag_plan_quotas", return_value=(1, 1)), \
             patch("rag_service.main._redis", return_value=fake_redis):
            fake_celery.send_task.return_value = fake_task
            # Exhaust the ingestion quota (limit=1) — one upload succeeds...
            r1 = client.post("/v1/documents", data={"eval_set": DEFAULT_EVAL_SET},
                              files={"file": ("a.md", b"x", "text/markdown")},
                              headers={"Authorization": f"Bearer {TENANT_A}"})
            # ...a second upload is rejected...
            r2 = client.post("/v1/documents", data={"eval_set": DEFAULT_EVAL_SET},
                              files={"file": ("b.md", b"y", "text/markdown")},
                              headers={"Authorization": f"Bearer {TENANT_A}"})
            # ...but retrieval (limit=1, separate counter) still succeeds.
            r3 = client.post("/v1/retrieve", json={"tenant_id": TENANT_A, "query": "q"},
                              headers={"Authorization": f"Bearer {TENANT_A}"})
        assert r1.status_code == 202
        assert r2.status_code == 429
        assert r3.status_code == 200


class TestUsageEndpoint:

    def test_returns_ingestion_and_retrieval_usage_nested(self):
        fake_redis = MagicMock()
        fake_redis.get.side_effect = lambda key: (
            "3" if key.startswith("rag_ingest_quota:") else "7" if key.startswith("rag_retrieve_quota:") else None
        )
        with patch("rag_service.main._get_rag_plan_quotas", return_value=(100, 200)), \
             patch("rag_service.main._redis", return_value=fake_redis), \
             patch("app.db.get_tenant_plan", return_value="free"):
            resp = client.get("/v1/usage", headers={"Authorization": f"Bearer {TENANT_A}"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["tenant_id"] == TENANT_A
        assert body["plan"] == "free"
        assert body["ingestion"] == {"used": 3, "limit": 100}
        assert body["retrieval"] == {"used": 7, "limit": 200}

    def test_zero_usage_when_nothing_recorded_yet(self):
        fake_redis = MagicMock()
        fake_redis.get.return_value = None
        with patch("rag_service.main._get_rag_plan_quotas", return_value=(5, 50)), \
             patch("rag_service.main._redis", return_value=fake_redis), \
             patch("app.db.get_tenant_plan", return_value="free"):
            resp = client.get("/v1/usage", headers={"Authorization": f"Bearer {TENANT_A}"})
        body = resp.json()
        assert body["ingestion"]["used"] == 0
        assert body["retrieval"]["used"] == 0

    def test_requires_auth(self):
        resp = client.get("/v1/usage")
        assert resp.status_code == 401
