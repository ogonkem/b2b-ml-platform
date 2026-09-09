# Requires the full docker stack running: docker compose up -d
#
# Additionally requires a real OPENAI_API_KEY in .env — every test in this
# file does real ingestion and/or real retrieval, both of which call
# OpenAI to embed text (see rag_harness/ingest_task.py's module docstring
# for the tradeoff). Without a real key, every test here is skipped rather
# than failing — this is a genuine external-credential gate, not a bug.
#
# API_TOKENS must include commercial_bank / microfinance_sacco /
# informal_digital_lender (see .env.example) — these double as both the
# bearer token and the tenant_id, letting a caller authenticate AS one of
# rule_engine/thresholds.py's three seeded demo tenants.
import time
from pathlib import Path

import httpx
import psycopg2
import pytest
import redis
from dotenv import dotenv_values
from minio import Minio
from psycopg2.extras import RealDictCursor

_env = dotenv_values(Path(__file__).resolve().parent.parent.parent / ".env")

OPENAI_API_KEY = _env.get("OPENAI_API_KEY", "")
pytestmark = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="requires a real OPENAI_API_KEY in .env — rag_harness embeds every "
           "ingested chunk and every retrieval query via OpenAI",
)

RAG_HARNESS_URL = "http://localhost:8001"

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "policy_docs"

# tenant_id -> fixture filename, matching rule_engine/thresholds.py's three
# seeded tenants (Prompt 2.1) exactly, so a query against one tenant's
# ingested policy can be checked against rule_engine's own understanding
# of what that tenant's policy contains.
TENANT_DOCS = {
    "commercial_bank":         "1_commercial_bank_lending_policy.md",
    "microfinance_sacco":      "2_microfinance_sacco_policy.md",
    "informal_digital_lender": "3_informal_digital_lender_policy.md",
}

# A query per tenant plus a keyword that appears in *that* tenant's fixture
# doc and nowhere in the other two — proves the top result is actually
# relevant, not just that the call succeeded.
RELEVANT_QUERY = {
    "commercial_bank":         "collateral valuation standards for secured loans",
    "microfinance_sacco":      "group guarantee solidarity lending",
    "informal_digital_lender": "mobile money automated lending decision",
}
DISTINCTIVE_KEYWORD = {
    "commercial_bank":         "collateral",
    "microfinance_sacco":      "solidarity",
    "informal_digital_lender": "mobile money",
}

QUOTA_TENANT = _env.get("QUOTA_TEST_TOKEN", "quota-test-token")


def _pg_conn():
    return psycopg2.connect(
        host="localhost",
        port=int(_env.get("POSTGRES_PORT", 5432)),
        user=_env.get("POSTGRES_USER"),
        password=_env.get("POSTGRES_PASSWORD"),
        dbname=_env.get("POSTGRES_DB"),
    )


def _minio_client():
    return Minio(
        _env.get("MINIO_PUBLIC_ENDPOINT", "localhost:9000"),
        access_key=_env.get("MINIO_ROOT_USER"),
        secret_key=_env.get("MINIO_ROOT_PASSWORD"),
        secure=False,
    )


def _redis_client():
    return redis.Redis(host="localhost", port=int(_env.get("REDIS_PORT", 6379)), decode_responses=True)


def _wait_for_ingest_complete(doc_id: str, tenant: str, timeout: float = 60.0) -> dict:
    """Polls GET /v1/documents/status/{doc_id} — same status field
    (queued/processing/complete/failed) pages/Policies.tsx polls."""
    headers = {"Authorization": f"Bearer {tenant}"}
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = httpx.get(f"{RAG_HARNESS_URL}/v1/documents/status/{doc_id}", headers=headers, timeout=10.0)
        resp.raise_for_status()
        last = resp.json()
        if last["status"] in ("complete", "failed"):
            return last
        time.sleep(1)
    raise TimeoutError(f"doc {doc_id} ({tenant}) did not finish ingestion within {timeout}s: {last}")


def _ingest(tenant: str, filename: str) -> dict:
    headers = {"Authorization": f"Bearer {tenant}"}
    content = (FIXTURES_DIR / filename).read_bytes()
    resp = httpx.post(
        f"{RAG_HARNESS_URL}/v1/documents",
        headers=headers,
        files={"file": (filename, content, "text/markdown")},
        timeout=30.0,
    )
    assert resp.status_code == 202, resp.text
    upload = resp.json()
    status = _wait_for_ingest_complete(upload["doc_id"], tenant)
    assert status["status"] == "complete", f"Ingestion failed for {tenant}: {status.get('error')}"
    return upload


@pytest.fixture(scope="module")
def ingested_docs():
    """Ingests all three fixture docs once per test module run — the
    retrieval, relevance, and cross-tenant tests all read from this same
    ingested state rather than each re-ingesting from scratch."""
    return {tenant: _ingest(tenant, filename) for tenant, filename in TENANT_DOCS.items()}


# ── Ingestion lands in doc_chunks / MinIO correctly ─────────────────────────

def test_ingest_all_three_docs_lands_in_doc_chunks_with_correct_metadata(ingested_docs):
    conn = _pg_conn()
    try:
        for tenant, upload in ingested_docs.items():
            doc_id = upload["doc_id"]
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT tenant_id, doc_id, doc_version, chunk_text, minio_object_path
                       FROM rag.doc_chunks WHERE doc_id = %s AND effective_to IS NULL""",
                    (doc_id,),
                )
                rows = cur.fetchall()
            assert len(rows) > 0, f"No chunks landed for tenant {tenant} (doc_id={doc_id})"
            for row in rows:
                assert row["tenant_id"] == tenant
                assert str(row["doc_id"]) == doc_id
                assert row["doc_version"] == "v1"
                assert row["minio_object_path"] == f"{tenant}/{doc_id}/v1/original.md"
    finally:
        conn.close()


def test_ingest_raw_file_lands_in_minio_at_expected_path(ingested_docs):
    minio = _minio_client()
    for tenant, upload in ingested_docs.items():
        doc_id = upload["doc_id"]
        expected_path = f"{tenant}/{doc_id}/v1/original.md"
        stat = minio.stat_object("doc-chunks-raw", expected_path)
        assert stat.size > 0


# ── Retrieval relevance per tenant ──────────────────────────────────────────

def test_retrieve_returns_relevant_top_result_per_tenant(ingested_docs):
    for tenant in TENANT_DOCS:
        resp = httpx.post(
            f"{RAG_HARNESS_URL}/v1/retrieve",
            headers={"Authorization": f"Bearer {tenant}"},
            json={"tenant_id": tenant, "query": RELEVANT_QUERY[tenant], "top_k": 3},
            timeout=30.0,
        )
        assert resp.status_code == 200, resp.text
        results = resp.json()
        assert len(results) > 0, f"No results for {tenant} — check ingestion actually completed"

        top = results[0]
        assert top["doc_version"] == "v1"
        assert DISTINCTIVE_KEYWORD[tenant].lower() in top["chunk_text"].lower(), (
            f"Top result for {tenant} doesn't mention '{DISTINCTIVE_KEYWORD[tenant]}' — "
            f"got: {top['chunk_text'][:200]!r}"
        )
        # doc 3 (informal_digital_lender) falls back to unstructured
        # paragraph chunking with no section_ref — expected, not a bug
        # (see rule_engine/thresholds.py's own comment on this tenant).
        if tenant == "informal_digital_lender":
            assert top["section_ref"] is None
        else:
            assert top["section_ref"] is not None


# ── Cross-tenant isolation, against the real running stack ─────────────────

def test_cross_tenant_isolation_even_when_other_tenant_is_the_closer_match(ingested_docs):
    """Query as microfinance_sacco with text that's topically about
    informal_digital_lender's content — the real hybrid search (vector +
    full-text) should find informal_digital_lender's chunk a much better
    match than anything in microfinance_sacco's own policy, which never
    mentions mobile money or automated scoring at all. Zero of its chunks
    may appear regardless."""
    resp = httpx.post(
        f"{RAG_HARNESS_URL}/v1/retrieve",
        headers={"Authorization": "Bearer microfinance_sacco"},
        json={"tenant_id": "microfinance_sacco", "query": RELEVANT_QUERY["informal_digital_lender"], "top_k": 10},
        timeout=30.0,
    )
    assert resp.status_code == 200, resp.text
    results = resp.json()

    informal_doc_id = ingested_docs["informal_digital_lender"]["doc_id"]
    assert all(r["doc_id"] != informal_doc_id for r in results)
    # Everything returned (if anything) must genuinely belong to the
    # querying tenant — there is no per-chunk tenant_id in the response,
    # so this is checked by doc_id ownership instead.
    sacco_doc_id = ingested_docs["microfinance_sacco"]["doc_id"]
    assert all(r["doc_id"] == sacco_doc_id for r in results)


# ── Retrieval quota: 429 + Redis TTL ─────────────────────────────────────────

def test_retrieval_quota_exceeded_returns_429_with_matching_ttl():
    """QUOTA_TEST_TOKEN (not the three demo tenants above) is the token this
    project already dedicates to quota-exhaustion tests — see CLAUDE.md and
    tests/integration/test_api_endpoints.py's own test_quota_enforced_after_limit.
    Reusing one of the three demo tenants here would burn out its retrieval
    quota for the rest of the month and break every other test in this file."""
    r = _redis_client()
    month = time.strftime("%Y_%m", time.gmtime())
    key = f"rag_retrieve_quota:{QUOTA_TENANT}:{month}"
    r.delete(key)   # start from a known state regardless of prior runs

    headers = {"Authorization": f"Bearer {QUOTA_TENANT}"}

    # Free plan's bundled retrieval quota (app/plans.py) is 50/mo — seed the
    # counter to one below the limit directly rather than making 49 real,
    # billed OpenAI embedding calls just to reach the same state.
    r.set(key, 49, ex=60 * 60 * 24 * 32)

    ok = httpx.post(
        f"{RAG_HARNESS_URL}/v1/retrieve",
        headers=headers,
        json={"tenant_id": QUOTA_TENANT, "query": "anything"},
        timeout=30.0,
    )
    assert ok.status_code == 200, ok.text   # the 50th call — exactly at the limit, still allowed

    exceeded = httpx.post(
        f"{RAG_HARNESS_URL}/v1/retrieve",
        headers=headers,
        json={"tenant_id": QUOTA_TENANT, "query": "anything"},
        timeout=30.0,
    )
    assert exceeded.status_code == 429, exceeded.text

    # Same ~32-day TTL pattern as app/main.py's own quota keys
    # (MONTHLY_QUOTA_LIMIT / check_and_increment_quota) — allow a little
    # slack for time elapsed since the key was set above.
    ttl = r.ttl(key)
    expected_ttl = 60 * 60 * 24 * 32
    assert expected_ttl - 30 <= ttl <= expected_ttl

    r.delete(key)   # leave the tenant's quota clean for whatever runs next
