"""
rag_service/main.py
Standalone FastAPI service for the RAG document-ingestion/retrieval
pipeline. Runs as its own container (see docker-compose.yml's rag_service
service) but shares the main stack's existing infrastructure rather than
provisioning its own:
  - Postgres — same instance as the api, new `rag` schema (rag_service/db.py)
  - MinIO    — same instance as the api, new `doc-chunks-raw` bucket
  - Redis    — same instance as celery_worker, same Celery app (ingestion
               runs as the rag_service.ingest_task.ingest_document task);
               also used here directly for per-tenant ingestion/retrieval
               quota counters, mirroring app.main.check_and_increment_quota

Auth reuses app.auth.verify_token directly (not reimplemented here) — a
bearer value that already authenticates against Selastone's API (a static
API_TOKENS entry, a JWT from /auth/login, or an sk_ API key) resolves to the
same tenant_id here with zero extra setup.
"""
import base64
import os
import uuid
from datetime import datetime
from typing import List, Optional

import redis
from fastapi import FastAPI, File, Form, HTTPException, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from minio import Minio
from pydantic import BaseModel

from app.auth import security_scheme, verify_token
from rag_service.db import get_cursor, init_schema
from rag_service.ingest_task import (
    DOC_CHUNKS_BUCKET,
    _embed_batch,
    _extract_ext,
    _format_vector,
    get_ingest_status,
    set_ingest_status,
)

try:
    from celery_worker.celery_app import celery_app as _celery_app
except ImportError:
    _celery_app = None

app = FastAPI(
    title="Selastone RAG Service",
    version="0.2.0",
    description="Document ingestion and retrieval harness for RAG pipelines.",
)

# The frontend SPA calls this service directly (pages/Policies.tsx), not
# just through app.main — same CORS setup as app/main.py, same
# FRONTEND_ORIGINS env var, for the same reason (Vite dev server + the
# built/nginx-served bundle both need to be allowed at once).
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FRONTEND_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RETRIEVAL_CANDIDATE_LIMIT = 20   # per ranking method, before RRF merge
RRF_K = 60

# Same thin-client construction as app/main.py's minio_client — no wrapper,
# talked to directly wherever chunk storage is needed later.
minio_client = Minio(
    f"{os.environ.get('MINIO_HOST', 'localhost')}:{os.environ.get('MINIO_PORT', 9000)}",
    access_key=os.environ.get("MINIO_ROOT_USER"),
    secret_key=os.environ.get("MINIO_ROOT_PASSWORD"),
    secure=False,
)

# ── Quota ─────────────────────────────────────────────────────────────────────
# Mirrors app.main.check_and_increment_quota's exact pattern — same
# check-before-increment ordering, same atomic INCRBY, same ~32-day TTL —
# but under its own key prefix rather than reusing "quota:" literally.
# app.main's redis_client and this module's _redis() point at the same
# Redis instance/DB, so reusing "quota:{tenant}:{month}" verbatim would
# silently share (and corrupt) the same counter Selastone's own prediction
# quota already uses for that tenant. Distinct prefixes are what make "same
# key shape" safe to reuse without becoming "same key."
RAG_INGESTION_QUOTA_PREFIX = "rag_ingest_quota"
RAG_RETRIEVAL_QUOTA_PREFIX = "rag_retrieve_quota"

_redis_client = None


def _redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            decode_responses=True,
        )
    return _redis_client


def _check_and_increment_quota(tenant_id: str, key_prefix: str, label: str, increment: int, monthly_limit: int) -> int:
    key = f"{key_prefix}:{tenant_id}:{datetime.utcnow().strftime('%Y_%m')}"
    # Check before incrementing — incrementing first (then rejecting on
    # overage) would permanently charge the tenant for a request that never
    # went through.
    existing = int(_redis().get(key) or 0)
    if existing + increment > monthly_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Monthly RAG {label} quota exceeded",
        )
    current = _redis().incrby(key, increment)
    if current <= increment:
        # Key was just created (or reset) — ~1-month TTL for automatic reset.
        _redis().expire(key, 60 * 60 * 24 * 32)
    return current


def _current_usage(tenant_id: str, key_prefix: str) -> int:
    key = f"{key_prefix}:{tenant_id}:{datetime.utcnow().strftime('%Y_%m')}"
    return int(_redis().get(key) or 0)


def _get_rag_plan_quotas(tenant_id: str) -> tuple:
    """(ingestion_quota, retrieval_quota) bundled with this tenant's
    existing Selastone plan (app/plans.py) — no separate RAG billing yet."""
    from app.db import get_tenant_plan
    from app.plans import DEFAULT_PLAN, PLANS
    plan = get_tenant_plan(tenant_id)
    details = PLANS.get(plan, PLANS[DEFAULT_PLAN])
    return details["rag_ingestion_quota"], details["rag_retrieval_quota"]

# ── Idempotent DDL / bucket setup — safe to run on every startup ───────────
try:
    init_schema()
    print("[OK] rag schema ready")
except Exception as e:
    print(f"[WARN] rag schema init failed: {e}")

try:
    if not minio_client.bucket_exists(DOC_CHUNKS_BUCKET):
        minio_client.make_bucket(DOC_CHUNKS_BUCKET)
    print(f"[OK] MinIO bucket '{DOC_CHUNKS_BUCKET}' ready")
except Exception as e:
    print(f"[WARN] MinIO bucket init failed: {e}")


# ── Schemas ───────────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    tenant_id: str
    query: str
    top_k: int = 5


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_version: str
    section_ref: Optional[str] = None
    chunk_text: str
    score: float


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(*ranked_lists, k: int = RRF_K):
    """Standard RRF: each list contributes 1/(k + rank) per row (rank
    1-indexed), summed across lists. Rows absent from a list simply don't
    contribute from it — no penalty beyond not being there."""
    scores: dict = {}
    rows_by_id: dict = {}
    for ranked in ranked_lists:
        for rank, row in enumerate(ranked, start=1):
            rid = row["id"]
            rows_by_id[rid] = row
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
    return sorted(
        ((rows_by_id[rid], score) for rid, score in scores.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )


@app.post("/v1/retrieve", response_model=List[RetrievedChunk], summary="Hybrid vector + full-text retrieval")
async def retrieve(
    payload: RetrieveRequest,
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    tenant = verify_token(token)
    # tenant_id in the body only ever guards against a caller mismatching
    # its own token/body pairing — the token's tenant_id (not the body's)
    # is what actually gets used as the WHERE-clause filter below.
    if payload.tenant_id != tenant:
        raise HTTPException(status_code=403, detail="tenant_id does not match authenticated token")

    _, retrieval_quota = _get_rag_plan_quotas(tenant)
    _check_and_increment_quota(tenant, RAG_RETRIEVAL_QUOTA_PREFIX, "retrieval", increment=1, monthly_limit=retrieval_quota)

    query_vector = _format_vector(_embed_batch([payload.query])[0])

    with get_cursor() as cur:
        # tenant_id is a hard WHERE-clause filter in both queries, applied
        # before ORDER BY/LIMIT — never a post-filter on the merged result.
        # A closer semantic or lexical match belonging to another tenant
        # never enters the candidate set at all, so RRF has nothing of
        # theirs to (mis)rank in the first place.
        cur.execute(
            """SELECT id, doc_id, doc_version, section_ref, chunk_text
               FROM rag.doc_chunks
               WHERE tenant_id = %s AND effective_to IS NULL
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (tenant, query_vector, RETRIEVAL_CANDIDATE_LIMIT),
        )
        vector_rows = cur.fetchall()

        cur.execute(
            """SELECT id, doc_id, doc_version, section_ref, chunk_text
               FROM rag.doc_chunks
               WHERE tenant_id = %s AND effective_to IS NULL
                 AND chunk_tsv @@ plainto_tsquery('english', %s)
               ORDER BY ts_rank(chunk_tsv, plainto_tsquery('english', %s)) DESC
               LIMIT %s""",
            (tenant, payload.query, payload.query, RETRIEVAL_CANDIDATE_LIMIT),
        )
        text_rows = cur.fetchall()

    fused = _reciprocal_rank_fusion(vector_rows, text_rows)[: payload.top_k]

    return [
        RetrievedChunk(
            chunk_id=str(row["id"]),
            doc_id=str(row["doc_id"]),
            doc_version=row["doc_version"],
            section_ref=row["section_ref"],
            chunk_text=row["chunk_text"],
            score=score,
        )
        for row, score in fused
    ]


# ── Documents ─────────────────────────────────────────────────────────────────

@app.post("/v1/documents", status_code=status.HTTP_202_ACCEPTED, summary="Upload a document for ingestion")
async def upload_document(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None, description="Provide to ingest a new version of an existing document"),
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    tenant = verify_token(token)

    try:
        _extract_ext(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    if doc_id is None:
        doc_id = str(uuid.uuid4())
    else:
        # rag.documents.id is a UUID column — a malformed value would
        # otherwise reach Postgres as a raw string and blow up with a 500
        # (InvalidTextRepresentation) instead of a clean 400.
        try:
            uuid.UUID(doc_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="doc_id is not a valid UUID")

        # A caller can only version a document that already belongs to
        # them — never someone else's doc_id, even if they happen to know it.
        with get_cursor() as cur:
            cur.execute("SELECT tenant_id FROM rag.documents WHERE id = %s", (doc_id,))
            row = cur.fetchone()
        if row is not None and row["tenant_id"] != tenant:
            raise HTTPException(status_code=403, detail="doc_id belongs to a different tenant")

    ingestion_quota, _ = _get_rag_plan_quotas(tenant)
    _check_and_increment_quota(tenant, RAG_INGESTION_QUOTA_PREFIX, "ingestion", increment=1, monthly_limit=ingestion_quota)

    if _celery_app is None:
        raise HTTPException(status_code=500, detail="Celery is not available")

    try:
        task = _celery_app.send_task("ingest_document", kwargs={
            "tenant_id":   tenant,
            "doc_id":      doc_id,
            "filename":    file.filename,
            "file_b64":    base64.b64encode(content).decode("ascii"),
            "ingested_by": tenant,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enqueue ingestion: {e}")

    # Set here (not left to the task) so GET /v1/documents/status/{doc_id}
    # has something to report immediately — the task may not start running
    # for a while if the worker is busy, and "unknown" until then would be
    # indistinguishable from a doc_id that was never enqueued at all.
    set_ingest_status(doc_id, "queued")

    return {
        "doc_id":   doc_id,
        "task_id":  task.id,
        "filename": file.filename,
        "status":   "queued",
    }


@app.get("/v1/documents", summary="List this tenant's documents")
async def list_documents(token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme)):
    tenant = verify_token(token)
    with get_cursor() as cur:
        cur.execute(
            """SELECT id, filename, current_version, uploaded_at
               FROM rag.documents WHERE tenant_id = %s
               ORDER BY uploaded_at DESC""",
            (tenant,),
        )
        return cur.fetchall()


@app.get("/v1/documents/status/{doc_id}", summary="Poll a document's ingestion status")
async def document_status(
    doc_id: str,
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    """status is one of "queued" | "processing" | "complete" | "failed" |
    "unknown" (never enqueued, or its 24h status TTL has expired — same TTL
    app.main's own job-status keys use)."""
    tenant = verify_token(token)

    try:
        uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Document not found")

    with get_cursor() as cur:
        cur.execute("SELECT id FROM rag.documents WHERE id = %s AND tenant_id = %s", (doc_id, tenant))
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Document not found")

    return {"doc_id": doc_id, **get_ingest_status(doc_id)}


@app.delete("/v1/documents/{doc_id}", summary="Soft-delete a document")
async def delete_document(
    doc_id: str,
    token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
):
    """Sets effective_to on every current chunk belonging to this document —
    matches the same versioning mechanism ingest_document uses to supersede
    an old version, rather than a separate deletion concept. The
    rag.documents row itself is left alone (still resolvable by doc_id for
    version history); nothing under this doc_id is selectable by
    /v1/retrieve afterward, since that always filters on effective_to IS NULL."""
    tenant = verify_token(token)

    # rag.documents.id is a UUID column — a malformed path value would
    # otherwise reach Postgres as a raw string and blow up with a 500
    # (InvalidTextRepresentation) instead of a clean 404.
    try:
        uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Document not found")

    with get_cursor(commit=True) as cur:
        cur.execute(
            "SELECT id FROM rag.documents WHERE id = %s AND tenant_id = %s",
            (doc_id, tenant),
        )
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Document not found")

        cur.execute(
            """UPDATE rag.doc_chunks SET effective_to = now()
               WHERE doc_id = %s AND tenant_id = %s AND effective_to IS NULL""",
            (doc_id, tenant),
        )
        chunks_deleted = cur.rowcount

    return {"doc_id": doc_id, "chunks_deleted": chunks_deleted, "status": "deleted"}


@app.get("/v1/usage", summary="This tenant's current-month RAG ingestion/retrieval quota usage")
async def usage(token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme)):
    tenant = verify_token(token)
    from app.db import get_tenant_plan

    plan = get_tenant_plan(tenant)
    ingestion_quota, retrieval_quota = _get_rag_plan_quotas(tenant)

    # Same top-level shape as app.main's /v1/usage (tenant_id, month, plan) —
    # nested per-counter, since there are two quotas here instead of one.
    return {
        "tenant_id": tenant,
        "month": datetime.utcnow().strftime("%Y-%m"),
        "plan": plan,
        "ingestion": {
            "used":  _current_usage(tenant, RAG_INGESTION_QUOTA_PREFIX),
            "limit": ingestion_quota,
        },
        "retrieval": {
            "used":  _current_usage(tenant, RAG_RETRIEVAL_QUOTA_PREFIX),
            "limit": retrieval_quota,
        },
    }


# ── Misc ──────────────────────────────────────────────────────────────────────

@app.get("/health", status_code=status.HTTP_200_OK, summary="Health Check")
async def health_check():
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1")
        db_connected = True
    except Exception:
        db_connected = False

    return {
        "status": "healthy" if db_connected else "degraded",
        "db_connected": db_connected,
    }


@app.get("/v1/whoami", summary="Resolve the calling tenant — auth smoke test")
async def whoami(token: Optional[HTTPAuthorizationCredentials] = Security(security_scheme)):
    tenant_id = verify_token(token)
    return {"tenant_id": tenant_id}
