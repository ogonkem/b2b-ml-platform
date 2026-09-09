"""
rag_service/ingest_task.py
Celery task for the RAG document-ingestion pipeline. Registered on the same
Celery app as celery_worker/tasks.py (same Redis broker/backend — see
celery_worker/celery_app.py's `include` list) rather than a separate Celery
instance, so it runs on the existing celery_worker container/queue.

Pipeline (see ingest_document below for the orchestration):
  1. Decode the uploaded file and upload the original to MinIO
     (doc-chunks-raw) at {tenant_id}/{doc_id}/{version}/original.{ext}
  2. Extract text: pdfplumber (PDF), python-docx (DOCX), passthrough (.md/.txt)
  3. Structure-aware chunk: detect section headers (numbered clauses,
     markdown headers, "SECTION N" style); fall back to paragraph splitting
     when a multi-page doc yields too few structural boundaries to trust
  4. Enforce a 400-token max / 50-token min chunk-size band
  5. sha256 each final chunk's text
  6. Diff against the doc_id's previous version by chunk_hash — only embed
     what's new or changed, reuse the previous embedding otherwise
  7. Supersede the previous version's rows and insert the new version's rows
  8. If the tenant uploaded an eval_set alongside this doc, score it against
     real retrieval and write a rag.eval_runs report (see run_eval below) —
     best-effort, never fails the ingestion itself

reevaluate_document is the Re-eval button's task: given a tenant's just-
adjusted lever inputs (already persisted to rag.tenant_pipeline_config by
the endpoint that enqueues this), it re-runs steps 2-8 above only if a
chunking-time lever changed, or just step 8 alone if only a query-time
lever (RRF_K, candidate_limit) changed — see rag_service/main.py's
POST /v1/documents/{doc_id}/eval.

Embedding: OpenAI text-embedding-3-small (1536-dim). Chosen specifically to
match rag.doc_chunks.embedding's VECTOR(1536) column exactly, with no
dimension-padding or truncation hacks. The tradeoff: unlike the rest of
this stack (postgres/redis/minio/clickhouse/mlflow/airflow all run locally
in docker-compose, per CLAUDE.md's "no cloud dependency" design), this adds
a network dependency on OpenAI's API, a per-token cost, and requires
OPENAI_API_KEY in .env — ingestion fails closed if that call fails or the
key is missing. This mirrors the one other external-API dependency already
in this codebase (Paystack billing, app/payments.py), so it isn't
unprecedented, but it is a real deviation from "runs entirely on one
machine." A local sentence-transformers model would avoid all of that, but
no commonly used pretrained model emits 1536-dim vectors natively, so using
one would mean either padding vectors to 1536 (wastes space and distorts
cosine similarity) or resizing the column to whatever dimension that model
produces (a separate, unrelated schema change). Swap _embed_batch() for a
local model if the network dependency becomes a problem — nothing else in
the pipeline needs to change.
"""
import base64
import hashlib
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from celery_worker.celery_app import celery_app
from rag_service.db import get_cursor, get_tenant_pipeline_config

DOC_CHUNKS_BUCKET = "doc-chunks-raw"
_SUPPORTED_EXTS   = ("pdf", "docx", "md", "txt")

EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1536
EMBED_BATCH_SIZE  = 100

MAX_CHUNK_TOKENS  = 400
MIN_CHUNK_TOKENS  = 50
# No literal page count is available for docx/md/txt (pagination there is a
# rendering concern, not a property of the raw text) — word count is a rough
# stand-in for "more than about one page" in that case. PDFs use their real
# page count instead (see _extract_text).
MULTI_PAGE_WORD_THRESHOLD = 400
MIN_STRUCTURAL_CHUNKS     = 3

# Ingestion status polling — same pattern as app/main.py's batch-job status
# (a Redis key set by whoever enqueues, updated by the task itself as it
# progresses), not Celery's own AsyncResult: the "job identity" the caller
# polls on (doc_id) is a business-level id passed as a task argument, not
# Celery's internal task_id, matching how batch_upload's job_id already
# works. 24h TTL, same as app/main.py's job-status keys.
INGEST_STATUS_KEY_PREFIX = "ingest_status"
INGEST_STATUS_TTL_SECONDS = 60 * 60 * 24

# Same status-polling pattern, one level up: a re-eval (triggered by
# ingest_document automatically, or reevaluate_document via the frontend's
# Re-eval button) reports its own status independently of ingestion status,
# since the two can now finish at different times (ingest, then eval).
EVAL_STATUS_KEY_PREFIX = "eval_status"
EVAL_STATUS_TTL_SECONDS = 60 * 60 * 24

# Top-k used when scoring eval queries against real retrieval — matches
# agent/graph.py's TOP_K_RETRIEVAL, since that's the "does this query
# actually surface the right chunk in practice" scenario an eval is meant
# to approximate.
EVAL_TOP_K = 5

_minio_client = None
_openai_client = None
_encoding = None
_redis_client = None


def _redis():
    global _redis_client
    if _redis_client is None:
        import redis
        _redis_client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            decode_responses=True,
        )
    return _redis_client


def set_ingest_status(doc_id: str, status: str, error: Optional[str] = None) -> None:
    r = _redis()
    r.set(f"{INGEST_STATUS_KEY_PREFIX}:{doc_id}", status, ex=INGEST_STATUS_TTL_SECONDS)
    if error is not None:
        r.set(f"{INGEST_STATUS_KEY_PREFIX}:{doc_id}:error", error, ex=INGEST_STATUS_TTL_SECONDS)


def get_ingest_status(doc_id: str) -> dict:
    r = _redis()
    status = r.get(f"{INGEST_STATUS_KEY_PREFIX}:{doc_id}")
    error = r.get(f"{INGEST_STATUS_KEY_PREFIX}:{doc_id}:error")
    return {"status": status or "unknown", "error": error}


def set_eval_status(doc_id: str, status: str, error: Optional[str] = None) -> None:
    r = _redis()
    r.set(f"{EVAL_STATUS_KEY_PREFIX}:{doc_id}", status, ex=EVAL_STATUS_TTL_SECONDS)
    if error is not None:
        r.set(f"{EVAL_STATUS_KEY_PREFIX}:{doc_id}:error", error, ex=EVAL_STATUS_TTL_SECONDS)


def get_eval_status(doc_id: str) -> dict:
    r = _redis()
    status = r.get(f"{EVAL_STATUS_KEY_PREFIX}:{doc_id}")
    error = r.get(f"{EVAL_STATUS_KEY_PREFIX}:{doc_id}:error")
    return {"status": status or "unknown", "error": error}


def _minio():
    global _minio_client
    if _minio_client is None:
        from minio import Minio
        _minio_client = Minio(
            f"{os.environ.get('MINIO_HOST', 'localhost')}:{os.environ.get('MINIO_PORT', 9000)}",
            access_key=os.environ.get("MINIO_ROOT_USER"),
            secret_key=os.environ.get("MINIO_ROOT_PASSWORD"),
            secure=False,
        )
    return _minio_client


def _ensure_bucket(client, bucket: str):
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def _openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def _tiktoken_encoding():
    global _encoding
    if _encoding is None:
        import tiktoken
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


# ── Chunk model ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    section_ref: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_hash: Optional[str] = None
    embedding_literal: Optional[str] = None


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_ext(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported file type: {filename!r} — expected one of {_SUPPORTED_EXTS}"
        )
    return ext


def _extract_text(file_bytes: bytes, ext: str) -> tuple[str, Optional[int]]:
    """Returns (text, page_count). page_count is only meaningful for PDFs —
    None for docx/md/txt, where the raw file has no fixed page count."""
    if ext == "pdf":
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n\n".join(pages), len(pages)

    if ext == "docx":
        import docx
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs), None

    # .md / .txt — passthrough
    return file_bytes.decode("utf-8"), None


# ── Structure-aware chunking ──────────────────────────────────────────────────

# Each pattern is checked against one line at a time (not the whole
# document), so they're mutually exclusive by construction: a numbered
# clause must start with a digit or §, a markdown header with #, and
# "SECTION N" with the word SECTION — a given line can only ever match one.
_NUMBERED_CLAUSE_RE = re.compile(r"^\s*§?\s*(\d{1,3}(?:\.\d{1,3}){1,4})\.?\s+\S.*$")
_MARKDOWN_HEADER_RE = re.compile(r"^\s*(#{1,6})\s+(\S.*)$")
_SECTION_WORD_RE    = re.compile(r"^\s*SECTION\s+(\d+)\b[\s:\-\u2013\u2014]*(.*)$", re.IGNORECASE)


def _detect_section_boundaries(lines: list[str]) -> list[tuple[int, str]]:
    """Returns [(line_index, section_ref), ...] for every line that looks
    like a section/clause heading."""
    boundaries = []
    for i, line in enumerate(lines):
        m = _NUMBERED_CLAUSE_RE.match(line)
        if m:
            boundaries.append((i, m.group(1)))
            continue
        m = _MARKDOWN_HEADER_RE.match(line)
        if m:
            boundaries.append((i, m.group(2).strip()))
            continue
        m = _SECTION_WORD_RE.match(line)
        if m:
            label = f"SECTION {m.group(1)}" + (f" {m.group(2).strip()}" if m.group(2).strip() else "")
            boundaries.append((i, label))
    return boundaries


def _fallback_paragraph_chunks(text: str) -> list[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [Chunk(text=p, section_ref=None) for p in paragraphs]


def structure_aware_chunks(
    text: str, *, page_count: Optional[int] = None, min_structural_chunks: Optional[int] = None,
) -> list[Chunk]:
    """Splits `text` on detected section boundaries. Falls back entirely to
    paragraph splitting (section_ref=None throughout) when fewer than
    min_structural_chunks boundaries are found in a multi-page document —
    a handful of stray numbers or a single header in a long, otherwise
    unstructured document isn't a reliable enough signal to chunk on.
    min_structural_chunks defaults to the module constant, overridable per
    tenant via rag.tenant_pipeline_config (see _run_ingestion)."""
    if min_structural_chunks is None:
        min_structural_chunks = MIN_STRUCTURAL_CHUNKS

    lines = text.splitlines()
    boundaries = _detect_section_boundaries(lines)

    word_count = len(text.split())
    is_multi_page = (page_count > 1) if page_count is not None else (word_count > MULTI_PAGE_WORD_THRESHOLD)

    if len(boundaries) < min_structural_chunks and is_multi_page:
        return _fallback_paragraph_chunks(text)

    if not boundaries:
        # Short, single-page-ish doc with no headings — one chunk, no section_ref.
        joined = text.strip()
        return [Chunk(text=joined, section_ref=None)] if joined else []

    chunks = []
    first_line = boundaries[0][0]
    if first_line > 0:
        preamble = "\n".join(lines[:first_line]).strip()
        if preamble:
            chunks.append(Chunk(text=preamble, section_ref=None))

    for idx, (line_no, ref) in enumerate(boundaries):
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        body = "\n".join(lines[line_no:end]).strip()
        if body:
            chunks.append(Chunk(text=body, section_ref=ref))

    return chunks


# ── Token-size enforcement ────────────────────────────────────────────────────

def _token_count(text: str) -> int:
    return len(_tiktoken_encoding().encode(text))


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    """Recursively splits on the coarsest separator that actually breaks the
    text into more than one piece (paragraph, then line, then sentence, then
    word), packing pieces back together up to max_tokens. Falls back to a
    hard token-boundary slice if nothing splits it at all (one long
    unbroken run of text)."""
    if _token_count(text) <= max_tokens:
        return [text]

    for sep in ["\n\n", "\n", ". ", " "]:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        packed, buf = [], ""
        for part in parts:
            candidate = f"{buf}{sep}{part}" if buf else part
            if _token_count(candidate) <= max_tokens:
                buf = candidate
            else:
                if buf:
                    packed.append(buf)
                buf = part
        if buf:
            packed.append(buf)

        result = []
        for piece in packed:
            if _token_count(piece) > max_tokens:
                result.extend(_split_long_text(piece, max_tokens))
            else:
                result.append(piece)
        return result

    tokens = _tiktoken_encoding().encode(text)
    encoding = _tiktoken_encoding()
    return [encoding.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def _enforce_token_bounds(
    chunks: list[Chunk], *, max_chunk_tokens: Optional[int] = None, min_chunk_tokens: Optional[int] = None,
) -> list[Chunk]:
    """max_chunk_tokens/min_chunk_tokens default to the module constants,
    overridable per tenant via rag.tenant_pipeline_config (see
    _run_ingestion)."""
    if max_chunk_tokens is None:
        max_chunk_tokens = MAX_CHUNK_TOKENS
    if min_chunk_tokens is None:
        min_chunk_tokens = MIN_CHUNK_TOKENS

    split_chunks: list[Chunk] = []
    for c in chunks:
        for piece in _split_long_text(c.text, max_chunk_tokens):
            split_chunks.append(Chunk(text=piece, section_ref=c.section_ref))

    # Merge anything under min_chunk_tokens into a neighbor — forward into
    # the next chunk when one exists (the merged chunk keeps the *next*
    # chunk's section_ref, since it ends up holding the larger share of the
    # merged text), otherwise backward into the previous one for a trailing
    # undersized chunk.
    merged: list[Chunk] = []
    i = 0
    while i < len(split_chunks):
        c = split_chunks[i]
        if len(split_chunks) > 1 and _token_count(c.text) < min_chunk_tokens:
            if i + 1 < len(split_chunks):
                nxt = split_chunks[i + 1]
                split_chunks[i + 1] = Chunk(text=f"{c.text}\n\n{nxt.text}", section_ref=nxt.section_ref)
                i += 1
                continue
            if merged:
                prev = merged.pop()
                merged.append(Chunk(text=f"{prev.text}\n\n{c.text}", section_ref=prev.section_ref))
                i += 1
                continue
        merged.append(c)
        i += 1
    return merged


# ── Embedding ─────────────────────────────────────────────────────────────────

def _format_vector(vec: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _embed_batch(texts: list[str], model: Optional[str] = None) -> list[list[float]]:
    if not texts:
        return []
    client = _openai()
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=model or EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in response.data)
    return embeddings


# ── Versioning + diffing ──────────────────────────────────────────────────────

def _next_version(cur, doc_id: str, tenant_id: str, filename: str) -> str:
    cur.execute("SELECT current_version FROM rag.documents WHERE id = %s", (doc_id,))
    row = cur.fetchone()
    if row is None:
        cur.execute(
            "INSERT INTO rag.documents (id, tenant_id, filename, current_version) VALUES (%s, %s, %s, %s)",
            (doc_id, tenant_id, filename, "v1"),
        )
        return "v1"

    m = re.match(r"^v(\d+)$", row["current_version"])
    next_version = f"v{int(m.group(1)) + 1}" if m else "v2"
    cur.execute("UPDATE rag.documents SET current_version = %s WHERE id = %s", (next_version, doc_id))
    return next_version


def _fetch_previous_chunk_embeddings(cur, doc_id: str) -> dict:
    """chunk_hash -> embedding (as the raw pgvector text literal Postgres
    returns) for whichever version of this doc_id is still current — i.e.
    the version this ingest is about to supersede. Reused verbatim for any
    chunk whose hash is unchanged, since it's already a valid `vector`
    literal for the INSERT below."""
    cur.execute(
        "SELECT chunk_hash, embedding FROM rag.doc_chunks WHERE doc_id = %s AND effective_to IS NULL",
        (doc_id,),
    )
    return {row["chunk_hash"]: row["embedding"] for row in cur.fetchall()}


# ── Ingestion (shared by the upload path and the Re-eval button's re-chunk
# path — see reevaluate_document below) ─────────────────────────────────────

def _run_ingestion(tenant_id: str, doc_id: str, filename: str, file_bytes: bytes, ingested_by: Optional[str] = None) -> dict:
    """The actual chunk/embed/diff/supersede pipeline, independent of where
    file_bytes came from (a fresh upload's base64 payload, or the original
    already sitting in MinIO for a re-chunk triggered by a chunking-lever
    adjustment). Reads this tenant's current chunking-lever overrides so a
    Re-eval-triggered re-chunk actually uses whatever was just adjusted."""
    ext = _extract_ext(filename)

    minio = _minio()
    _ensure_bucket(minio, DOC_CHUNKS_BUCKET)

    with get_cursor(commit=True) as cur:
        version = _next_version(cur, doc_id, tenant_id, filename)

    object_path = f"{tenant_id}/{doc_id}/{version}/original.{ext}"
    minio.put_object(
        DOC_CHUNKS_BUCKET, object_path,
        data=io.BytesIO(file_bytes), length=len(file_bytes),
    )

    config = get_tenant_pipeline_config(tenant_id, {
        "max_chunk_tokens":      MAX_CHUNK_TOKENS,
        "min_chunk_tokens":      MIN_CHUNK_TOKENS,
        "min_structural_chunks": MIN_STRUCTURAL_CHUNKS,
    })

    text, page_count = _extract_text(file_bytes, ext)
    chunks = _enforce_token_bounds(
        structure_aware_chunks(text, page_count=page_count, min_structural_chunks=config["min_structural_chunks"]),
        max_chunk_tokens=config["max_chunk_tokens"], min_chunk_tokens=config["min_chunk_tokens"],
    )

    for i, c in enumerate(chunks):
        c.chunk_index = i
        c.chunk_hash  = hashlib.sha256(c.text.encode()).hexdigest()

    with get_cursor() as cur:
        previous = _fetch_previous_chunk_embeddings(cur, doc_id)

    to_embed = [c for c in chunks if c.chunk_hash not in previous]
    embeddings = _embed_batch([c.text for c in to_embed])
    for c, emb in zip(to_embed, embeddings):
        c.embedding_literal = _format_vector(emb)
    for c in chunks:
        if c.chunk_hash in previous:
            c.embedding_literal = previous[c.chunk_hash]

    with get_cursor(commit=True) as cur:
        # Supersede the previous version's rows before inserting the new
        # ones, so this never marks the rows it's about to insert.
        cur.execute(
            "UPDATE rag.doc_chunks SET effective_to = now() WHERE doc_id = %s AND effective_to IS NULL",
            (doc_id,),
        )
        for c in chunks:
            cur.execute(
                """INSERT INTO rag.doc_chunks
                     (tenant_id, doc_id, doc_version, section_ref, chunk_index,
                      chunk_text, chunk_hash, embedding, minio_object_path, ingested_by)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s)""",
                (tenant_id, doc_id, version, c.section_ref, c.chunk_index,
                 c.text, c.chunk_hash, c.embedding_literal, object_path, ingested_by),
            )

    return {
        "doc_id":          doc_id,
        "version":         version,
        "chunk_count":     len(chunks),
        "chunks_embedded": len(to_embed),
        "chunks_reused":   len(chunks) - len(to_embed),
        "minio_object_path": object_path,
    }


# ── Eval (runs after chunking — both right after a normal ingest, if this
# doc has an eval_set, and from the Re-eval button via reevaluate_document
# below) ──────────────────────────────────────────────────────────────────

def _score_eval_query(results: list[dict], expected_text_substring: str, expected_section_ref: Optional[str]) -> Optional[int]:
    """Returns the 1-indexed rank of the first retrieved chunk that matches
    this query's expected answer, or None if nothing in `results` matches.
    A match is either the expected substring appearing in the chunk's text
    (case-insensitive — survives re-chunking, since exact chunk boundaries
    can shift) or, when given, an exact expected_section_ref match."""
    substring = expected_text_substring.lower()
    for rank, r in enumerate(results, start=1):
        if substring in r["chunk_text"].lower():
            return rank
        if expected_section_ref and r.get("section_ref") == expected_section_ref:
            return rank
    return None


def _compute_eval_metrics(per_query: list[dict]) -> dict:
    total = len(per_query)
    hits = [q for q in per_query if q["hit"]]
    return {
        "recall_at_k": (len(hits) / total) if total else 0.0,
        "mrr":         (sum(1.0 / q["rank"] for q in hits) / total) if total else 0.0,
        "total_queries": total,
        "top_k": EVAL_TOP_K,
        "per_query": per_query,
    }


def run_eval(tenant_id: str, doc_id: str) -> dict:
    """Scores this doc's current eval_set (if any) against real retrieval,
    using the tenant's current retrieval-lever config, and bypassing the
    retrieval quota entirely (see rag_service/main.py's _execute_retrieval)
    — this is platform QA, not tenant usage. Never raises: a failure here
    (most commonly the same missing-OPENAI_API_KEY that blocks retrieval
    generally) degrades to a "failed" eval_runs row rather than breaking
    whatever ingestion task called this as its last step."""
    try:
        with get_cursor() as cur:
            cur.execute(
                "SELECT version, queries FROM rag.eval_sets WHERE doc_id = %s ORDER BY uploaded_at DESC LIMIT 1",
                (doc_id,),
            )
            eval_set = cur.fetchone()
    except Exception:
        eval_set = None

    if eval_set is None:
        set_eval_status(doc_id, "no_eval_set")
        return {"status": "no_eval_set"}

    with get_cursor() as cur:
        cur.execute("SELECT current_version FROM rag.documents WHERE id = %s", (doc_id,))
        doc_row = cur.fetchone()
    doc_version = doc_row["current_version"] if doc_row else None

    # Recorded alongside the metrics purely for the report — lets a human
    # see exactly which lever values produced a given recall@k/MRR without
    # cross-referencing rag.tenant_pipeline_config separately.
    lever_snapshot = get_tenant_pipeline_config(tenant_id, {
        "max_chunk_tokens": MAX_CHUNK_TOKENS, "min_chunk_tokens": MIN_CHUNK_TOKENS,
        "min_structural_chunks": MIN_STRUCTURAL_CHUNKS,
        "rrf_k": 60, "candidate_limit": 20,   # mirrors rag_service/main.py's RRF_K/RETRIEVAL_CANDIDATE_LIMIT
    })

    try:
        from rag_service.main import _execute_retrieval   # deferred: main.py imports this module at load time
        per_query = []
        for item in eval_set["queries"]:
            results = _execute_retrieval(tenant_id, item["query"], top_k=EVAL_TOP_K)
            rank = _score_eval_query(results, item["expected_text_substring"], item.get("expected_section_ref"))
            per_query.append({"query": item["query"], "hit": rank is not None, "rank": rank})
    except Exception as e:
        with get_cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO rag.eval_runs (tenant_id, doc_id, doc_version, eval_set_version, status, error, lever_snapshot)
                   VALUES (%s, %s, %s, %s, 'failed', %s, %s::jsonb)""",
                (tenant_id, doc_id, doc_version, eval_set["version"], str(e), json.dumps(lever_snapshot)),
            )
        set_eval_status(doc_id, "failed", error=str(e))
        return {"status": "failed", "error": str(e)}

    metrics = _compute_eval_metrics(per_query)
    with get_cursor(commit=True) as cur:
        cur.execute(
            """INSERT INTO rag.eval_runs (tenant_id, doc_id, doc_version, eval_set_version, status, metrics, lever_snapshot)
               VALUES (%s, %s, %s, %s, 'complete', %s::jsonb, %s::jsonb)""",
            (tenant_id, doc_id, doc_version, eval_set["version"], json.dumps(metrics), json.dumps(lever_snapshot)),
        )
    set_eval_status(doc_id, "complete")
    return {"status": "complete", "metrics": metrics}


# ── Tasks ─────────────────────────────────────────────────────────────────────

@celery_app.task(name="ingest_document", bind=True, max_retries=2)
def ingest_document(
    self,
    tenant_id: str,
    doc_id: str,
    filename: str,
    file_b64: str,
    ingested_by: Optional[str] = None,
):
    """
    file_b64 is the raw file, base64-encoded — task_serializer is "json"
    (celery_worker/celery_app.py), which can't carry raw bytes as a task
    argument directly.
    """
    set_ingest_status(doc_id, "processing")
    try:
        result = _run_ingestion(tenant_id, doc_id, filename, base64.b64decode(file_b64), ingested_by)
    except Exception as e:
        set_ingest_status(doc_id, "failed", error=str(e))
        raise

    set_ingest_status(doc_id, "complete")

    # Best-effort — an eval failure (or simply no eval_set for this doc)
    # must never turn a successful ingest into a reported failure. run_eval
    # itself is exception-safe, but a second layer here costs nothing.
    try:
        run_eval(tenant_id, doc_id)
    except Exception:
        pass

    return result


@celery_app.task(name="reevaluate_document", bind=True, max_retries=1)
def reevaluate_document(self, tenant_id: str, doc_id: str, rechunk: bool = False):
    """Triggered by the frontend's Re-eval button (POST /v1/documents/{doc_id}/eval)
    after any lever-input changes have already been persisted to
    rag.tenant_pipeline_config. Only re-chunks (and therefore re-embeds) when
    a chunking-time lever changed — a query-time-only adjustment (RRF_K,
    candidate_limit) just re-scores the document's current chunks."""
    set_eval_status(doc_id, "processing")
    try:
        if rechunk:
            with get_cursor() as cur:
                cur.execute("SELECT filename, current_version FROM rag.documents WHERE id = %s", (doc_id,))
                doc_row = cur.fetchone()
            if doc_row is None:
                raise ValueError(f"Document {doc_id} not found")

            ext = _extract_ext(doc_row["filename"])
            object_path = f"{tenant_id}/{doc_id}/{doc_row['current_version']}/original.{ext}"
            minio = _minio()
            response = minio.get_object(DOC_CHUNKS_BUCKET, object_path)
            try:
                file_bytes = response.read()
            finally:
                response.close()
                response.release_conn()

            _run_ingestion(tenant_id, doc_id, doc_row["filename"], file_bytes, ingested_by=tenant_id)

        result = run_eval(tenant_id, doc_id)
    except Exception as e:
        set_eval_status(doc_id, "failed", error=str(e))
        raise
    return result
