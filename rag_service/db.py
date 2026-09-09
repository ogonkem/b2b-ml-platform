"""
rag_service/db.py
Thin psycopg2 layer for the `rag` schema — no ORM, matching app/db.py's
style and this project's existing convention of thin clients everywhere.

Shares the same Postgres instance as the main api (same POSTGRES_* env
vars, same database), just in its own schema so it never touches app.* or
Airflow's tables.

Requires the pgvector extension (embedding VECTOR(1536) + the ivfflat
index below) — see docker-compose.yml's postgres service, which runs
pgvector/pgvector:pg15 instead of vanilla postgres:15-alpine for exactly
this reason.
"""
import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor


def _connect():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        dbname=os.environ.get("POSTGRES_DB", "selastone_db"),
    )


@contextmanager
def get_cursor(commit: bool = False):
    """A fresh connection per call — simple and safe at this project's scale,
    no pool to manage or go stale."""
    conn = _connect()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        yield cur
        if commit:
            conn.commit()
    finally:
        conn.close()


def init_schema():
    """Idempotent — safe to call on every startup, same CREATE TABLE IF NOT
    EXISTS / ALTER TABLE ... ADD COLUMN IF NOT EXISTS style as app.db's."""
    with get_cursor(commit=True) as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS rag;")

        # Needed before anything below touches the vector type or ivfflat —
        # must run ahead of the CREATE TABLE that declares an embedding
        # VECTOR(1536) column.
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # One row per uploaded document — doc_chunks.doc_id references this.
        # Tracks whichever version is current; individual chunk rows carry
        # their own doc_version so old chunks stay attributable after a
        # re-ingest bumps current_version.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.documents (
                id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id       TEXT NOT NULL,
                filename        TEXT NOT NULL,
                current_version TEXT NOT NULL,
                uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.doc_chunks (
                id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id          TEXT NOT NULL,
                doc_id             UUID NOT NULL REFERENCES rag.documents(id),
                doc_version        TEXT NOT NULL,
                section_ref        TEXT,               -- nullable, populated when structure detected
                chunk_index        INT NOT NULL,        -- always populated, fallback ordering
                chunk_text         TEXT NOT NULL,
                chunk_hash         TEXT NOT NULL,        -- sha256 of chunk_text, for re-ingestion diffing
                embedding          VECTOR(1536),
                effective_from     TIMESTAMPTZ DEFAULT now(),
                effective_to       TIMESTAMPTZ,          -- NULL = current version
                minio_object_path  TEXT NOT NULL,
                ingested_at        TIMESTAMPTZ DEFAULT now(),
                ingested_by        TEXT
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS doc_chunks_embedding_ivfflat_idx
                ON rag.doc_chunks USING ivfflat (embedding vector_cosine_ops);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS doc_chunks_tenant_doc_effective_idx
                ON rag.doc_chunks (tenant_id, doc_id, effective_to);
        """)

        # Added after doc_chunks already existed, same as app.db's
        # ALTER TABLE ... ADD COLUMN IF NOT EXISTS columns — a generated
        # column backing full-text search alongside the vector search above.
        cur.execute("""
            ALTER TABLE rag.doc_chunks ADD COLUMN IF NOT EXISTS chunk_tsv tsvector
                GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED;
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS doc_chunks_chunk_tsv_gin_idx
                ON rag.doc_chunks USING GIN (chunk_tsv);
        """)

        # Tenant-supplied ground truth for a document, uploaded alongside the
        # file itself (POST /v1/documents' optional eval_set field) — hash-
        # diffed and versioned the same way doc_chunks are (see
        # rag_service/ingest_task.py's chunk_hash diffing), just one level up.
        # doc_id is deliberately NOT a foreign key: an eval_set can be
        # inserted for a brand-new doc_id before rag.documents' own row for
        # it exists yet (that row is created inside the ingest Celery task,
        # which runs after this table is written to during upload).
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.eval_sets (
                id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id    TEXT NOT NULL,
                doc_id       UUID NOT NULL,
                version      TEXT NOT NULL,
                eval_hash    TEXT NOT NULL,
                queries      JSONB NOT NULL,
                uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS eval_sets_doc_id_idx
                ON rag.eval_sets (doc_id, uploaded_at DESC);
        """)

        # One row per eval run (post-ingestion, or from the Re-eval button) —
        # a quality report, not authoritative state, so also no FK to
        # rag.documents for the same reason as eval_sets above.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.eval_runs (
                id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id         TEXT NOT NULL,
                doc_id            UUID NOT NULL,
                doc_version       TEXT,
                eval_set_version  TEXT,
                status            TEXT NOT NULL,
                metrics           JSONB,
                lever_snapshot    JSONB,
                error             TEXT,
                created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS eval_runs_doc_id_idx
                ON rag.eval_runs (doc_id, created_at DESC);
        """)

        # Per-tenant overrides for the pipeline's tunable levers — every
        # column NULL means "use this service's compiled-in default"
        # (rag_service/ingest_task.py's MAX_CHUNK_TOKENS etc., rag_service/
        # main.py's RRF_K/RETRIEVAL_CANDIDATE_LIMIT). A row only needs to
        # exist once a human has actually adjusted something via the
        # Re-eval button's lever inputs.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.tenant_pipeline_config (
                tenant_id              TEXT PRIMARY KEY,
                max_chunk_tokens       INT,
                min_chunk_tokens       INT,
                min_structural_chunks  INT,
                rrf_k                  INT,
                candidate_limit        INT,
                updated_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_by             TEXT
            );
        """)


def get_tenant_pipeline_config(tenant_id: str, defaults: dict) -> dict:
    """Merges `defaults` (the caller's own compiled-in constants) with
    whatever non-NULL overrides this tenant has set. Degrades to `defaults`
    on ANY failure (missing table on a not-yet-migrated install, a DB
    hiccup, etc.) rather than raising — this is a tuning-knob lookup for an
    optional quality-of-life feature, never something that should be able
    to break retrieval itself, same philosophy as app/main.py's SHAP
    computation degrading to an empty list on failure instead of 500ing the
    whole prediction."""
    config = dict(defaults)
    try:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM rag.tenant_pipeline_config WHERE tenant_id = %s", (tenant_id,))
            row = cur.fetchone()
        if row:
            for key in defaults:
                if row.get(key) is not None:
                    config[key] = row[key]
    except Exception:
        pass
    return config


def upsert_tenant_pipeline_config(tenant_id: str, updated_by: str, **fields) -> dict:
    """Partial update — only the keys in `fields` with a non-None value are
    written; everything else on the row (or its NULL/default state, if the
    row doesn't exist yet) is left alone."""
    fields = {k: v for k, v in fields.items() if v is not None}
    if not fields:
        return get_tenant_pipeline_config(tenant_id, {})

    columns = ", ".join(fields.keys())
    placeholders = ", ".join(["%s"] * len(fields))
    conflict_updates = ", ".join(f"{k} = EXCLUDED.{k}" for k in fields)

    with get_cursor(commit=True) as cur:
        cur.execute(
            f"""INSERT INTO rag.tenant_pipeline_config (tenant_id, {columns}, updated_at, updated_by)
                VALUES (%s, {placeholders}, now(), %s)
                ON CONFLICT (tenant_id) DO UPDATE SET {conflict_updates}, updated_at = now(), updated_by = EXCLUDED.updated_by""",
            (tenant_id, *fields.values(), updated_by),
        )
        cur.execute("SELECT * FROM rag.tenant_pipeline_config WHERE tenant_id = %s", (tenant_id,))
        return cur.fetchone()
