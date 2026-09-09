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
