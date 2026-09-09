"""
agent/db.py
Thin psycopg2 layer for the `agent` schema — no ORM, matching app/db.py's
style. Same Postgres instance as everything else in this stack, in its own
schema (`agent`) so it never touches app.* or rag.*.

agent.agent_decisions is the full audit trace for every POST
/v1/agent/assess call — see agent/graph.py's audit node for what gets
written, and app/main.py's GET /v1/agent/decisions* endpoints for how it's
read back.

Column notes (not self-evident from the DDL alone):
  - application_ref: the application's own identifier (echoed back from
    Selastone's /v1/predict as application_id), not a foreign key into any
    table here — this service doesn't store applications itself.
  - retrieved_chunk_ids: just the UUIDs from rag_service's /v1/retrieve
    response, not the chunk text/scores — the full chunk content already
    lives in rag.doc_chunks; duplicating it here would let the two drift
    out of sync on a re-ingest. Look chunks up by id if the full text is
    needed for a given decision.
  - policy_doc_version: the doc_version of the single highest-ranked
    retrieved chunk (or NULL when policy_aligned is false and nothing was
    retrieved) — a decision can retrieve chunks from more than one
    document/version, so this records the one the decision was most
    grounded in, not an exhaustive list. See agent/graph.py's
    _resolve_policy_doc_version.
"""
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor

# Without this, retrieved_chunk_ids (a uuid[] column) comes back from any
# cursor as the raw Postgres array literal text ("{11111111-...}") instead
# of a parsed list of uuid.UUID — register_uuid() registers the cast
# globally for every connection/cursor in this process, not just one.
psycopg2.extras.register_uuid()


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
        cur.execute("CREATE SCHEMA IF NOT EXISTS agent;")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent.agent_decisions (
                id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id             TEXT NOT NULL,
                application_ref       TEXT NOT NULL,
                risk_score            FLOAT,
                model_version         TEXT,
                rule_engine_version   TEXT,
                decision              TEXT,
                triggered_thresholds  JSONB,
                retrieved_chunk_ids   UUID[],
                policy_doc_version    TEXT,
                policy_aligned        BOOLEAN,
                llm_narrative         TEXT,
                created_at            TIMESTAMPTZ DEFAULT now()
            );
        """)
        # Backs GET /v1/agent/decisions/{application_ref}.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS agent_decisions_application_ref_idx
                ON agent.agent_decisions (application_ref);
        """)
        # Backs GET /v1/agent/decisions?tenant_id=&policy_doc_version=.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS agent_decisions_tenant_policy_version_idx
                ON agent.agent_decisions (tenant_id, policy_doc_version, created_at DESC);
        """)
