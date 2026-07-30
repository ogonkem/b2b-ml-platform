"""
app/db.py
Thin psycopg2 layer for app.tenants / app.users / app.api_keys — no ORM,
matching this project's existing style of thin clients (redis-py, minio,
clickhouse-connect are all used directly, nowhere else wraps an ORM).

Postgres already runs for Airflow's own metadata (docker-compose.yml); this
uses the same instance, kept in a separate "app" schema so it never touches
Airflow's tables.
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
    """Idempotent — safe to call on every API startup."""
    with get_cursor(commit=True) as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS app;")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS app.tenants (
                id          SERIAL PRIMARY KEY,
                tenant_id   TEXT UNIQUE NOT NULL,
                name        TEXT NOT NULL,
                invite_code TEXT UNIQUE NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS app.users (
                id            SERIAL PRIMARY KEY,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                tenant_id     TEXT NOT NULL REFERENCES app.tenants(tenant_id),
                role          TEXT NOT NULL DEFAULT 'user',
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS app.api_keys (
                id         SERIAL PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES app.users(id),
                key_hash   TEXT UNIQUE NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
