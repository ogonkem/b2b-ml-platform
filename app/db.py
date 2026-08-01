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
        # Added after app.tenants already existed in deployed databases —
        # ADD COLUMN IF NOT EXISTS instead of a CREATE TABLE, so every
        # pre-existing tenant row picks up the 'free' default silently.
        cur.execute("""
            ALTER TABLE app.tenants ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'free';
        """)
        # Paystack billing state — 'none' until a tenant ever starts a
        # checkout; the webhook (app/main.py) is the only writer of
        # subscription_status besides this default.
        cur.execute("""
            ALTER TABLE app.tenants ADD COLUMN IF NOT EXISTS paystack_customer_code TEXT;
        """)
        cur.execute("""
            ALTER TABLE app.tenants ADD COLUMN IF NOT EXISTS paystack_subscription_code TEXT;
        """)
        cur.execute("""
            ALTER TABLE app.tenants ADD COLUMN IF NOT EXISTS subscription_status TEXT NOT NULL DEFAULT 'none';
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


def get_tenant_plan(tenant_id: str) -> str:
    """The tenant's subscription plan id (see app/plans.py). Falls back to
    the default plan if this tenant has no app.tenants row at all — a
    static API_TOKENS value is a valid tenant_id everywhere else in the
    app but was never registered, so there's no row to read a plan from —
    or if Postgres is unreachable, matching the graceful-degradation style
    already used for ClickHouse elsewhere in this codebase."""
    from app.plans import DEFAULT_PLAN
    try:
        with get_cursor() as cur:
            cur.execute("SELECT plan FROM app.tenants WHERE tenant_id = %s", (tenant_id,))
            row = cur.fetchone()
            return row["plan"] if row else DEFAULT_PLAN
    except Exception:
        return DEFAULT_PLAN


def set_tenant_plan(tenant_id: str, plan: str) -> bool:
    """Returns whether a row was actually updated — False means this
    tenant_id has no app.tenants row (e.g. a static API_TOKENS value),
    which the caller should turn into a clear rejection rather than a
    silent no-op."""
    with get_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE app.tenants SET plan = %s WHERE tenant_id = %s",
            (plan, tenant_id),
        )
        return cur.rowcount > 0


def get_tenant_id_by_email(email: str) -> str | None:
    """Resolves a tenant_id from a user's email — the Paystack webhook only
    carries the paying customer's email, not our tenant_id, so this is how
    it maps a payment event back to the right tenant."""
    with get_cursor() as cur:
        cur.execute("SELECT tenant_id FROM app.users WHERE email = %s", (email,))
        row = cur.fetchone()
        return row["tenant_id"] if row else None


def get_tenant_row(tenant_id: str) -> dict | None:
    """Full app.tenants row, including Paystack billing state — unlike
    get_tenant_plan() this returns None (not a default) when the tenant
    has no row, since callers here need to distinguish "never subscribed"
    from "on the free plan"."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM app.tenants WHERE tenant_id = %s", (tenant_id,))
        return cur.fetchone()


def set_tenant_subscription(
    tenant_id: str, plan: str, customer_code: str, subscription_code: str, status: str
) -> bool:
    """The Paystack webhook's write path — updates plan and billing state
    together so they never disagree (e.g. plan says 'pro' but
    subscription_status says 'canceled')."""
    with get_cursor(commit=True) as cur:
        cur.execute(
            """UPDATE app.tenants
               SET plan = %s, paystack_customer_code = %s,
                   paystack_subscription_code = %s, subscription_status = %s
               WHERE tenant_id = %s""",
            (plan, customer_code, subscription_code, status, tenant_id),
        )
        return cur.rowcount > 0
