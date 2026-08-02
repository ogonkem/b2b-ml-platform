# CLAUDE.md

Operational guidance for working in this repo. For architecture and design
rationale, read `README.md` first — it's thorough and already explains the
"why" behind the stack, the feature pipeline, drift detection, etc. This
file is about how to work here day to day: commands, conventions, and
gotchas that aren't obvious from reading the code once.

## What this is

Selastone — a loan-default risk-scoring platform. Docker Compose stack:
`postgres`, `redis`, `minio`, `clickhouse`, `mlflow`, `airflow`, `api`
(FastAPI), `celery_worker`, `frontend` (React/Vite, nginx-served),
`prometheus`, `grafana`. Runs entirely on one machine — no cloud
dependency, `.env` holds every credential (gitignored; `.env.example` is
the template with blanks).

## Core convention: thin clients, no ORM

`app/`, `celery_worker/`, `airflow_dags/` all talk to Postgres/Redis/
MinIO/ClickHouse directly (`psycopg2`, `redis-py`, `minio`,
`clickhouse-connect`) — nowhere in this codebase wraps one of these in an
ORM or SDK abstraction. New code should match that, not introduce one.

Postgres schema changes (`app/db.py`'s `init_schema()`) are idempotent
`CREATE TABLE IF NOT EXISTS` / `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
run at every API startup — there's no migration tool, and there shouldn't
be one added for this project's scale. Follow that pattern for new
columns/tables rather than introducing Alembic or similar.

## Rebuilding after a change

Backend (`app/`, `celery_worker/`, `requirements.txt`):
```bash
docker compose build api celery_worker
docker compose up -d --force-recreate api celery_worker
```

Frontend (`frontend/`):
```bash
cd frontend && npm run build      # typecheck + build — do this first, always
docker compose build frontend && docker compose up -d --force-recreate frontend
```
Or run `npm run dev` for the Vite dev server on `:5173` instead of rebuilding
the Docker image on every change — both talk to the same API on `:8000`.

## Tests

- `pytest tests/unit` — fully mocked, no live stack needed, this is what CI
  runs. Every test file that imports `app.main` must mock `redis.Redis`,
  `clickhouse_connect.get_client`, and `psycopg2.connect` **before** the
  import, because `app.main` calls `init_schema()` at import time — skip
  this and the test either hangs trying to reach a real Postgres or fails
  outright. Copy the mocking boilerplate from any existing file in
  `tests/unit/` (e.g. `test_plans.py`, `test_payments.py`) rather than
  reinventing it.
- `pytest tests/integration` and `lifecycle_tests/` need the full stack up
  (`docker compose up -d`) and hit real services. Not run in CI.
- The shared static test token (`API_TOKENS` in `.env`, tenant defaults to
  the `free` plan at 100 predictions/month) gets used by both integration
  tests and any manual smoke-testing against the live stack — it's easy to
  exhaust its monthly quota this way, which shows up as integration test
  failures (429) unrelated to whatever you were actually testing. Check
  `GET /v1/usage` with that token if a batch/predict test fails
  mysteriously; reset with `redis-cli DEL quota:<token>:<YYYY_MM>` if so.

## Auth model

`verify_token()` in `app/main.py` accepts three kinds of bearer value,
all resolving to the same `tenant_id`: a static `API_TOKENS` entry, a JWT
from `/auth/login`/`/auth/register`, or an `sk_...` API key from
`/auth/api-key`. A static token has no `app.tenants` row — plan/billing
endpoints that need a real row (plan switching, checkout) reject it with a
clear message rather than silently no-op'ing.

## Subscription tiers & Paystack billing

Tier catalog is `app/plans.py` (code, not a DB table) — `free`/`starter`/
`pro` are self-serve or checkout-based, `enterprise` has no fixed price
and is sales-assisted, never sold through checkout.

Paystack gotchas discovered the hard way — don't re-assume the opposite:
- `POST /transaction/initialize` requires an explicit `amount` **even when
  a `plan` code is given** — it does not infer the charge from the plan.
- Attaching a `plan` to that same call **does not create a recurring
  subscription** — it only charges once. The actual subscription has to be
  created explicitly via `POST /subscription` using the card authorization
  from the resulting `charge.success` webhook (see `app/payments.py`'s
  `create_subscription()` and the webhook handler in `app/main.py`).
- Paystack overwrites `callback_url`'s query string with its own
  `trxref`/`reference` params on redirect — don't rely on a custom query
  param surviving that round trip (`frontend/src/pages/Plans.tsx` detects
  return-from-checkout via Paystack's own params instead).
- The **webhook URL** (Paystack calling us) and the **callback URL**
  (browser redirect after checkout) are different settings — only the
  webhook needs a public tunnel (`ngrok http 8000`) for local testing; the
  callback URL is just the user's own browser navigating, so `localhost`
  works fine there.
- Checkout's `callback_url` redirects to the *first* entry in
  `FRONTEND_ORIGINS`. If you started checkout from a different origin
  (Vite `:5173` vs. the Docker/nginx build on `:3001`), you land back on a
  different origin than you logged in on — `localStorage` is per-origin,
  so the app looks logged out there even though billing processed
  correctly. Navigate back to the origin you actually logged in on.

## Windows-specific gotchas

- Python scripts that print emoji (`⚠️`, `✓`) crash on Windows with a
  `UnicodeEncodeError` under the default `cp1252` console encoding unless
  run with `PYTHONIOENCODING=utf-8` set first.
- The Bash tool's `/tmp` resolves outside the repo on Windows/git-bash
  (MSYS path translation) — write scratch scripts/files inside the repo
  (e.g. a throwaway file in the repo root) instead, and delete them when
  done rather than relying on `/tmp`.
- `MINIO_HOST` (`minio`, the Docker-internal hostname) and
  `MINIO_PUBLIC_ENDPOINT` (`localhost:9000`) are deliberately different —
  the former is for server-to-server calls inside the compose network, the
  latter is for presigned URLs handed to the browser. Don't collapse them
  back into one variable; that was a real bug this project already hit
  once (browser couldn't resolve `minio:9000`).
