# Requires full docker stack running: docker compose up -d
# Run after containers are healthy (allow ~60s for startup).
import io, csv, os
import pytest
import httpx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_URL  = "http://localhost:8000"
_raw      = os.environ.get("API_TOKENS", "dev-token")
API_TOKEN = _raw.split(",")[0].strip()
HEADERS   = {"Authorization": f"Bearer {API_TOKEN}"}


def make_labeled_csv(rows: int = 20) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[
        "ID", "year", "loan_amount", "property_value",
        "income", "Credit_Score", "actual_outcome",
    ])
    writer.writeheader()
    for i in range(rows):
        writer.writerow({
            "ID": i, "year": 2023,
            "loan_amount": 200_000 + i * 1_000,
            "property_value": 300_000,
            "income": 6_000, "Credit_Score": 700,
            "actual_outcome": i % 2,
        })
    return buf.getvalue().encode()


# ── Labeled data upload ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_labeled_data_upload_accepted():
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/labeled-data",
            headers=HEADERS,
            files={"file": ("actuals.csv", make_labeled_csv(20), "text/csv")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"]        == "stored"
    assert body["rows_received"] == 20
    assert "object"              in body
    assert API_TOKEN             in body["object"]

@pytest.mark.asyncio
async def test_labeled_data_without_target_column_rejected():
    bad_csv = b"ID,loan_amount,income\n1,200000,6000\n"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/labeled-data",
            headers=HEADERS,
            files={"file": ("bad.csv", bad_csv, "text/csv")},
        )
    assert response.status_code == 400

@pytest.mark.asyncio
async def test_labeled_data_no_auth_rejected():
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/labeled-data",
            files={"file": ("f.csv", make_labeled_csv(), "text/csv")},
        )
    assert response.status_code == 401


# ── Batch results polling ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_results_unknown_job_returns_404():
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{BASE_URL}/v1/batch/results/nonexistent-job-id",
            headers=HEADERS,
        )
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_batch_upload_then_results_flow():
    """Upload a batch CSV, then immediately poll results — job should be queued."""
    batch_csv = b"ID,year,loan_amount,property_value,income,Credit_Score\n1,2023,200000,300000,6000,720\n"
    async with httpx.AsyncClient(timeout=30.0) as client:
        upload_resp = await client.post(
            f"{BASE_URL}/v1/batch/upload",
            headers=HEADERS,
            files={"file": ("batch.csv", batch_csv, "text/csv")},
        )
        assert upload_resp.status_code == 200
        job_id = upload_resp.json()["job_id"]

        results_resp = await client.get(
            f"{BASE_URL}/v1/batch/results/{job_id}",
            headers=HEADERS,
        )
    assert results_resp.status_code == 200
    body = results_resp.json()
    assert body["job_id"]  == job_id
    assert body["status"]  in ("queued", "processing", "complete")
