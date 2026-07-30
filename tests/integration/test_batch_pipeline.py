import io, csv
import httpx
import pytest
from pathlib import Path
from dotenv import dotenv_values

# Read .env directly (not load_dotenv) so tokens match the running container
# without mutating the real process environment — that leaks vars like
# AIRFLOW__DATABASE__SQL_ALCHEMY_CONN into any test that runs later in the
# same pytest session, including unrelated unit tests that import airflow.
_env = dotenv_values(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_URL  = "http://localhost:8000"
_raw      = _env.get("API_TOKENS", "dev-token")
API_TOKEN = _raw.split(",")[0].strip()
HEADERS   = {"Authorization": f"Bearer {API_TOKEN}"}


def make_csv_bytes(n_rows=50):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[
        "ID", "year", "loan_amount", "property_value", "income", "Credit_Score"
    ])
    writer.writeheader()
    for i in range(n_rows):
        writer.writerow({
            "ID": i, "year": 2023,
            "loan_amount": 200000, "property_value": 300000,
            "income": 5000, "Credit_Score": 700
        })
    return buf.getvalue().encode()


def test_batch_upload_accepted():
    response = httpx.post(
        f"{BASE_URL}/v1/batch/upload",
        headers=HEADERS,
        files={"file": ("test.csv", make_csv_bytes(50), "text/csv")},
        timeout=30.0,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rows_received"] == 50
    assert "job_id" in body
    assert body["status"] == "queued"


def test_batch_upload_bad_csv_returns_400():
    response = httpx.post(
        f"{BASE_URL}/v1/batch/upload",
        headers=HEADERS,
        files={"file": ("bad.csv", b"not,a,valid\x00csv\xff", "text/csv")},
        timeout=30.0,
    )
    assert response.status_code == 400