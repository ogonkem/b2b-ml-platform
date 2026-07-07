# Requires the full docker stack running: docker compose up -d
import os
import pytest
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Load .env so tokens match exactly what the running container has
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_URL = "http://localhost:8000"

_raw_tokens = os.environ.get("API_TOKENS", "dev-token")
API_TOKEN = _raw_tokens.split(",")[0].strip()
QUOTA_TEST_TOKEN = os.environ.get("QUOTA_TEST_TOKEN", "quota-test-token")

HEADERS       = {"Authorization": f"Bearer {API_TOKEN}"}
QUOTA_HEADERS = {"Authorization": f"Bearer {QUOTA_TEST_TOKEN}"}

print(f"Using token: {API_TOKEN}")   # helpful debug line

@pytest.mark.asyncio
async def test_end_to_end_low_risk():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/predict",
            json={
                "ID": 1,
                "year": 2023,
                "loan_amount": 80000.0,
                "property_value": 350000.0,
                "income": 15000.0,
                "Credit_Score": 850.0,
            },
            headers=HEADERS
        )
    assert response.status_code == 200
    body = response.json()
    # Only assert structure and valid range — not specific values
    assert "default_probability" in body
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["default_prediction"] in (0, 1)
    assert body["status"] == "success"

@pytest.mark.asyncio
async def test_end_to_end_high_risk():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/predict",
            json={
                "ID": 2,
                "year": 2023,
                "loan_amount": 480000.0,
                "property_value": 200000.0,
                "income": 1200.0,
                "Credit_Score": 520.0,
                "dtir1": 85.0,
                "LTV": 95.0,
            },
            headers=HEADERS
        )
    assert response.status_code == 200
    body = response.json()
    assert "default_probability" in body
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["default_prediction"] in (0, 1)
    assert body["status"] == "success"


@pytest.mark.asyncio
async def test_quota_enforced_after_limit():
    async with httpx.AsyncClient() as client:
        for _ in range(1001):
            r = await client.post(
                f"{BASE_URL}/v1/predict",
                json={"ID": 9, "year": 2023,
                      "loan_amount": 1.0, "property_value": 1.0,
                      "income": 1.0, "Credit_Score": 600.0},
                headers=QUOTA_HEADERS
            )
        assert r.status_code == 429, f"Got {r.status_code}: {r.text}"