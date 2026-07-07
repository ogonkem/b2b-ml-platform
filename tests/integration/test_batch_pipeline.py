import httpx, io, csv

BATCH_URL = "http://localhost:8001"
HEADERS   = {"Authorization": "Bearer dev-token"}

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
        f"{BATCH_URL}/batch/upload",
        headers=HEADERS,
        files={"file": ("test.csv", make_csv_bytes(50), "text/csv")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rows_received"] == 50
    assert "job_id" in body
    assert body["status"] == "queued"

def test_batch_upload_bad_csv_returns_400():
    response = httpx.post(
        f"{BATCH_URL}/batch/upload",
        headers=HEADERS,
        files={"file": ("bad.csv", b"not,a,valid\x00csv\xff", "text/csv")}
    )
    assert response.status_code == 400