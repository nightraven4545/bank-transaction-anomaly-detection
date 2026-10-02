from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import MAX_BYTES, app

client = TestClient(app)
SAMPLE = Path(__file__).parent / "data" / "bank_transactions_data_2.csv"
HEADER = "AccountID,TransactionAmount,TransactionDuration,LoginAttempts,AccountBalance\n"


def upload(content):
    return client.post("/api/score", files={"file": ("upload.csv", content, "text/csv")})


def test_sample_scores_every_transaction():
    body = client.get("/api/sample").json()
    assert len(body["data"]) == 2512
    assert {"ensemble", "reason", *body["features"]} <= set(body["columns"])


def test_uploading_the_sample_gives_the_same_result():
    assert upload(SAMPLE.read_bytes()).json() == client.get("/api/sample").json()


def test_minimal_upload_without_optional_columns():
    body = upload(HEADER + "".join(f"A{i % 7},{10 + i},{30 + i},1,{1000 + i}\n" for i in range(60))).json()
    assert len(body["data"]) == 60 and "TransactionID" not in body["columns"]


@pytest.mark.parametrize(
    ("content", "status", "message"),
    [
        ("a,b\n1,2\n", 400, "Missing required columns"),
        (HEADER + "A1,10,50,abc,100\n" * 60, 400, "must be numeric"),
        (HEADER + "A1,10,50,1,100\n" * 3, 400, "Need between"),
        ("", 400, "Could not read"),
        ("x" * (MAX_BYTES + 1), 413, "too large"),
    ],
    ids=["missing-column", "non-numeric", "too-few-rows", "empty-file", "too-large"],
)
def test_bad_uploads_are_rejected_with_a_reason(content, status, message):
    response = upload(content)
    assert response.status_code == status
    assert message in response.json()["detail"]


def test_dashboard_page_is_served():
    response = client.get("/")
    assert response.status_code == 200 and "<title>Transaction Anomaly Detector</title>" in response.text
