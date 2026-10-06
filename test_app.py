import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import MAX_BYTES, analyse, app
from report import OUT, SAMPLES

client = TestClient(app)
HEADER = "AccountID,TransactionAmount,TransactionDuration,LoginAttempts,AccountBalance\n"


def upload(content):
    return client.post("/api/score", files={"file": ("upload.csv", content, "text/csv")})


def published(key):
    return json.loads((OUT / f"{key}.json").read_text())


@pytest.mark.parametrize("key", SAMPLES)
def test_published_samples_match_the_scoring_code(key):
    """static/data/*.json comes from report.py. This fails if detect.py changes and nobody re-runs it."""
    path, recipe = SAMPLES[key]
    body = published(key)
    assert body == analyse(pd.read_csv(path), recipe)
    assert {"iforest", "knn", "lof", "ensemble", "reason", *body["features"]} <= set(body["columns"])


def test_uploading_the_sample_gives_the_published_scores():
    assert upload(SAMPLES["kaggle"][0].read_bytes()).json() == published("kaggle")


def test_report_has_the_blocks_the_page_reads():
    report = published("report")
    assert {"notes", "planted", "ulb"} <= report.keys()
    assert report["ulb"]["frauds"] == 492 and report["notes"]["kaggle"]["rows"] == 2512


def test_minimal_upload_without_optional_columns():
    body = upload(HEADER + "".join(f"A{i % 7},{10 + i},{30 + i},1,{1000 + i}\n" for i in range(60))).json()
    assert len(body["data"]) == 60 and "TransactionID" not in body["columns"]


def test_bank_statement_export_is_scored():
    """HDFC-style export: account details above the table, padded headers, a separator row,
    Indian thousands separators, blank credit cells and a footer."""
    rows = "".join(
        f'{1 + i % 28:02d}/0{4 + i // 28}/25,UPI-SHOP-{i},{1 + i % 28:02d}/0{4 + i // 28}/25,'
        f'"{"1,33,750.00" if i == 7 else f"{100 + i * 10:,}.00"}",,REF{i},"{200000 - i * 500:,}.00"\n'
        for i in range(40)
    )
    statement = (
        "HDFC BANK Ltd.\nStatement of account\nAccount No :,50100XXXXXXXX\n\n"
        "  Date     ,Narration,Value Dat,Debit Amount       ,Credit Amount      ,Chq/Ref Number   ,Closing Balance\n"
        "********,********,********,********,********,********,********\n"
        + rows
        + ",,,,,,\n05/05/25,SALARY CREDIT,05/05/25,,50000.00,REF99,250000.00\n"
        + "STATEMENT SUMMARY :-,,,,,,\n"
    )
    response = upload(statement)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["recipe"] == "ledger" and len(body["data"]) == 40  # the salary credit is not scored
    row = dict(zip(body["columns"], max(body["data"], key=lambda r: r[body["columns"].index("ensemble")])))
    assert row["Debit"] == 133750.0 and row["Narration"] == "UPI-SHOP-7"


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
