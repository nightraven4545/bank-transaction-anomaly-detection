"""FastAPI app: scores bank transactions and serves the dashboard page. Deploys to Vercel as-is."""

from functools import cache
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from detect import FEATURES, REQUIRED, add_features, explain, score

ROOT = Path(__file__).parent
SAMPLE = ROOT / "data" / "bank_transactions_data_2.csv"
MAX_BYTES = 4_000_000  # Vercel rejects request bodies over 4.5 MB anyway
MIN_ROWS, MAX_ROWS = 50, 10_000  # KNN/LOF need neighbours; the cap keeps scoring and responses small
OPTIONAL = ["TransactionID", "Channel"]  # passed through to the table when present

app = FastAPI(title="Bank transaction anomaly detector")


def analyse(df: pd.DataFrame) -> dict:
    """Score every transaction; return compact columns + rows for the dashboard."""
    df = add_features(df)
    columns = list(dict.fromkeys([c for c in OPTIONAL if c in df.columns] + REQUIRED + FEATURES))
    out = df[columns].join(score(df)["ensemble"]).assign(reason=explain(df)).round(4)
    out = out.astype(object).where(out.notna(), None)  # JSON has no NaN
    return {"features": FEATURES, **out.to_dict(orient="split", index=False)}


@cache
def sample_result() -> dict:
    return analyse(pd.read_csv(SAMPLE))


@app.get("/api/sample")
def sample():
    return sample_result()


@app.post("/api/score")
def score_upload(file: UploadFile):
    """Score an uploaded CSV. Uploads are untrusted, so every check runs before any modelling."""
    content = file.file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, f"File too large: the limit is {MAX_BYTES // 1_000_000} MB.")
    try:
        df = pd.read_csv(BytesIO(content))
    except ValueError as e:  # covers parser, empty-file and encoding errors
        raise HTTPException(400, f"Could not read the CSV: {e}") from None

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing required columns: {', '.join(missing)}")
    bad = [c for c in REQUIRED[1:] if not pd.api.types.is_numeric_dtype(df[c]) or not np.isfinite(df[c]).all()]
    if bad:
        raise HTTPException(400, f"These columns must be numeric with no blanks: {', '.join(bad)}")
    if not MIN_ROWS <= len(df) <= MAX_ROWS:
        raise HTTPException(400, f"Need between {MIN_ROWS} and {MAX_ROWS:,} rows, got {len(df):,}.")
    return analyse(df)


# Last, so the API routes above take precedence. On Vercel these files are served from the CDN.
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
