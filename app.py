"""FastAPI app: scores uploaded transactions and serves the dashboard page. Deploys to Vercel as-is.

The bundled datasets are scored ahead of time by report.py into static/data/*.json (served from the CDN),
so the only work done here is scoring uploads.
"""

import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from detect import RECIPES, REQUIRED, explain, score

ROOT = Path(__file__).parent
MAX_BYTES = 4_000_000  # Vercel rejects request bodies over 4.5 MB anyway
MIN_ROWS, MAX_ROWS = 30, 10_000  # LOF needs 20 neighbours; the cap keeps scoring and responses small
# Shown in the table and case view when present. Never used as features.
CONTEXT = ["TransactionID", "TransactionDate", "AccountID", "Agency", "Vendor", "Category", "Narration",
           "TransactionType", "Channel", "Location", "AccountBalance", "Balance"]
# Bank-statement exports (HDFC, SBI, ICICI and most others): normalised header name -> our column.
# First match wins, so a statement with both "Date" and "Value Date" uses "Date".
STATEMENT = {
    "TransactionDate": ["date", "txndate", "transactiondate", "valuedate", "valuedt"],
    "Narration": ["narration", "description", "transactionremarks", "particulars", "remarks"],
    "Debit": ["debit", "debitamount", "withdrawalamt", "withdrawalamount", "withdrawalamountinr", "withdrawal"],
    "Balance": ["balance", "closingbalance", "balanceinr"],
}

app = FastAPI(title="Transaction anomaly detector")


def analyse(df: pd.DataFrame, recipe: str = "bank") -> dict:
    """Score every transaction; return compact columns + rows for the dashboard."""
    make, features = RECIPES[recipe]
    df = make(df)
    columns = list(dict.fromkeys([c for c in CONTEXT if c in df.columns] + features))
    out = df[columns].join(score(df, features)).assign(reason=explain(df, features))
    for c in out.select_dtypes("datetime"):
        out[c] = out[c].dt.strftime("%Y-%m-%d")
    out = out.round(4)
    out = out.astype(object).where(out.notna(), None)  # JSON has no NaN
    return {"recipe": recipe, "features": features, **out.to_dict(orient="split", index=False)}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def parse_upload(content: bytes) -> tuple[pd.DataFrame, str]:
    """Read either the bank schema (REQUIRED columns) or a bank-statement export. Raises ValueError with a reason."""
    text = content.decode("utf-8-sig")
    bank = {norm(c) for c in REQUIRED}
    # Bank exports put account details above the table, so look for the header row in the first 30 lines.
    # ponytail: splits the header on commas; a quoted header containing a comma would be missed.
    for i, line in enumerate(text.splitlines()[:30]):
        names = {norm(c) for c in line.split(",")}
        if bank <= names:
            return pd.read_csv(StringIO(text), skiprows=i), "bank"
        if all(names & set(STATEMENT[f]) for f in ["TransactionDate", "Debit", "Balance"]):
            raw = pd.read_csv(StringIO(text), skiprows=i, skipinitialspace=True, dtype=str)
            by_name = {}
            for col in raw.columns:
                by_name.setdefault(norm(col), col)
            picked = {f: next((by_name[a] for a in aliases if a in by_name), None) for f, aliases in STATEMENT.items()}
            df = pd.DataFrame({f: raw[c].str.strip() for f, c in picked.items() if c is not None})
            # Junk rows (asterisk separators, footer totals) have no valid date: drop them.
            df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["TransactionDate"])
            for c in ["Debit", "Balance"]:
                df[c] = pd.to_numeric(df[c].str.replace(",", ""), errors="coerce")
            df["Debit"] = df["Debit"].fillna(0)  # blank debit cell = a credit row
            return df.assign(AccountID="statement"), "ledger"
    raise ValueError(
        f"Missing required columns: {', '.join(REQUIRED)}. "
        "Or upload a bank statement with Date, Debit (or Withdrawal) and Balance columns."
    )


@app.post("/api/score")
def score_upload(file: UploadFile):
    """Score an uploaded CSV. Uploads are untrusted, so every check runs before any modelling. Nothing is stored."""
    content = file.file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, f"File too large: the limit is {MAX_BYTES // 1_000_000} MB.")
    if not content.strip():
        raise HTTPException(400, "Could not read the CSV: the file is empty.")
    try:
        df, recipe = parse_upload(content)
    except ValueError as e:  # covers parser, encoding and missing-column errors
        message = str(e) if str(e).startswith("Missing required") else f"Could not read the CSV: {e}"
        raise HTTPException(400, message) from None

    numeric = REQUIRED[1:] if recipe == "bank" else ["Debit", "Balance"]
    bad = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c]) or not np.isfinite(df[c]).all()]
    if bad:
        raise HTTPException(400, f"These columns must be numeric with no blanks: {', '.join(bad)}")
    rows = len(df) if recipe == "bank" else int((df["Debit"] > 0).sum())
    if not MIN_ROWS <= rows <= MAX_ROWS:
        kind = "rows" if recipe == "bank" else "withdrawals"
        raise HTTPException(400, f"Need between {MIN_ROWS} and {MAX_ROWS:,} {kind}, got {rows:,}.")
    return analyse(df, recipe)


# Last, so the API route above takes precedence. On Vercel these files are served from the CDN.
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
