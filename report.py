"""Builds everything the site shows, offline.

1. Fetches the two public datasets once (DC purchase cards, Czech bank sample) into data/.
2. Scores every bundled dataset into static/data/{dc,czech,kaggle}.json (the same output as the upload API),
   so the page loads from the CDN instead of waiting for a cold Python function.
3. Measures the detectors into static/data/report.json: real fraud labels (ULB card data, downloaded
   from OpenML, never committed) and planted frauds (Kaggle file, 20 runs).

Run `python report.py`. The first run downloads about 220 MB and takes a few minutes. Importing it runs nothing,
so notebook.ipynb can reuse plant_anomalies / evaluate / to_probability.
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf
from sklearn.metrics import average_precision_score, roc_auc_score

from detect import add_features, flag, score

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "static" / "data"
KAGGLE, DC, CZECH = DATA / "bank_transactions_data_2.csv", DATA / "dc_pcard.csv", DATA / "czech_bank_sample.csv"
SAMPLES = {"dc": (DC, "pcard"), "czech": (CZECH, "ledger"), "kaggle": (KAGGLE, "bank")}  # key -> (csv, recipe)
KINDS = ["account takeover", "big-ticket", "balance drain"]
RATES = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]  # review capacity, share of rows

DC_API = "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Public_Service_WebMercator/MapServer/50/query"
BERKA = "https://raw.githubusercontent.com/jlacko/berka-dataset/master/"
OPERATION = {"VYBER KARTOU": "card withdrawal", "VKLAD": "cash deposit", "PREVOD Z UCTU": "transfer in",
             "VYBER": "cash withdrawal", "PREVOD NA UCET": "transfer out"}
PURPOSE = {"POJISTNE": "insurance", "SLUZBY": "statement fee", "UROK": "interest", "SANKC. UROK": "overdraft interest",
           "SIPO": "household bill", "DUCHOD": "pension", "UVER": "loan payment"}


def fetch_dc(n: int = 10_000) -> pd.DataFrame:
    """Newest n Washington DC purchase-card payments (opendata.dc.gov, CC BY 4.0), oldest first."""
    frames, kept = [], 0
    while kept < n:
        query = urllib.parse.urlencode({
            "where": "TRANSACTION_AMOUNT > 0",  # refunds and zero rows are not purchases
            "outFields": "AGENCY,TRANSACTION_DATE,TRANSACTION_AMOUNT,VENDOR_NAME,MCC_DESCRIPTION",
            "orderByFields": "TRANSACTION_DATE DESC,OBJECTID DESC",  # OBJECTID breaks date ties, so pages never overlap
            "resultOffset": 1000 * len(frames), "resultRecordCount": 1000, "f": "json",
        })
        with urllib.request.urlopen(f"{DC_API}?{query}", timeout=60) as response:
            features = json.load(response)["features"]
        if not features:
            break
        page = pd.DataFrame([f["attributes"] for f in features])
        # Dispute re-bills carry unreliable vendor/category; "Unknown New Agency" is a placeholder, not an agency.
        page = page[(page["VENDOR_NAME"] != "DISPUTE REBILL") & (page["AGENCY"] != "Unknown New Agency")]
        frames.append(page)
        kept += len(page)
    df = pd.concat(frames).head(n)
    dates = pd.to_datetime(df["TRANSACTION_DATE"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    return pd.DataFrame({
        "TransactionDate": dates.dt.strftime("%Y-%m-%d"),
        "Agency": df["AGENCY"].str.split().str.join(" "),  # the source pads some names with double spaces
        "Vendor": df["VENDOR_NAME"].str.split().str.join(" "),
        "Category": df["MCC_DESCRIPTION"].str.split().str.join(" "),
        "TransactionAmount": df["TRANSACTION_AMOUNT"],
    }).iloc[::-1]


def fetch_czech(n_accounts: int = 40, seed: int = 0) -> pd.DataFrame:
    """Full histories of n random accounts from the PKDD'99 Czech bank ledger, in the bank-statement schema."""
    trans = pd.read_csv(BERKA + "trans.asc", sep=";", dtype={"date": str}, low_memory=False)
    accounts = np.random.default_rng(seed).choice(trans["account_id"].unique(), n_accounts, replace=False)
    t = trans[trans["account_id"].isin(accounts)].sort_values(["account_id", "date", "trans_id"])
    purpose = t["k_symbol"].str.strip().map(PURPOSE)
    operation = t["operation"].str.strip().map(OPERATION)
    return pd.DataFrame({
        "AccountID": t["account_id"],
        "TransactionDate": pd.to_datetime(t["date"], format="%y%m%d").dt.strftime("%Y-%m-%d"),
        "Narration": (operation + " · " + purpose).fillna(operation).fillna(purpose).fillna("other"),
        "Debit": t["amount"].where(t["type"] != "PRIJEM", 0),  # PRIJEM = money in; VYDAJ and VYBER = money out
        "Balance": t["balance"],
    })


def to_probability(scores) -> np.ndarray:
    """PyOD's 'unify' scaling: standardise, then erf -> outlier probability in 0..1."""
    scores = np.asarray(scores, dtype=float)
    return erf((scores - scores.mean()) / (scores.std() * np.sqrt(2))).clip(0, 1)


def amount_rule(amount: pd.Series) -> pd.Series:
    """The baseline a bank might start with: how far the amount sits from the median, in MADs."""
    dev = (amount - amount.median()).abs()
    return dev / dev.median()


def plant_anomalies(raw: pd.DataFrame, seed: int, n_each: int = 9) -> tuple[pd.DataFrame, pd.Series]:
    """Copy random real transactions, turn them into the three fraud patterns, mix them back in."""
    rng = np.random.default_rng(seed)
    account_median = raw.groupby("AccountID")["TransactionAmount"].median()
    p = raw.sample(3 * n_each, random_state=seed).reset_index(drop=True)
    p["kind"] = np.repeat(KINDS, n_each)

    t = p["kind"] == "account takeover"
    p.loc[t, "LoginAttempts"] = rng.integers(4, 6, t.sum())
    p.loc[t, "Channel"] = "Online"
    p.loc[t, "TransactionAmount"] *= rng.uniform(3, 6, t.sum())

    b = p["kind"] == "big-ticket"
    p.loc[b, "TransactionAmount"] = p.loc[b, "AccountID"].map(account_median).to_numpy() * rng.uniform(5, 10, b.sum())

    d = p["kind"] == "balance drain"
    available = p.loc[d, "AccountBalance"] + p.loc[d, "TransactionAmount"]
    p.loc[d, "TransactionAmount"] = available * rng.uniform(0.9, 1.0, d.sum())
    p.loc[d, "AccountBalance"] = (available - p.loc[d, "TransactionAmount"]).clip(lower=1)
    p["TransactionID"] = [f"PLANTED_{i:02d}" for i in range(len(p))]

    mixed = add_features(pd.concat([raw.assign(kind="original"), p], ignore_index=True))
    return mixed, (mixed["kind"] != "original").astype(int)


def evaluate(mixed: pd.DataFrame, y: pd.Series, seed: int) -> list[dict]:
    """Rank quality of every method on one planted run."""
    s = score(mixed, seed=seed)
    candidates = {
        "amount only (robust z)": amount_rule(mixed["TransactionAmount"]),
        "Isolation Forest": s["iforest"],
        "KNN": s["knn"],
        "LOF": s["lof"],
        "ensemble: mean of probabilities": pd.DataFrame({c: to_probability(s[c]) for c in ["iforest", "knn", "lof"]}, index=s.index).mean(axis=1),
        "ensemble: mean of z-scores": s["ensemble"],
    }
    rows = []
    for method, sc in candidates.items():
        alerted = flag(sc, 0.02)
        rows.append({
            "method": method,
            "ROC-AUC": roc_auc_score(y, sc),
            "avg precision": average_precision_score(y, sc),
            "recall @2%": alerted[y == 1].mean(),
            **{f"recall: {k}": alerted[mixed["kind"] == k].mean() for k in KINDS},
        })
    return rows


def ulb_report() -> dict:
    """Real labels: 284,807 European card payments (Sept 2013), 492 confirmed frauds. Labels only grade the result."""
    # Only needed here. Downloads about 150 MB once into ~/scikit_learn_data.
    from sklearn.datasets import fetch_openml

    df = fetch_openml(data_id=1597, as_frame=True, parser="pandas").frame
    y = (df["Class"].astype(str) == "1").to_numpy()
    # Time is seconds since the capture started, i.e. row order, so it is left out.
    s = score(df, [f"V{i}" for i in range(1, 29)] + ["Amount"])
    methods = {"ensemble": s["ensemble"], "Isolation Forest": s["iforest"], "KNN": s["knn"], "LOF": s["lof"],
               "amount rule": amount_rule(df["Amount"])}
    base = y.mean()
    out = {"rows": len(df), "frauds": int(y.sum()), "rates": RATES, "recall": {}, "precision": {}, "roc_auc": {}, "pr_auc": {}}
    for name, sc in methods.items():
        flags = [flag(sc, r).to_numpy() for r in RATES]
        out["recall"][name] = [round(float(f[y].mean()), 4) for f in flags]
        out["precision"][name] = [round(float(y[f].mean()), 4) for f in flags]
        out["roc_auc"][name] = round(roc_auc_score(y, sc), 4)
        out["pr_auc"][name] = round(average_precision_score(y, sc), 4)
    out["recall"]["random"], out["precision"]["random"] = RATES, [round(float(base), 4)] * len(RATES)
    out["roc_auc"]["random"], out["pr_auc"]["random"] = 0.5, round(float(base), 4)
    return out


def planted_report(raw: pd.DataFrame) -> dict:
    runs = pd.DataFrame([row for seed in range(20) for row in evaluate(*plant_anomalies(raw, seed), seed)])
    by_method = runs.groupby("method", sort=False)
    agg = by_method[["ROC-AUC", "avg precision", "recall @2%"]].agg(["mean", "std"]).round(3)
    return {
        "runs": 20, "planted_per_run": 27,
        "methods": {m: {k: [agg.loc[m, (k, "mean")], agg.loc[m, (k, "std")]] for k in ["ROC-AUC", "avg precision", "recall @2%"]}
                    for m in agg.index},
        "by_type": by_method[[f"recall: {k}" for k in KINDS]].mean().round(3).rename(columns=lambda c: c[8:]).to_dict("index"),
    }


def notes(raw: pd.DataFrame) -> dict:
    """Numbers behind the data-quality sentences on the page."""
    dc, czech = pd.read_csv(DC), pd.read_csv(CZECH)
    multi = raw.groupby("AccountID").filter(lambda g: len(g) > 1).groupby("AccountID")
    hours = pd.to_datetime(raw["TransactionDate"]).dt.hour
    prob = to_probability(score(add_features(raw))["iforest"])
    return {
        "kaggle": {
            "rows": len(raw), "accounts": raw["AccountID"].nunique(),
            "prev_date_later": round(float((pd.to_datetime(raw["PreviousTransactionDate"]) > pd.to_datetime(raw["TransactionDate"])).mean()), 3),
            "first_hour": int(hours.min()), "last_hour": int(hours.max()),
            "new_device_every_time": round(float((multi["DeviceID"].nunique() == multi.size()).mean()), 3),
            "prob_over_75": int((prob > 0.75).sum()), "top_2pct": round(0.02 * len(raw)),
        },
        "dc": {"rows": len(dc), "start": dc["TransactionDate"].min(), "end": dc["TransactionDate"].max(),
               "agencies": dc["Agency"].nunique(), "vendors": dc["Vendor"].nunique()},
        "czech": {"rows": len(czech), "withdrawals": int((czech["Debit"] > 0).sum()), "accounts": czech["AccountID"].nunique(),
                  "start": czech["TransactionDate"].min(), "end": czech["TransactionDate"].max()},
    }


if __name__ == "__main__":
    from app import analyse  # the exact function the upload API uses

    if not DC.exists():
        fetch_dc().to_csv(DC, index=False)
    if not CZECH.exists():
        fetch_czech().to_csv(CZECH, index=False)
    OUT.mkdir(exist_ok=True)
    for key, (path, recipe) in SAMPLES.items():
        (OUT / f"{key}.json").write_text(json.dumps(analyse(pd.read_csv(path), recipe), separators=(",", ":")))
        print("scored", key)
    raw = pd.read_csv(KAGGLE)
    report = {"notes": notes(raw), "planted": planted_report(raw)}
    print("planted done")
    report["ulb"] = ulb_report()
    (OUT / "report.json").write_text(json.dumps(report, indent=1, allow_nan=False))
    print(json.dumps(report, indent=1))
