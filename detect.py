"""Anomaly detection for transactions: feature recipes, scores, alerts, reason codes.

Shared by notebook.ipynb, report.py, app.py and the tests.
Every feature is built so that higher means more unusual, which keeps reason codes one-directional.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler

REQUIRED = ["AccountID", "TransactionAmount", "TransactionDuration", "LoginAttempts", "AccountBalance"]

# Behavioural features only. Age, occupation and raw balance describe the customer,
# not the transaction, so they are left out on purpose (no flagging people for being old or rich).
FEATURES = [
    "TransactionAmount",
    "TransactionDuration",
    "LoginAttempts",
    "amount_to_balance",
    "amount_vs_account_median",
]
# Government purchase cards have no cardholder ID, so the agency and the merchant category are the baselines.
PCARD_FEATURES = ["TransactionAmount", "amount_vs_agency_median", "amount_vs_category_median", "category_rarity_in_agency"]
# Bank ledgers and statements: money going out, against the account's own history.
LEDGER_FEATURES = ["Debit", "share_of_balance", "amount_vs_account_median", "days_since_prev"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratios that put each amount in context of the balance and the account's own habits."""
    out = df.copy()
    # clip() guards against division by zero on uploaded data with empty accounts
    out["amount_to_balance"] = out["TransactionAmount"] / out["AccountBalance"].clip(lower=1)
    account_median = out.groupby("AccountID")["TransactionAmount"].transform("median")
    out["amount_vs_account_median"] = out["TransactionAmount"] / account_median.clip(lower=0.01)
    return out


def pcard_features(df: pd.DataFrame) -> pd.DataFrame:
    """Columns TransactionAmount, Agency, Category: compare each purchase with its agency and its category."""
    out = df.copy()
    amount = out["TransactionAmount"]
    for col in ["Agency", "Category"]:
        out[f"amount_vs_{col.lower()}_median"] = amount / out.groupby(col)["TransactionAmount"].transform("median").clip(lower=0.01)
    per_agency = out.groupby("Agency")["Agency"].transform("size")
    out["category_rarity_in_agency"] = 1 - out.groupby(["Agency", "Category"])["Agency"].transform("size") / per_agency
    return out


def ledger_features(df: pd.DataFrame) -> pd.DataFrame:
    """Columns AccountID, TransactionDate, Debit, Balance (after the transaction). Returns debit rows only."""
    out = df.assign(TransactionDate=pd.to_datetime(df["TransactionDate"]))
    out = out.sort_values(["AccountID", "TransactionDate"], kind="stable")
    out = out[out["Debit"] > 0].copy()
    # Balance before = balance after + debit. Clipping the denominator makes overdraft withdrawals score high.
    out["share_of_balance"] = out["Debit"] / (out["Balance"] + out["Debit"]).clip(lower=1)
    by_account = out.groupby("AccountID")
    out["amount_vs_account_median"] = out["Debit"] / by_account["Debit"].transform("median").clip(lower=0.01)
    gap = by_account["TransactionDate"].diff().dt.days  # long gap = dormant account suddenly paying out
    out["days_since_prev"] = gap.fillna(gap.groupby(out["AccountID"]).transform("median")).fillna(0)
    return out


RECIPES = {
    "bank": (add_features, FEATURES),
    "pcard": (pcard_features, PCARD_FEATURES),
    "ledger": (ledger_features, LEDGER_FEATURES),
}


def score(df: pd.DataFrame, features: list[str] = FEATURES, seed: int = 42) -> pd.DataFrame:
    """Anomaly z-scores from Isolation Forest, KNN and LOF, plus their mean ('ensemble').

    The site ranks by 'iforest'. report.py measured it best on real fraud labels, on planted frauds
    and on reason quality; LOF breaks on duplicate-heavy data and drags the mean down. KNN, LOF and the
    mean are kept as second opinions and for the comparison on the results page.

    Same estimators and defaults that PyOD's IForest/KNN/LOF wrap (test_detect.py checks the
    scores match), minus PyOD's heavy numba dependency so the app stays small on Vercel.
    z = how many standard deviations above the average transaction a score sits.
    """
    X = RobustScaler().fit_transform(df[features])
    # Training-set scores throughout: re-scoring the training data as "new" points would make
    # KNN/LOF count each point as its own neighbour and bias the scores downward.
    raw = {
        "iforest": -IsolationForest(random_state=seed).fit(X).score_samples(X),
        "knn": NearestNeighbors(n_neighbors=5).fit(X).kneighbors()[0][:, -1],  # 5th neighbour, self excluded
        "lof": -LocalOutlierFactor(n_neighbors=20).fit(X).negative_outlier_factor_,
    }
    out = pd.DataFrame({name: (s - s.mean()) / s.std() for name, s in raw.items()}, index=df.index)
    # Average z-scores, not outlier probabilities: probabilities saturate near 1 and flatten
    # the top of the ranking (notebook section 8 measures the difference).
    out["ensemble"] = out.mean(axis=1)
    return out


def flag(scores: pd.Series, alert_rate: float = 0.02) -> pd.Series:
    """Capacity-based alerting: flag the top `alert_rate` share, like a fraud team sized to N reviews/day."""
    n_alerts = max(1, round(alert_rate * len(scores)))
    return scores.rank(ascending=False, method="first") <= n_alerts


def explain(df: pd.DataFrame, features: list[str] = FEATURES, top: int = 2) -> pd.Series:
    """Reason codes: the features where each transaction ranks highest (higher always means more unusual)."""
    pct = df[features].rank(pct=True)
    order = pct.to_numpy().argsort(axis=1)[:, ::-1][:, :top]
    return pd.Series(
        [", ".join(f"{features[j]} p{pct.iat[i, j] * 100:.0f}" for j in row) for i, row in enumerate(order)],
        index=df.index,
    )
