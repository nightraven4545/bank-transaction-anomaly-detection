"""Anomaly detection for bank transactions: features, scores, alerts, reason codes.

Shared by notebook.ipynb, app.py and test_detect.py.
"""

import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratios that put each amount in context of the balance and the account's own habits."""
    out = df.copy()
    # clip() guards against division by zero on uploaded data with empty accounts
    out["amount_to_balance"] = out["TransactionAmount"] / out["AccountBalance"].clip(lower=1)
    account_median = out.groupby("AccountID")["TransactionAmount"].transform("median")
    out["amount_vs_account_median"] = out["TransactionAmount"] / account_median.clip(lower=0.01)
    return out


def score(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Anomaly z-scores from Isolation Forest, KNN and LOF, plus their mean ('ensemble').

    z = how many standard deviations above the average transaction a score sits.
    """
    X = RobustScaler().fit_transform(df[FEATURES])
    detectors = {"iforest": IForest(random_state=seed), "knn": KNN(), "lof": LOF()}
    # decision_scores_ (training scores), not decision_function(X): re-scoring the training
    # data makes KNN/LOF count each point as its own neighbour, which biases scores downward.
    raw = {name: det.fit(X).decision_scores_ for name, det in detectors.items()}
    out = pd.DataFrame({name: (s - s.mean()) / s.std() for name, s in raw.items()}, index=df.index)
    # Average z-scores, not outlier probabilities: probabilities saturate near 1 and flatten
    # the top of the ranking (notebook section 8 measures the difference).
    out["ensemble"] = out.mean(axis=1)
    return out


def flag(scores: pd.Series, alert_rate: float = 0.02) -> pd.Series:
    """Capacity-based alerting: flag the top `alert_rate` share, like a fraud team sized to N reviews/day."""
    n_alerts = max(1, round(alert_rate * len(scores)))
    return scores.rank(ascending=False, method="first") <= n_alerts


def explain(df: pd.DataFrame, top: int = 2) -> pd.Series:
    """Reason codes: the features where each transaction sits furthest from typical (by percentile)."""
    pct = df[FEATURES].rank(pct=True)
    order = (pct - 0.5).abs().to_numpy().argsort(axis=1)[:, ::-1][:, :top]
    return pd.Series(
        [", ".join(f"{FEATURES[j]} p{pct.iat[i, j] * 100:.0f}" for j in row) for i, row in enumerate(order)],
        index=df.index,
    )
