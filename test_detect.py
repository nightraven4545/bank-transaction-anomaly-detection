from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from detect import (
    FEATURES,
    RECIPES,
    add_features,
    explain,
    flag,
    ledger_features,
    score,
)

DATA = Path(__file__).parent / "data" / "bank_transactions_data_2.csv"


@pytest.fixture(scope="module")
def scored():
    df = pd.read_csv(DATA)
    fraud = df.iloc[[0]].assign(
        TransactionID="TX_FRAUD",
        TransactionAmount=25_000.0,
        LoginAttempts=5,
        TransactionDuration=600,
        AccountBalance=50.0,
    )
    df = add_features(pd.concat([df, fraud], ignore_index=True))
    return df, score(df)


def test_obvious_fraud_ranks_first_and_is_flagged(scored):
    df, scores = scored
    top = scores["ensemble"].idxmax()
    assert df.loc[top, "TransactionID"] == "TX_FRAUD"
    assert flag(scores["ensemble"], 0.02)[top]


def test_scores_are_standardised_and_aligned(scored):
    df, scores = scored
    assert scores.index.equals(df.index)
    assert scores.notna().all().all()
    detectors = scores[["iforest", "knn", "lof"]]
    assert detectors.mean().abs().max() < 1e-9
    assert (detectors.std(ddof=0) - 1).abs().max() < 1e-9
    assert (scores["ensemble"] - detectors.mean(axis=1)).abs().max() < 1e-12


def test_flag_matches_alert_rate(scored):
    _, scores = scored
    assert flag(scores["ensemble"], 0.05).sum() == round(0.05 * len(scores))


def test_scores_match_pyod(scored):
    pytest.importorskip("pyod")  # dev dependency: the app itself runs on scikit-learn only
    from pyod.models.iforest import IForest
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF

    df, scores = scored
    X = RobustScaler().fit_transform(df[FEATURES])
    for name, detector in {"iforest": IForest(random_state=42), "knn": KNN(), "lof": LOF()}.items():
        s = detector.fit(X).decision_scores_
        assert np.allclose((s - s.mean()) / s.std(), scores[name]), name


def test_explain_points_at_extreme_features(scored):
    df, scores = scored
    top = scores["ensemble"].idxmax()
    assert explain(df).loc[top].count("p100") == 2
    assert explain(df, ["TransactionAmount"], top=1).loc[top] == "TransactionAmount p100"


def test_custom_feature_list(scored):
    df, _ = scored
    scores = score(df, ["TransactionAmount", "LoginAttempts"])
    assert list(scores.columns) == ["iforest", "knn", "lof", "ensemble"] and scores.notna().all().all()


LEDGER = pd.DataFrame({
    "AccountID": [1, 1, 1, 2, 2],
    "TransactionDate": ["2024-01-05", "2024-01-01", "2024-01-09", "2024-01-01", "2024-01-02"],
    "Debit": [10, 20, 0, 500, 30],  # the 0 is a credit row
    "Balance": [-200, 100, 50, 0, 10],  # -200: an overdraft withdrawal
})
PCARD = pd.DataFrame({
    "TransactionAmount": [10, 20, 30, 5000],
    "Agency": ["A", "A", "B", "B"],
    "Category": ["x", "x", "y", "z"],
})


@pytest.mark.parametrize(("recipe", "df"), [("pcard", PCARD), ("ledger", LEDGER)])
def test_recipes_give_finite_features(recipe, df):
    make, features = RECIPES[recipe]
    assert np.isfinite(make(df)[features].to_numpy(dtype=float)).all()  # incl. each account's first row


def test_ledger_scores_withdrawals_and_overdrafts_rank_high():
    out = ledger_features(LEDGER)
    assert len(out) == 4  # credit row dropped
    assert out.loc[out["Balance"] < 0, "share_of_balance"].item() == out["share_of_balance"].max()
