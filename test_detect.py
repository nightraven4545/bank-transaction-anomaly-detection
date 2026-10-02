from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from detect import FEATURES, add_features, explain, flag, score

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
    reasons = explain(df).loc[scores["ensemble"].idxmax()]
    assert reasons.count("p100") == 2
