"""Streamlit dashboard: upload bank transactions, get ranked anomaly alerts with reasons."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from detect import FEATURES, REQUIRED, add_features, explain, flag, score

SAMPLE = Path(__file__).parent / "data" / "bank_transactions_data_2.csv"
MIN_ROWS, MAX_ROWS = 50, 50_000  # KNN/LOF need neighbours; cap keeps scoring fast

st.set_page_config(page_title="Transaction Anomaly Detector", page_icon=":mag:", layout="wide")
st.title("Bank transaction anomaly detector")
st.caption("Isolation Forest + KNN + LOF ensemble (PyOD) ranks transactions by how unusual they are.")

with st.sidebar:
    upload = st.file_uploader("Upload transactions CSV", type="csv")
    alert_rate = st.slider("Alert rate (% of transactions to review)", 0.5, 10.0, 2.0, 0.5) / 100
    st.caption(f"Required columns: {', '.join(REQUIRED)}")
    if upload is None:
        st.info("Using the bundled sample (2,512 transactions).")

# Uploaded files are untrusted: validate before any modelling.
try:
    df = pd.read_csv(upload if upload is not None else SAMPLE)
except ValueError as e:  # covers parser, empty-file and encoding errors
    st.error(f"Could not read the CSV: {e}")
    st.stop()

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()
bad = [c for c in REQUIRED[1:] if not pd.api.types.is_numeric_dtype(df[c]) or not np.isfinite(df[c]).all()]
if bad:
    st.error(f"These columns must be numeric with no blanks: {', '.join(bad)}")
    st.stop()
if not MIN_ROWS <= len(df) <= MAX_ROWS:
    st.error(f"Need between {MIN_ROWS} and {MAX_ROWS:,} rows, got {len(df):,}.")
    st.stop()


@st.cache_data(show_spinner="Scoring transactions...")
def run(df: pd.DataFrame) -> pd.DataFrame:
    df = add_features(df)
    return df.join(score(df)).assign(reason=explain(df))


res = run(df)
res["flagged"] = flag(res["ensemble"], alert_rate)
alerts = res[res["flagged"]].sort_values("ensemble", ascending=False)

c1, c2, c3 = st.columns(3)
c1.metric("Transactions", f"{len(res):,}")
c2.metric("Flagged for review", f"{len(alerts):,}")
c3.metric("Flagged amount", f"${alerts['TransactionAmount'].sum():,.0f}")

st.subheader("Alerts, most suspicious first")
shown = [c for c in ["TransactionID", *REQUIRED, "Channel"] if c in res.columns]
st.dataframe(
    alerts[["ensemble", "reason", *shown]],
    hide_index=True,
    column_config={
        "ensemble": st.column_config.NumberColumn("Anomaly score (z)", format="%.1f"),
        "reason": st.column_config.TextColumn("Why (feature percentile)"),
    },
)
st.download_button("Download alerts CSV", alerts.to_csv(index=False), "alerts.csv", "text/csv")

st.subheader("Where the alerts sit")
left, right = st.columns(2)
x = left.selectbox("X axis", FEATURES, index=0)
y = right.selectbox("Y axis", FEATURES, index=3)
plot = pd.DataFrame({x: res[x], "normal": res[y].where(~res["flagged"]), "flagged": res[y].where(res["flagged"])})
st.scatter_chart(plot, x=x, y=["normal", "flagged"], color=["#9aa5b1", "#d1495b"], x_label=x, y_label=y)

with st.expander("How it works"):
    st.markdown(
        "- **Features**: amount, duration, login attempts, amount ÷ balance, amount ÷ the account's usual amount.\n"
        "- **Detectors**: Isolation Forest (easy to isolate), KNN (far from neighbours), "
        "LOF (sparser than its neighbourhood). Each score is standardised (z) and the three are averaged.\n"
        "- **Alerts**: the top *alert rate* share, sized to review capacity.\n"
        "- **Why**: the two features where the transaction sits furthest from typical (p99 = top 1%)."
    )
