"""Streamlit dashboard to explore A22A reporting artifacts."""

from __future__ import annotations

import json
import streamlit as st

from a22a.reports.sources import (
    load_latest_json,
    load_latest_parquet_or_csv,
    reports_out_dir,
)

st.set_page_config(page_title="A22A — Dashboard", layout="wide")

st.title("A22A — Reporting & Analytics Dashboard")

out_dir = reports_out_dir()
st.sidebar.header("Artifacts")
summary_path = out_dir / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    st.sidebar.success(f"Summary loaded from `{summary_path}`")
    st.sidebar.json(summary)
else:
    st.sidebar.info("No batch summary found yet. Run `make report_batch`.")

st.markdown("This dashboard surfaces the latest artifacts produced by the pipeline.")

meta_col, calibration_col = st.columns(2)

with meta_col:
    st.subheader("Meta — Final Probabilities")
    meta_df, meta_path = load_latest_parquet_or_csv("artifacts/meta/final_probs_*.parquet")
    if meta_df is None:
        st.info("No meta probabilities found yet.")
    else:
        st.caption(f"Latest file: `{meta_path}`")
        st.dataframe(meta_df.head(50))

with calibration_col:
    st.subheader("Calibration Report")
    calib_json, calib_path = load_latest_json("artifacts/meta/calibration_report_*.json")
    if calib_json is None:
        st.info("No calibration reports found yet.")
    else:
        st.caption(f"Latest file: `{calib_path}`")
        st.json(calib_json)

st.divider()

portfolio_col, exposure_col = st.columns(2)

with portfolio_col:
    st.subheader("Portfolio — Recent Picks")
    portfolio_df, portfolio_path = load_latest_parquet_or_csv("artifacts/portfolio/picks_week_*.parquet")
    if portfolio_df is None:
        st.info("No portfolio picks found yet.")
    else:
        st.caption(f"Latest file: `{portfolio_path}`")
        st.dataframe(portfolio_df.head(100))

with exposure_col:
    st.subheader("Portfolio Exposure Snapshot")
    exposure_png = out_dir / "portfolio_exposure.png"
    if exposure_png.exists():
        st.image(str(exposure_png))
    else:
        st.info("Run the batch reporter to generate exposure plots.")

st.divider()

bankroll_col, bankroll_plot_col = st.columns(2)

with bankroll_col:
    st.subheader("Bankroll History")
    bankroll_df, bankroll_path = load_latest_parquet_or_csv("artifacts/portfolio/bankroll_*.parquet")
    if bankroll_df is None:
        st.info("No bankroll history found yet.")
    else:
        st.caption(f"Latest file: `{bankroll_path}`")
        st.dataframe(bankroll_df.tail(50))

with bankroll_plot_col:
    st.subheader("Bankroll Curve")
    bankroll_png = out_dir / "bankroll_curve.png"
    if bankroll_png.exists():
        st.image(str(bankroll_png))
    else:
        st.info("Run the batch reporter to generate bankroll plots.")

st.divider()

clv_col, clv_plot_col = st.columns(2)

with clv_col:
    st.subheader("Market — CLV")
    clv_df, clv_path = load_latest_parquet_or_csv("artifacts/market/clv_*.parquet")
    if clv_df is None:
        st.info("No CLV artifacts found yet.")
    else:
        st.caption(f"Latest file: `{clv_path}`")
        st.dataframe(clv_df.head(100))

with clv_plot_col:
    st.subheader("CLV Distribution")
    clv_png = out_dir / "clv_distribution.png"
    if clv_png.exists():
        st.image(str(clv_png))
    else:
        st.info("Run the batch reporter to generate CLV plots.")

st.divider()

calibration_png = out_dir / "calibration.png"
st.subheader("Calibration Chart")
if calibration_png.exists():
    st.image(str(calibration_png))
else:
    st.info("Run the batch reporter to generate calibration charts.")
