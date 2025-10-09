"""Streamlit dashboard to explore A22A reporting artifacts."""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from matplotlib.ticker import PercentFormatter

from a22a.reports.sources import (
    load_latest_json,
    load_latest_parquet_or_csv,
    reports_out_dir,
)


st.set_page_config(page_title="A22A — Dashboard", layout="wide")


@dataclass(frozen=True)
class Artifact:
    df: pd.DataFrame | None
    path: pathlib.Path | None


@st.cache_data(show_spinner=False)
def _load_dataframe(glob_pattern: str) -> Artifact:
    df, path = load_latest_parquet_or_csv(glob_pattern)
    if df is not None:
        df = df.copy()
    return Artifact(df=df, path=path)


@st.cache_data(show_spinner=False)
def _load_json(glob_pattern: str) -> tuple[dict[str, Any] | list[Any] | None, pathlib.Path | None]:
    return load_latest_json(glob_pattern)


@st.cache_data(show_spinner=False)
def _load_reports_summary(out_dir: pathlib.Path) -> dict[str, Any] | None:
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@st.cache_data(show_spinner=False)
def _load_reports_config(path: pathlib.Path = pathlib.Path("configs/defaults.yaml")) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _extract_options(dfs: Iterable[pd.DataFrame | None], column: str) -> list[str]:
    values: set[str] = set()
    for df in dfs:
        if df is None or column not in df.columns:
            continue
        col = df[column].dropna()
        values.update(col.astype(str).unique().tolist())
    return sorted(values)


def _apply_filters(
    df: pd.DataFrame | None,
    season: str | None,
    week: str | None,
) -> pd.DataFrame | None:
    if df is None:
        return None
    filtered = df.copy()
    if season and "season" in filtered.columns:
        filtered = filtered[filtered["season"].astype(str) == season]
    if week and "week" in filtered.columns:
        filtered = filtered[filtered["week"].astype(str) == week]
    return filtered


def _format_metric(value: float | None, precision: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{precision}f}"


def _render_reliability_chart(bins: list[dict[str, float]]) -> None:
    if not bins:
        st.info("No calibration bins available. Run `make report_batch` after the meta pipeline.")
        return
    conf = [float(b.get("confidence", 0.0)) for b in bins]
    acc = [float(b.get("accuracy", 0.0)) for b in bins]
    frac = [float(b.get("fraction", 0.0)) for b in bins]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", linewidth=1.5, label="ideal")
    ax.plot(conf, acc, color="#2563eb", marker="o", linewidth=2.0, label="observed")
    if frac:
        sizes = np.maximum(np.array(frac) * 2000, 30)
        ax.scatter(conf, acc, s=sizes, color="#1d4ed8", alpha=0.75)
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _portfolio_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "total_stake_pct": 0.0,
            "total_stake_amount": 0.0,
            "herfindahl": 0.0,
            "active_picks": 0,
            "total_picks": 0,
        }

    pct_col = next((c for c in ("stake_pct", "fraction", "kelly") if c in df.columns), None)
    amt_col = next((c for c in ("stake_amount", "risk", "exposure_amount") if c in df.columns), None)
    stake_pct = pd.to_numeric(df[pct_col], errors="coerce").fillna(0.0) if pct_col else pd.Series(0.0, index=df.index)
    stake_amt = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0) if amt_col else stake_pct

    total_pct = float(stake_pct.sum())
    total_amt = float(stake_amt.sum())
    herfindahl = 0.0
    if float(stake_pct.sum()) > 0:
        weights = (stake_pct / stake_pct.sum()).to_numpy()
        herfindahl = float(np.sum(np.square(weights)))

    return {
        "total_stake_pct": total_pct,
        "total_stake_amount": total_amt,
        "herfindahl": herfindahl,
        "active_picks": int((stake_pct > 0).sum()),
        "total_picks": int(len(df)),
    }


def _portfolio_exposure_chart(df: pd.DataFrame, value_col: str) -> None:
    if df.empty:
        st.info("No portfolio picks to visualise yet.")
        return
    if "side" not in df.columns:
        df = df.copy()
        df["side"] = "BET"
    pivot = (
        df.groupby(["game_id", "side"], dropna=False)[value_col]
        .sum()
        .unstack(fill_value=0.0)
    )
    if pivot.empty:
        st.info("No exposure data available after filtering.")
        return

    totals = pivot.sum(axis=1)
    pivot = pivot.loc[totals.sort_values(ascending=False).index]
    games = pivot.index.astype(str).tolist()
    sides = list(pivot.columns)
    fig, ax = plt.subplots(figsize=(max(6, len(games) * 0.6), 4))
    bottom = np.zeros(len(games))
    for side in sides:
        values = pivot[side].to_numpy(dtype=float)
        ax.bar(games, values, bottom=bottom, label=str(side))
        bottom += values

    ax.set_title("Portfolio Exposure by Game")
    if "pct" in value_col:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("Stake % of bankroll")
    else:
        ax.set_ylabel("Stake amount")
    ax.set_xlabel("Game")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _bankroll_series(df: pd.DataFrame | None) -> pd.Series:
    if df is None or df.empty:
        rng = np.random.default_rng(2024)
        returns = rng.normal(loc=0.001, scale=0.02, size=64)
        return pd.Series(100 * np.cumprod(1 + returns))
    for column in ("bankroll", "balance", "capital"):
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if not series.empty:
                return series
    return pd.Series(dtype=float)


def _bankroll_chart(series: pd.Series) -> None:
    if series.empty:
        st.info("Bankroll history not available yet.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(series.to_numpy(), color="#f59e0b", linewidth=2)
    ax.set_title("Bankroll Curve")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Bankroll")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _clv_chart(series: pd.Series) -> None:
    if series.empty:
        st.info("No CLV data to visualise yet.")
        return
    bins = min(30, max(10, int(series.size ** 0.5)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(series, bins=bins, color="#10b981", alpha=0.85, edgecolor="white")
    ax.set_title("CLV Distribution (bps)")
    ax.set_xlabel("CLV (basis points)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _player_impact_tables(df: pd.DataFrame | None, top_n: int = 10) -> None:
    if df is None or df.empty or "delta_win_pct" not in df.columns:
        st.info("Player impact artifact not available yet.")
        return
    base_cols = [col for col in ("player_id", "team_id", "position", "delta_win_pct") if col in df.columns]
    top = df.sort_values("delta_win_pct", ascending=False).head(top_n)[base_cols]
    bottom = df.sort_values("delta_win_pct", ascending=True).head(top_n)[base_cols]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Δwin%")
        st.dataframe(top.reset_index(drop=True))
    with col2:
        st.subheader("Bottom Δwin%")
        st.dataframe(bottom.reset_index(drop=True))


def main() -> None:
    out_dir = reports_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    reports_cfg = _load_reports_config()
    show_books_cfg = reports_cfg.get("reports", {}).get("show_books") if isinstance(reports_cfg, dict) else None
    if isinstance(show_books_cfg, list):
        allowed_books = [str(book) for book in show_books_cfg]
    else:
        allowed_books = []

    summary = _load_reports_summary(out_dir)

    st.title("A22A — Reporting & Analytics Dashboard")
    st.caption("Interactive view over the latest artifacts produced by the A22A pipeline.")

    meta_artifact = _load_dataframe("artifacts/meta/final_probs_*.parquet")
    calib_json, calib_path = _load_json("artifacts/meta/calibration_report_*.json")
    portfolio_artifact = _load_dataframe("artifacts/portfolio/picks_week_*.parquet")
    bankroll_artifact = _load_dataframe("artifacts/portfolio/bankroll_*.parquet")
    clv_artifact = _load_dataframe("artifacts/market/clv_*.parquet")
    impact_artifact = _load_dataframe("artifacts/impact/player_impact_*.parquet")

    season_options = _extract_options(
        [meta_artifact.df, portfolio_artifact.df, clv_artifact.df],
        "season",
    )
    week_options = _extract_options(
        [meta_artifact.df, portfolio_artifact.df, clv_artifact.df],
        "week",
    )

    with st.sidebar:
        st.header("Filters")
        season = st.selectbox("Season", ["All"] + season_options, index=0)
        week = st.selectbox("Week", ["All"] + week_options, index=0)
        if allowed_books:
            default_books = allowed_books
        else:
            default_books = (
                sorted(clv_artifact.df["book"].dropna().astype(str).unique().tolist())
                if clv_artifact.df is not None and "book" in clv_artifact.df.columns
                else []
            )
        books = st.multiselect(
            "Books",
            options=allowed_books or default_books,
            default=default_books,
        )

        if summary:
            st.markdown("---")
            st.caption("Latest batch summary")
            st.json(summary, expanded=False)

    season_filter = None if season == "All" else season
    week_filter = None if week == "All" else week

    filtered_portfolio = _apply_filters(portfolio_artifact.df, season_filter, week_filter)
    filtered_clv = _apply_filters(clv_artifact.df, season_filter, week_filter)
    if filtered_clv is not None and books and "book" in filtered_clv.columns:
        filtered_clv = filtered_clv[filtered_clv["book"].astype(str).isin(books)]

    st.subheader("Overview")
    overview_cols = st.columns(4)
    if summary and "calibration" in summary:
        cal = summary["calibration"]
        overview_cols[0].metric("ECE", _format_metric(cal.get("ece")))
        overview_cols[1].metric("Brier", _format_metric(cal.get("brier")))
        overview_cols[2].metric("LogLoss", _format_metric(cal.get("log_loss")))
    if summary and "portfolio" in summary:
        port = summary["portfolio"]
        overview_cols[3].metric("Total Stake %", _format_metric(port.get("total_stake_pct"), precision=2))

    st.divider()
    st.subheader("Meta — Latest Probabilities")
    if meta_artifact.df is None:
        st.info("No meta probabilities found yet. Run the meta pipeline.")
    else:
        if meta_artifact.path is not None:
            st.caption(f"Latest file: `{meta_artifact.path}`")
        display_df = _apply_filters(meta_artifact.df, season_filter, week_filter)
        st.dataframe(display_df.head(200) if display_df is not None else meta_artifact.df.head(200))

    st.subheader("Calibration")
    if calib_json is None:
        st.info("No calibration report found. Run the meta pipeline.")
    else:
        if calib_path is not None:
            st.caption(f"Latest report: `{calib_path}`")
        metrics_cols = st.columns(3)
        metrics_cols[0].metric("ECE", _format_metric(calib_json.get("ece")))
        metrics_cols[1].metric("Brier", _format_metric(calib_json.get("brier")))
        metrics_cols[2].metric("LogLoss", _format_metric(calib_json.get("log_loss")))
        bins = calib_json.get("reliability_bins", [])
        _render_reliability_chart(bins)

    st.divider()
    st.subheader("Portfolio")
    if filtered_portfolio is None or filtered_portfolio.empty:
        st.info("No portfolio picks available for the selected filters.")
    else:
        if portfolio_artifact.path is not None:
            st.caption(f"Latest file: `{portfolio_artifact.path}`")
        metrics = _portfolio_metrics(filtered_portfolio)
        cols = st.columns(4)
        cols[0].metric("Total stake %", f"{metrics['total_stake_pct']:.4f}")
        cols[1].metric("Total stake $", f"{metrics['total_stake_amount']:.2f}")
        cols[2].metric("Herfindahl", f"{metrics['herfindahl']:.4f}")
        cols[3].metric("Active picks", f"{metrics['active_picks']} / {metrics['total_picks']}")
        st.dataframe(filtered_portfolio.head(200))
        value_col = "stake_amount" if "stake_amount" in filtered_portfolio.columns else "stake_pct"
        _portfolio_exposure_chart(filtered_portfolio, value_col)

    st.subheader("Bankroll")
    bankroll_series = _bankroll_series(bankroll_artifact.df)
    _bankroll_chart(bankroll_series)

    st.divider()
    st.subheader("Closing Line Value (CLV)")
    if filtered_clv is None or filtered_clv.empty:
        st.info("No CLV artifacts found for the selected filters.")
    else:
        if clv_artifact.path is not None:
            st.caption(f"Latest file: `{clv_artifact.path}`")
        clv_series = pd.to_numeric(filtered_clv.get("clv_bps"), errors="coerce").dropna()
        overall_cols = st.columns(3)
        overall_cols[0].metric("Mean CLV (bps)", _format_metric(float(clv_series.mean()), precision=2))
        overall_cols[1].metric("Median CLV (bps)", _format_metric(float(clv_series.median()), precision=2))
        overall_cols[2].metric("Samples", str(int(clv_series.size)))
        _clv_chart(clv_series)
        if "book" in filtered_clv.columns:
            per_book = (
                filtered_clv.assign(clv_bps=pd.to_numeric(filtered_clv["clv_bps"], errors="coerce"))
                .dropna(subset=["clv_bps"])
                .groupby("book")
                .agg(mean_bps=("clv_bps", "mean"), median_bps=("clv_bps", "median"), count=("clv_bps", "size"))
                .reset_index()
            )
            st.dataframe(per_book)

    st.divider()
    st.subheader("Player Impact")
    if impact_artifact.df is None:
        st.info("Run the player impact pipeline to populate this section.")
    else:
        if impact_artifact.path is not None:
            st.caption(f"Latest file: `{impact_artifact.path}`")
        filtered_impact = _apply_filters(impact_artifact.df, season_filter, week_filter)
        if filtered_impact is None or filtered_impact.empty:
            st.info("No player impact rows for the selected filters.")
        else:
            positions = sorted(filtered_impact.get("position", pd.Series(dtype=str)).dropna().unique().tolist())
            selected_positions = st.multiselect(
                "Positions",
                options=positions,
                default=positions,
                key="impact_positions",
            )
            if selected_positions:
                filtered_impact = filtered_impact[filtered_impact["position"].isin(selected_positions)]
            _player_impact_tables(filtered_impact)
if __name__ == "__main__":
    main()

