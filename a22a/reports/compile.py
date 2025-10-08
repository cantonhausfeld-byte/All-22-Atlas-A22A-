"""Batch compilation for reporting artifacts."""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from a22a.reports.sources import (
    load_latest_json,
    load_latest_parquet_or_csv,
    reports_out_dir,
)


def _ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_placeholder_line(path: pathlib.Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], color="#2563eb", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("placeholder x")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_histogram(path: pathlib.Path, title: str, values: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(values, bins=20, color="#10b981", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _safe_numeric_series(frame: pd.DataFrame, column_candidates: list[str]) -> pd.Series | None:
    for column in column_candidates:
        if column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not series.empty:
                return series
    return None


def main() -> dict[str, Any]:
    out_dir = _ensure_dir(reports_out_dir())

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "artifacts": {},
    }

    # Meta calibration inputs
    meta_df, meta_path = load_latest_parquet_or_csv("artifacts/meta/final_probs_*.parquet")
    if meta_path:
        summary["artifacts"]["meta_final_probs"] = meta_path.name

    calib_json, calib_path = load_latest_json("artifacts/meta/calibration_report_*.json")
    if calib_path:
        summary["artifacts"]["calibration_report"] = calib_path.name
        summary["calibration"] = calib_json
    else:
        summary["calibration"] = None

    calibration_png = out_dir / "calibration.png"
    _write_placeholder_line(calibration_png, "A22A — Calibration", "expected vs actual")

    # Portfolio exposure placeholder
    portfolio_df, portfolio_path = load_latest_parquet_or_csv("artifacts/portfolio/picks_week_*.parquet")
    if portfolio_path:
        summary["artifacts"]["portfolio_picks"] = portfolio_path.name
        exposure_series = _safe_numeric_series(portfolio_df, ["stake", "wager", "risk"]) or pd.Series([0])
    else:
        exposure_series = pd.Series([0])
    exposure_png = out_dir / "portfolio_exposure.png"
    _write_histogram(exposure_png, "Portfolio Exposure", exposure_series)

    # Bankroll curve placeholder (requires cumulative bankroll)
    bankroll_df, bankroll_path = load_latest_parquet_or_csv("artifacts/portfolio/bankroll_*.parquet")
    bankroll_png = out_dir / "bankroll_curve.png"
    if bankroll_path and bankroll_df is not None:
        summary["artifacts"]["bankroll_history"] = bankroll_path.name
        bankroll_series = _safe_numeric_series(bankroll_df, ["bankroll", "balance", "capital"]) or pd.Series([0, 1])
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(bankroll_series.to_numpy(), color="#f59e0b", linewidth=2)
        ax.set_title("Bankroll Curve")
        ax.set_xlabel("observation")
        ax.set_ylabel("bankroll")
        fig.tight_layout()
        fig.savefig(bankroll_png, bbox_inches="tight")
        plt.close(fig)
    else:
        _write_placeholder_line(bankroll_png, "Bankroll Curve", "bankroll")

    # CLV distribution placeholder
    clv_df, clv_path = load_latest_parquet_or_csv("artifacts/market/clv_*.parquet")
    if clv_path:
        summary["artifacts"]["market_clv"] = clv_path.name
        clv_series = _safe_numeric_series(clv_df, ["edge", "clv", "delta"])
    else:
        clv_series = None
    clv_png = out_dir / "clv_distribution.png"
    if clv_series is not None:
        _write_histogram(clv_png, "CLV Distribution", clv_series)
    else:
        _write_placeholder_line(clv_png, "CLV Distribution", "edge")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[reports] summary written to {summary_path.relative_to(pathlib.Path('.'))}")
    return summary


if __name__ == "__main__":
    main()
