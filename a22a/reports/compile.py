"""Batch compilation for reporting artifacts."""

from __future__ import annotations

import json
import logging
import math
import pathlib
import random
from datetime import datetime, timezone
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import PercentFormatter

from a22a.reports.sources import (
    load_latest_json,
    load_latest_parquet_or_csv,
    reports_out_dir,
)


LOGGER = logging.getLogger("a22a.reports.compile")


def _ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_config(path: pathlib.Path | None = None) -> dict[str, Any]:
    if path is None:
        path = pathlib.Path(__file__).resolve().parents[2] / "configs" / "defaults.yaml"
    if not path.exists():
        fallback = pathlib.Path("configs/defaults.yaml")
        path = fallback if fallback.exists() else path
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - config parse failures are rare
        LOGGER.warning("failed to load config %s: %s", path, exc)
        return {}


def _safe_numeric_series(frame: pd.DataFrame | None, column_candidates: Iterable[str]) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    for column in column_candidates:
        if column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not series.empty:
                return series
    return pd.Series(dtype=float)


def _herfindahl_index(weights: pd.Series) -> float:
    if weights.empty:
        return 0.0
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    values = (weights / total).to_numpy()
    return float(np.sum(np.square(values)))


def _synthetic_calibration_bins(n_bins: int = 10) -> list[dict[str, float]]:
    bins: list[dict[str, float]] = []
    width = 1.0 / n_bins
    for idx in range(n_bins):
        lo = idx * width
        hi = (idx + 1) * width
        mid = (lo + hi) / 2
        bins.append(
            {
                "lower": float(lo),
                "upper": float(hi),
                "count": 0.0,
                "fraction": 0.0,
                "confidence": float(mid),
                "accuracy": float(mid),
            }
        )
    return bins


def _plot_reliability(path: pathlib.Path, bins: Iterable[dict[str, float]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", linewidth=1.5, label="ideal")

    conf = [float(b.get("confidence", 0.0)) for b in bins]
    acc = [float(b.get("accuracy", 0.0)) for b in bins]
    frac = [float(b.get("fraction", 0.0)) for b in bins]

    if conf and acc:
        ax.plot(conf, acc, marker="o", color="#2563eb", linewidth=2.0, label="observed")
        ax.scatter(conf, acc, s=np.maximum(np.array(frac) * 2000, 30), color="#1d4ed8", alpha=0.75)

    ax.set_title(title)
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_portfolio_exposure(
    path: pathlib.Path,
    df: pd.DataFrame,
    value_col: str,
    title: str,
) -> None:
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title)
        ax.set_xlabel("Game")
        ax.set_ylabel("Exposure")
        ax.text(0.5, 0.5, "No picks", ha="center", va="center", transform=ax.transAxes, color="#6b7280")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    pivot = (
        df.groupby(["game_id", "side"], dropna=False)[value_col]
        .sum()
        .unstack(fill_value=0.0)
    )
    if pivot.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title)
        ax.set_xlabel("Game")
        ax.set_ylabel("Exposure")
        ax.text(0.5, 0.5, "No picks", ha="center", va="center", transform=ax.transAxes, color="#6b7280")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
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

    ax.set_title(title)
    ylabel = "Stake amount"
    if "pct" in value_col:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ylabel = "Stake % of bankroll"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Game")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_bankroll(path: pathlib.Path, series: pd.Series, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(series.to_numpy(), color="#f59e0b", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Bankroll")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_histogram(path: pathlib.Path, series: pd.Series, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(series, bins=min(30, max(10, int(series.size ** 0.5))), color="#10b981", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _synthetic_bankroll(length: int = 64, start: float = 100.0) -> pd.Series:
    rng = np.random.default_rng(1729)
    returns = rng.normal(loc=0.001, scale=0.02, size=length)
    bankroll = start * np.cumprod(1.0 + returns)
    return pd.Series(bankroll)


def _calibration_payload(
    meta_df: pd.DataFrame | None,
    calib_json: dict[str, Any] | None,
    bins: int,
) -> dict[str, Any]:
    if calib_json:
        payload = {
            "ece": float(calib_json.get("ece", math.nan)),
            "brier": float(calib_json.get("brier", math.nan)),
            "log_loss": float(calib_json.get("log_loss", math.nan)),
            "bins": [
                {
                    "lower": float(row.get("lower", 0.0)),
                    "upper": float(row.get("upper", 0.0)),
                    "count": float(row.get("count", 0.0)),
                    "fraction": float(row.get("fraction", 0.0)),
                    "confidence": float(row.get("confidence", 0.0)),
                    "accuracy": float(row.get("accuracy", 0.0)),
                }
                for row in calib_json.get("reliability_bins", [])
            ],
            "status": "ok",
        }
        return payload

    payload = {
        "ece": None,
        "brier": None,
        "log_loss": None,
        "bins": _synthetic_calibration_bins(bins),
        "status": "missing",
        "message": "calibration report unavailable",
    }
    return payload


def _portfolio_metrics(df: pd.DataFrame, portfolio_cfg: dict[str, Any]) -> dict[str, Any]:
    if df.empty:
        metrics = {
            "total_stake_pct": 0.0,
            "total_stake_amount": 0.0,
            "active_picks": 0,
            "picks_total": 0,
            "herfindahl": 0.0,
            "max_game_exposure": 0.0,
            "books": [],
        }
        metrics.update(
            {
                "max_stake_pct_per_bet": portfolio_cfg.get("max_stake_pct_per_bet"),
                "max_game_exposure_pct": portfolio_cfg.get("max_game_exposure_pct"),
                "max_weekly_exposure_pct": portfolio_cfg.get("max_weekly_exposure_pct"),
            }
        )
        return metrics

    pct_col = next((c for c in ("stake_pct", "fraction", "kelly") if c in df.columns), None)
    amt_col = next((c for c in ("stake_amount", "risk", "exposure_amount") if c in df.columns), None)

    stake_pct = pd.to_numeric(df[pct_col], errors="coerce").fillna(0.0) if pct_col else pd.Series(0.0, index=df.index)
    stake_amt = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0) if amt_col else stake_pct

    total_pct = float(stake_pct.sum())
    total_amt = float(stake_amt.sum())
    active_count = int((stake_pct > 0).sum()) if pct_col else int(len(df))

    herfindahl = _herfindahl_index(stake_pct)
    if "game_id" in df.columns:
        per_game_sum = df.assign(_stake=stake_pct).groupby("game_id")["_stake"].sum(min_count=1)
        max_game_exposure = float(per_game_sum.max()) if not per_game_sum.empty else 0.0
    else:
        max_game_exposure = float(stake_pct.max()) if not stake_pct.empty else 0.0

    books = []
    if "book" in df.columns:
        books = sorted({str(book) for book in df["book"].dropna().unique()})

    metrics = {
        "total_stake_pct": round(total_pct, 6),
        "total_stake_amount": round(total_amt, 2),
        "active_picks": active_count,
        "picks_total": int(len(df)),
        "herfindahl": round(herfindahl, 6),
        "max_game_exposure": round(max_game_exposure, 6),
        "books": books,
    }

    metrics.update(
        {
            "max_stake_pct_per_bet": portfolio_cfg.get("max_stake_pct_per_bet"),
            "max_game_exposure_pct": portfolio_cfg.get("max_game_exposure_pct"),
            "max_weekly_exposure_pct": portfolio_cfg.get("max_weekly_exposure_pct"),
        }
    )
    return metrics


def _bankroll_series(df: pd.DataFrame | None) -> tuple[pd.Series, dict[str, Any]]:
    if df is None or df.empty:
        series = _synthetic_bankroll()
        return series, {"status": "synthetic", "length": int(series.size)}

    series = _safe_numeric_series(df, ["bankroll", "balance", "capital"])
    if series.empty:
        series = _synthetic_bankroll()
        return series, {"status": "synthetic", "length": int(series.size)}

    info = {
        "status": "actual",
        "length": int(series.size),
        "starting": float(series.iloc[0]),
        "ending": float(series.iloc[-1]),
    }
    if series.iloc[0] != 0:
        info["return_pct"] = float(series.iloc[-1] / series.iloc[0] - 1.0)
    return series, info


def _clv_summary(df: pd.DataFrame, show_books: set[str]) -> tuple[pd.Series, dict[str, Any]]:
    if df is None or df.empty:
        return pd.Series(dtype=float), {"status": "missing"}

    working = df.copy()
    if show_books:
        working = working[working["book"].isin(show_books)]
    clv_series = _safe_numeric_series(working, ["clv_bps", "edge_bps", "edge"])
    if clv_series.empty:
        return clv_series, {"status": "missing"}

    per_book = (
        working.assign(clv_bps=pd.to_numeric(working.get("clv_bps"), errors="coerce"))
        .dropna(subset=["clv_bps"])
        .groupby("book")
        .agg(mean_bps=("clv_bps", "mean"), median_bps=("clv_bps", "median"), count=("clv_bps", "size"))
        .reset_index()
    )

    info: dict[str, Any] = {
        "status": "ok",
        "count": int(clv_series.size),
        "mean_bps": float(clv_series.mean()),
        "median_bps": float(clv_series.median()),
        "per_book": [
            {
                "book": str(row["book"]),
                "mean_bps": float(row["mean_bps"]),
                "median_bps": float(row["median_bps"]),
                "count": int(row["count"]),
            }
            for row in per_book.to_dict("records")
        ],
    }
    return clv_series, info


def _impact_summary(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None or df.empty:
        return None
    working = df.copy()
    if "delta_win_pct" not in working.columns:
        return None

    def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            records.append(
                {
                    "player_id": str(getattr(row, "player_id", "")),
                    "team_id": None if pd.isna(getattr(row, "team_id", None)) else str(getattr(row, "team_id", "")),
                    "position": None if pd.isna(getattr(row, "position", None)) else str(getattr(row, "position", "")),
                    "delta_win_pct": float(getattr(row, "delta_win_pct", 0.0)),
                }
            )
        return records

    top = working.sort_values("delta_win_pct", ascending=False).head(5)
    bottom = working.sort_values("delta_win_pct", ascending=True).head(5)
    return {
        "top": _records(top),
        "bottom": _records(bottom),
        "count": int(len(working)),
    }


def main() -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="[reports] %(message)s")
    random.seed(1729)
    np.random.seed(1729)

    config = _load_config()
    reports_cfg = config.get("reports", {}) if isinstance(config, dict) else {}
    calibrate_cfg = config.get("calibrate", {}) if isinstance(config, dict) else {}
    portfolio_cfg = config.get("portfolio", {}) if isinstance(config, dict) else {}

    out_dir = _ensure_dir(reports_out_dir())

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "artifacts": {},
    }

    bins = int(calibrate_cfg.get("bins", 10)) if isinstance(calibrate_cfg, dict) else 10

    meta_df, meta_path = load_latest_parquet_or_csv("artifacts/meta/final_probs_*.parquet")
    if meta_path:
        summary["artifacts"]["meta_final_probs"] = meta_path.name

    calib_json, calib_path = load_latest_json("artifacts/meta/calibration_report_*.json")
    if calib_path:
        summary["artifacts"]["calibration_report"] = calib_path.name

    calibration_payload = _calibration_payload(meta_df, calib_json, bins)
    calibration_json_path = out_dir / "calibration.json"
    calibration_json_path.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")
    summary["calibration"] = calibration_payload

    calibration_bins = calibration_payload.get("bins", [])
    calibration_png = out_dir / "calibration.png"
    _plot_reliability(calibration_png, calibration_bins, "A22A — Calibration")

    portfolio_df, portfolio_path = load_latest_parquet_or_csv("artifacts/portfolio/picks_week_*.parquet")
    if portfolio_path:
        summary["artifacts"]["portfolio_picks"] = portfolio_path.name
    portfolio_df = portfolio_df if portfolio_df is not None else pd.DataFrame()
    portfolio_metrics = _portfolio_metrics(portfolio_df.copy(), portfolio_cfg)
    summary["portfolio"] = portfolio_metrics

    if not portfolio_df.empty:
        value_col = "stake_amount" if "stake_amount" in portfolio_df.columns else "stake_pct"
        plot_df = portfolio_df.copy()
        if "side" not in plot_df.columns:
            plot_df["side"] = "BET"
        plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce").fillna(0.0)
        plot_df = plot_df[["game_id", "side", value_col]]
    else:
        plot_df = pd.DataFrame(columns=["game_id", "side", "stake_pct"])
        value_col = "stake_pct"
    exposure_png = out_dir / "portfolio_exposure.png"
    _plot_portfolio_exposure(exposure_png, plot_df, value_col, "Portfolio Exposure")

    bankroll_df, bankroll_path = load_latest_parquet_or_csv("artifacts/portfolio/bankroll_*.parquet")
    if bankroll_path:
        summary["artifacts"]["bankroll_history"] = bankroll_path.name
    bankroll_series, bankroll_info = _bankroll_series(bankroll_df)
    summary["bankroll"] = bankroll_info
    bankroll_png = out_dir / "bankroll_curve.png"
    _plot_bankroll(bankroll_png, bankroll_series, "Portfolio Bankroll")

    clv_df, clv_path = load_latest_parquet_or_csv("artifacts/market/clv_*.parquet")
    if clv_path:
        summary["artifacts"]["market_clv"] = clv_path.name
    clv_df = clv_df if clv_df is not None else pd.DataFrame()
    show_books = set(reports_cfg.get("show_books", [])) if isinstance(reports_cfg, dict) else set()
    clv_series, clv_info = _clv_summary(clv_df, show_books)
    summary["clv"] = clv_info
    clv_png = out_dir / "clv_distribution.png"
    if clv_series.empty:
        _plot_reliability(clv_png, _synthetic_calibration_bins(), "CLV Distribution")
    else:
        _plot_histogram(clv_png, clv_series, "CLV Distribution", "CLV (bps)")

    impact_df, impact_path = load_latest_parquet_or_csv("artifacts/impact/player_impact_*.parquet")
    if impact_path:
        summary["artifacts"]["player_impact"] = impact_path.name
    impact_summary = _impact_summary(impact_df if impact_df is not None else None)
    if impact_summary:
        summary["impact"] = impact_summary

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("summary written to %s", summary_path.relative_to(pathlib.Path(".")))
    return summary


if __name__ == "__main__":
    main()
