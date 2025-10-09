"""Historical backtesting harness for the A22A portfolio."""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import time
from dataclasses import replace
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml

from a22a.metrics.calibration import brier_score, ece
from a22a.portfolio.optimize import (
    PCfg,
    apply_correlation_guard,
    build_weekly_slate,
    load_config as load_portfolio_config,
    load_final_probabilities,
    load_sim_summary,
    size_portfolio,
)
from a22a.store import MetricsStore

DEFAULT_CONFIG_PATH = "configs/defaults.yaml"
ARTIFACT_DIR = pathlib.Path("artifacts/backtest")


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.timezone.utc)


def _load_config(path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    cfg_path = pathlib.Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_weeks(weeks_cfg: Iterable[int] | str | None) -> list[int]:
    if isinstance(weeks_cfg, str):
        if weeks_cfg.lower() == "all":
            return list(range(1, 19))
        return []
    if weeks_cfg is None:
        return [1, 2, 3]
    return [int(w) for w in weeks_cfg]


def _week_seed(season: int, week: int) -> int:
    return season * 100 + week


def _synthetic_week(season: int, week: int, n_games: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(_week_seed(season, week))
    probs = np.clip(0.45 + 0.12 * rng.normal(size=n_games), 0.05, 0.95)
    df = pd.DataFrame(
        {
            "game_id": [f"{season}W{week:02d}_{i:02d}" for i in range(n_games)],
            "p_home": probs,
        }
    )
    df["p_away"] = 1.0 - df["p_home"]
    return df


def _weekly_probabilities(
    base_df: pd.DataFrame | None,
    season: int,
    week: int,
    slate_size: int = 8,
) -> pd.DataFrame:
    if base_df is None or base_df.empty:
        return _synthetic_week(season, week, n_games=slate_size)

    rng = np.random.default_rng(_week_seed(season, week))
    sample_size = min(len(base_df), slate_size)
    if sample_size <= 0:
        return _synthetic_week(season, week, n_games=slate_size)
    indices = rng.choice(len(base_df), size=sample_size, replace=False)
    subset = base_df.iloc[indices].copy().reset_index(drop=True)
    if "p_away" not in subset.columns:
        subset["p_away"] = 1.0 - subset["p_home"]
    return subset


def _simulate_outcomes(picks: pd.DataFrame, bankroll_start: float, seed: int) -> dict[str, Any]:
    if picks.empty:
        return {
            "bankroll_curve": [float(bankroll_start)],
            "picks": picks,
            "wins": 0,
            "profit": 0.0,
            "stake": 0.0,
            "clv_sum": 0.0,
        }

    rng = np.random.default_rng(seed)
    picks = picks.copy()
    picks["prob_win"] = picks.get("prob_pick", picks.get("p_home", 0.5)).astype(float).clip(0.05, 0.95)

    skill_noise = rng.normal(0.0, 0.05, size=len(picks))
    true_probs = np.clip(picks["prob_win"].to_numpy() + skill_noise, 0.01, 0.99)
    outcomes = rng.random(len(picks)) < true_probs
    picks["outcome"] = outcomes.astype(int)

    closing_noise = rng.normal(0.0, 0.02, size=len(picks))
    picks["close_prob"] = np.clip(picks["prob_win"].to_numpy() + closing_noise, 0.01, 0.99)
    picks["clv_bps"] = (picks["close_prob"] - picks["prob_win"]) * 10000.0

    stake = picks.get("stake_amount", pd.Series(dtype=float)).astype(float)
    profit = np.where(picks["outcome"] == 1, stake, -stake)
    picks["profit"] = profit

    bankroll_curve = [float(bankroll_start)]
    running = bankroll_start
    for delta in profit:
        running += float(delta)
        bankroll_curve.append(float(running))

    return {
        "bankroll_curve": bankroll_curve,
        "picks": picks,
        "wins": int(picks["outcome"].sum()),
        "profit": float(picks["profit"].sum()),
        "stake": float(stake.sum()),
        "clv_sum": float(picks["clv_bps"].sum()),
    }


def _week_to_date(season: int, week: int) -> _dt.date:
    base = _dt.date(int(season), 1, 1)
    return base + _dt.timedelta(days=(week - 1) * 7)


def _season_summary(weeks: list[dict[str, Any]]) -> dict[str, float]:
    if not weeks:
        return {
            "roi": 0.0,
            "win_pct": 0.0,
            "ece": 0.0,
            "brier": 0.0,
            "clv_bps_mean": 0.0,
        }
    total_profit = sum(week["totals"]["profit"] for week in weeks)
    total_stake = sum(week["totals"]["stake"] for week in weeks)
    total_wins = sum(week["totals"]["wins"] for week in weeks)
    total_bets = sum(week["totals"]["bets"] for week in weeks)
    ece_weighted = sum(week["calibration"]["ece"] * week["totals"]["bets"] for week in weeks)
    brier_weighted = sum(week["calibration"]["brier"] * week["totals"]["bets"] for week in weeks)
    clv_sum = sum(week["totals"]["clv_sum"] for week in weeks)

    roi = total_profit / total_stake if total_stake else 0.0
    win_pct = total_wins / total_bets if total_bets else 0.0
    avg_ece = ece_weighted / total_bets if total_bets else 0.0
    avg_brier = brier_weighted / total_bets if total_bets else 0.0
    clv_mean = clv_sum / total_bets if total_bets else 0.0

    return {
        "roi": float(roi),
        "win_pct": float(win_pct),
        "ece": float(avg_ece),
        "brier": float(avg_brier),
        "clv_bps_mean": float(clv_mean),
    }


def _aggregate_summary(seasons: list[dict[str, Any]]) -> dict[str, float]:
    weeks = [week for season in seasons for week in season.get("weeks", [])]
    return _season_summary(weeks)


def run_backtest(config_path: str = DEFAULT_CONFIG_PATH) -> Tuple[dict[str, Any], pathlib.Path]:
    start = time.perf_counter()

    config = _load_config(config_path)
    backtest_cfg = config.get("backtest", {}) or {}
    store_cfg = config.get("store", {}) or {}

    seasons = [int(s) for s in backtest_cfg.get("seasons", [])]
    if not seasons:
        seasons = [2023]
    weeks = _resolve_weeks(backtest_cfg.get("weeks"))
    bankroll_start = float(backtest_cfg.get("bankroll_start", 100_000))

    prob_df, prob_path = load_final_probabilities()
    sim_summary = load_sim_summary()
    pcfg: PCfg = load_portfolio_config()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_store = MetricsStore(store_cfg.get("duckdb_path", "./artifacts/store/a22a_metrics.duckdb"))

    seasons_payload: list[dict[str, Any]] = []
    store_rows: list[dict[str, Any]] = []

    for season in seasons:
        bankroll = bankroll_start
        weekly_payload: list[dict[str, Any]] = []
        for week in weeks:
            weekly_probs = _weekly_probabilities(prob_df, season, week)
            slate = build_weekly_slate(weekly_probs, sim_summary)
            guarded = apply_correlation_guard(slate, pcfg)
            week_cfg = replace(pcfg, bankroll=bankroll)
            sized = size_portfolio(guarded, week_cfg)
            active = sized.loc[sized.get("stake_amount", 0) > 0].reset_index(drop=True)

            simulation = _simulate_outcomes(active, bankroll, _week_seed(season, week))
            picks = simulation["picks"]
            bankroll_curve = simulation["bankroll_curve"]
            bankroll = bankroll_curve[-1]

            wins = simulation["wins"]
            stake_total = simulation["stake"]
            profit_total = simulation["profit"]
            bets = len(picks)
            roi_val = profit_total / stake_total if stake_total else 0.0
            win_pct = wins / bets if bets else 0.0
            unit_return = profit_total / bets if bets else 0.0

            if bets:
                bins = max(4, min(10, bets))
                ece_val = float(ece(picks["prob_win"], picks["outcome"], bins=bins))
                brier_val = float(brier_score(picks["prob_win"], picks["outcome"]))
                clv_mean = float(picks["clv_bps"].mean())
                clv_median = float(picks["clv_bps"].median())
                clv_positive = float((picks["clv_bps"] > 0).mean())
            else:
                ece_val = 0.0
                brier_val = 0.0
                clv_mean = 0.0
                clv_median = 0.0
                clv_positive = 0.0

            week_payload = {
                "season": season,
                "week": week,
                "bankroll_start": float(bankroll_curve[0]),
                "bankroll_end": float(bankroll_curve[-1]),
                "bankroll_curve": bankroll_curve,
                "n_bets": bets,
                "win_pct": win_pct,
                "roi": roi_val,
                "unit_return": unit_return,
                "clv": {
                    "mean_bps": clv_mean,
                    "median_bps": clv_median,
                    "positive_rate": clv_positive,
                },
                "calibration": {
                    "ece": ece_val,
                    "brier": brier_val,
                },
                "totals": {
                    "wins": wins,
                    "bets": bets,
                    "stake": stake_total,
                    "profit": profit_total,
                    "clv_sum": simulation["clv_sum"],
                },
                "picks": picks.to_dict(orient="records"),
            }

            weekly_payload.append(week_payload)

            store_rows.append(
                {
                    "date": _week_to_date(season, week),
                    "season": season,
                    "week": week,
                    "n_bets": bets,
                    "roi": roi_val,
                    "win_pct": win_pct,
                    "ece": ece_val,
                    "clv_bps_mean": clv_mean,
                    "bankroll": bankroll_curve[-1],
                }
            )

        season_summary = _season_summary(weekly_payload)
        seasons_payload.append(
            {
                "season": season,
                "weeks": weekly_payload,
                "summary": season_summary,
            }
        )

    metrics_store.append_backtest_rows(store_rows)

    aggregate = _aggregate_summary(seasons_payload)

    now_utc = _utcnow()
    runtime_s = time.perf_counter() - start
    payload = {
        "generated_at": now_utc.isoformat().replace("+00:00", "Z"),
        "runtime_s": runtime_s,
        "config": {
            "seasons": seasons,
            "weeks": weeks,
            "bankroll_start": bankroll_start,
            "probabilities_source": prob_path.name if prob_path else "synthetic",
        },
        "seasons": seasons_payload,
        "aggregate": aggregate,
    }

    timestamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
    output_path = ARTIFACT_DIR / f"summary_{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    summary_line = \
        f"seasons={seasons} weeks={weeks} bankroll_start={bankroll_start:.2f} runtime={runtime_s:.2f}s"
    print(f"[backtest] {summary_line}")
    print(f"[backtest] aggregate={aggregate}")
    print(f"[backtest] wrote {output_path}")

    return payload, output_path


def main() -> None:
    run_backtest()


if __name__ == "__main__":  # pragma: no cover
    main()

