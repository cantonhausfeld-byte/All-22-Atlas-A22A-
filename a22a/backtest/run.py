"""Bootstrap backtest runner for phase 19."""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import Any, Iterable, Tuple

import yaml

from a22a.store import MetricsStore

from . import metrics as metrics_lib

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


def _enumerate_weeks(weeks_cfg: Iterable[int] | str) -> list[int]:
    if isinstance(weeks_cfg, str):
        return [1, 2, 3] if weeks_cfg.lower() == "all" else []
    return list(weeks_cfg)


def _simulate_weekly_returns(seed: int) -> dict[str, list[float]]:
    # Deterministic mini-slate per season
    payouts = [110.0 + seed, 0.0, 132.0 + seed / 2, 118.0]
    stakes = [100.0, 95.0, 105.0, 102.0]
    probabilities = [0.55 + 0.01 * seed, 0.48 + 0.005 * seed, 0.6 + 0.004 * seed, 0.52]
    outcomes = [1, 0, 1, 0]
    open_prices = [0.51, 0.49, 0.52, 0.5]
    close_prices = [0.53 + 0.002 * seed, 0.5, 0.54 + 0.001 * seed, 0.51]
    returns = [p - s for p, s in zip(payouts, stakes)]
    equity_curve = []
    bankroll = 0.0
    for r in returns:
        bankroll += r
        equity_curve.append(bankroll)
    return {
        "payouts": payouts,
        "stakes": stakes,
        "probabilities": probabilities,
        "outcomes": outcomes,
        "open_prices": open_prices,
        "close_prices": close_prices,
        "returns": returns,
        "equity_curve": equity_curve,
    }


def _season_metrics(seed: int, bankroll_start: float) -> tuple[dict[str, float], float]:
    weekly = _simulate_weekly_returns(seed)
    roi = metrics_lib.roi(weekly["payouts"], weekly["stakes"])
    wins = sum(weekly["outcomes"])
    win_pct = metrics_lib.win_rate(wins, len(weekly["outcomes"]))
    ece = metrics_lib.expected_calibration_error(weekly["probabilities"], weekly["outcomes"])
    clv = metrics_lib.clv_basis_points(weekly["open_prices"], weekly["close_prices"])
    drawdown = metrics_lib.max_drawdown(weekly["equity_curve"])
    sharpe = metrics_lib.sharpe_like(weekly["returns"])
    ending_bankroll = bankroll_start * (1 + roi)
    metrics = {
        "roi": round(roi, 4),
        "win_rate": round(win_pct, 4),
        "ece": round(ece, 4),
        "clv_bps": round(clv, 2),
        "max_drawdown": round(drawdown, 4),
        "sharpe_like": round(sharpe, 4),
    }
    return metrics, ending_bankroll


def run_backtest(config_path: str = DEFAULT_CONFIG_PATH) -> Tuple[dict[str, Any], pathlib.Path]:
    config = _load_config(config_path)
    backtest_cfg = config.get("backtest", {}) or {}
    store_cfg = config.get("store", {}) or {}

    seasons = list(backtest_cfg.get("seasons", []))
    weeks = _enumerate_weeks(backtest_cfg.get("weeks", []))
    bankroll_start = float(backtest_cfg.get("bankroll_start", 100_000))
    use_conformal = bool(backtest_cfg.get("use_conformal", True))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    season_summaries = []
    metrics_store = MetricsStore(store_cfg.get("duckdb_path", "./artifacts/store/a22a_metrics.duckdb"))
    run_id = _utcnow().strftime("bt-%Y%m%d%H%M%S")

    for idx, season in enumerate(seasons):
        metrics, ending_bankroll = _season_metrics(idx, bankroll_start)
        season_summary = {
            "season": season,
            "weeks": weeks,
            "metrics": metrics,
            "bankroll_start": bankroll_start,
            "bankroll_end": round(ending_bankroll, 2),
            "use_conformal": use_conformal,
        }
        season_summaries.append(season_summary)
        metrics_store.append_metrics(
            "backtest",
            metrics,
            run_id=run_id,
            context={"season": season},
        )

    aggregate = (
        {
            key: round(
                sum(summary["metrics"][key] for summary in season_summaries) / len(season_summaries),
                4,
            )
            for key in ("roi", "win_rate", "ece", "clv_bps", "max_drawdown", "sharpe_like")
        }
        if season_summaries
        else {}
    )

    now_utc = _utcnow()
    payload = {
        "generated_at": now_utc.isoformat().replace("+00:00", "Z"),
        "run_id": run_id,
        "seasons": season_summaries,
        "aggregate": aggregate,
    }

    timestamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
    output_path = ARTIFACT_DIR / f"summary_{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"[backtest] seasons={seasons} weeks={weeks} wrote {output_path}")
    print(f"[backtest] aggregate={aggregate}")
    return payload, output_path


def main() -> None:
    run_backtest()


if __name__ == "__main__":  # pragma: no cover
    main()
