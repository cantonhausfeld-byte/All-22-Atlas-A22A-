"""Phase 19 backtesting bootstrap.

The implementation simulates a deterministic synthetic slate so the command can
run end-to-end inside CI without relying on upstream phases. Metrics are
computed via ``a22a.backtest.metrics`` and appended to the DuckDB metrics store.
"""

from __future__ import annotations

import json
import pathlib
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, List, Sequence, Tuple

import yaml

from . import metrics as metrics_lib
from a22a.store.metrics_store import MetricsStore, ensure_store

DEFAULT_CONFIG_PATH = pathlib.Path("configs/defaults.yaml")
ARTIFACT_DIR = pathlib.Path("artifacts/backtest")
SUMMARY_PREFIX = "summary_"


@dataclass(slots=True)
class Bet:
    stake: float
    open_prob: float
    close_prob: float
    outcome: int

    def payout(self) -> float:
        implied_odds = max(self.open_prob, 1e-3)
        price = (1 / implied_odds) - 1
        return self.stake * (1 + price) if self.outcome else 0.0


def _load_config(path: pathlib.Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data or {}


def _resolve_weeks(weeks: Iterable[int] | str | None) -> List[int]:
    if weeks is None:
        return [1]
    if isinstance(weeks, str):
        if weeks.lower() == "all":
            return list(range(1, 19))
        return [1]
    return [int(w) for w in weeks]


def _generate_bets(seed: int, n_bets: int, stake: float) -> List[Bet]:
    rng = random.Random(seed)
    bets: List[Bet] = []
    for _ in range(n_bets):
        open_prob = max(0.05, min(0.95, rng.uniform(0.40, 0.65)))
        close_prob = max(0.05, min(0.95, open_prob + rng.uniform(-0.03, 0.03)))
        outcome = 1 if rng.random() < open_prob else 0
        bets.append(Bet(stake=stake, open_prob=open_prob, close_prob=close_prob, outcome=outcome))
    return bets


def _bankroll_curve(start: float, bets: Sequence[Bet]) -> List[float]:
    bankroll = start
    curve = [bankroll]
    for bet in bets:
        bankroll -= bet.stake
        bankroll += bet.payout()
        curve.append(bankroll)
    return curve


def _summarise(bets: Sequence[Bet], bankroll_start: float) -> dict[str, float]:
    payouts = [bet.payout() for bet in bets]
    stakes = [bet.stake for bet in bets]
    outcomes = [bet.outcome for bet in bets]
    open_probs = [bet.open_prob for bet in bets]
    close_probs = [bet.close_prob for bet in bets]

    roi = metrics_lib.roi(payouts, stakes)
    win_pct = metrics_lib.win_rate(sum(outcomes), len(outcomes))
    ece = metrics_lib.expected_calibration_error(open_probs, outcomes)
    clv_bps = metrics_lib.clv_basis_points(open_probs, close_probs)
    drawdown = metrics_lib.max_drawdown(_bankroll_curve(bankroll_start, bets))
    weights = [stake / bankroll_start for stake in stakes if bankroll_start]
    herfindahl = metrics_lib.herfindahl_index(weights)

    return {
        "bets": len(bets),
        "roi": roi,
        "win_pct": win_pct,
        "ece": ece,
        "clv_bps_mean": clv_bps,
        "max_drawdown": drawdown,
        "herfindahl": herfindahl,
        "coverage": min(1.0, max(outcomes) if outcomes else 0.0),
    }


def _append_store(record: dict[str, Any], store_path: pathlib.Path) -> None:
    store = MetricsStore(store_path)
    store.append(
        {
            "ts": datetime.now(timezone.utc),
            "season": record.get("season"),
            "week": record.get("week"),
            "n_bets": record.get("bets", 0),
            "win_pct": record.get("win_pct", 0.0),
            "roi": record.get("roi", 0.0),
            "ece": record.get("ece", 0.0),
            "clv_bps_mean": record.get("clv_bps_mean", 0.0),
            "drawdown": record.get("max_drawdown", 0.0),
            "herfindahl": record.get("herfindahl", 0.0),
            "bankroll": record.get("bankroll", 0.0),
        }
    )


def run_backtest(config_path: pathlib.Path | None = None) -> Tuple[dict[str, Any], pathlib.Path]:
    start = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(config_path or DEFAULT_CONFIG_PATH)
    backtest_cfg = cfg.get("backtest", {}) or {}
    store_cfg = cfg.get("store", {}) or {}

    seasons = [int(s) for s in backtest_cfg.get("seasons", [datetime.now().year])]
    weeks = _resolve_weeks(backtest_cfg.get("weeks"))
    bankroll_start = float(backtest_cfg.get("bankroll_start", 100_000))

    ensure_store(store_cfg.get("duckdb_path", "./artifacts/store/a22a_metrics.duckdb"))
    store_path = pathlib.Path(store_cfg.get("duckdb_path", "./artifacts/store/a22a_metrics.duckdb"))

    summaries: List[dict[str, Any]] = []
    for season in seasons:
        for week in weeks:
            bets = _generate_bets(seed=season * 100 + week, n_bets=8, stake=bankroll_start * 0.01)
            summary = _summarise(bets, bankroll_start)
            summary.update({
                "season": season,
                "week": week,
                "bankroll": bankroll_start + sum(bet.payout() - bet.stake for bet in bets),
            })
            summaries.append(summary)
            _append_store(summary, store_path)

    combined = {
        "seasons": seasons,
        "weeks": weeks,
        "results": summaries,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timings": {"runtime_s": round(time.time() - start, 3)},
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_path = ARTIFACT_DIR / f"{SUMMARY_PREFIX}{timestamp}.json"
    output_path.write_text(json.dumps(combined, indent=2))

    print(f"[backtest] simulated {len(summaries)} slates")
    print(f"[backtest] wrote {output_path}")
    return combined, output_path


def main() -> None:
    run_backtest()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
