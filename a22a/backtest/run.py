"""Phase 19 backtester that simulates historical slates and appends metrics."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import yaml

from a22a.reports.sources import load_latest_parquet_or_csv
from a22a.store.metrics_store import MetricsStore, ensure_store

from . import metrics as metrics_lib

DEFAULT_CONFIG_PATH = pathlib.Path("configs/defaults.yaml")
ARTIFACT_DIR = pathlib.Path("artifacts/backtest")
SUMMARY_PREFIX = "summary_"
DEFAULT_SEED = 20241019


@dataclass(slots=True)
class Bet:
    game_id: str
    selection: str
    stake: float
    open_prob: float
    close_prob: float
    outcome: int

    def payout(self) -> float:
        price = max(self.open_prob, 1e-6)
        decimal_odds = 1.0 / price
        return self.stake * decimal_odds if self.outcome else 0.0

    def profit(self) -> float:
        return self.payout() - self.stake


def _load_config(path: pathlib.Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_weeks(weeks: Iterable[int] | str | None) -> List[int]:
    if weeks is None:
        return [1]
    if isinstance(weeks, str):
        if weeks.lower() == "all":
            return list(range(1, 19))
        return [int(weeks)]
    return [int(w) for w in weeks]


def _load_artifact_df(glob: str) -> pd.DataFrame | None:
    df, _ = load_latest_parquet_or_csv(glob)
    return df


def _normalise_selection(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"home", "h", "1"}:
        return "HOME"
    if text in {"away", "a", "2"}:
        return "AWAY"
    return "HOME"


def _build_clv_map(clv_df: pd.DataFrame | None) -> Dict[Tuple[str, str], float]:
    if clv_df is None or clv_df.empty:
        return {}
    working = clv_df.copy()
    if "selection" not in working.columns:
        working["selection"] = "HOME"
    if "close_prob" not in working.columns:
        return {}
    working["selection"] = working["selection"].apply(_normalise_selection)
    working["close_prob"] = pd.to_numeric(working["close_prob"], errors="coerce")
    working = working.dropna(subset=["close_prob", "game_id"])
    by_key: Dict[Tuple[str, str], float] = {}
    for row in working.itertuples(index=False):
        key = (str(getattr(row, "game_id")), str(getattr(row, "selection")))
        close_prob = float(getattr(row, "close_prob", math.nan))
        if math.isnan(close_prob):
            continue
        by_key[key] = close_prob
    return by_key


def _make_seed(season: int, week: int, base: int = DEFAULT_SEED) -> int:
    key = f"{base}-{season}-{week}".encode("utf-8")
    return int.from_bytes(hashlib.blake2s(key, digest_size=4).digest(), "big")


def _deterministic_outcome(prob: float, *, season: int, week: int, game_id: str, selection: str) -> int:
    key = f"{season}-{week}-{game_id}-{selection}-{prob:.6f}".encode("utf-8")
    hashed = hashlib.blake2s(key, digest_size=8).digest()
    draw = int.from_bytes(hashed, "big") / float(1 << 64)
    return 1 if draw < max(0.0, min(1.0, prob)) else 0


def _extract_open_prob(row: pd.Series) -> float:
    selection = _normalise_selection(row.get("side"))
    if selection == "HOME":
        return float(pd.to_numeric(row.get("p_home"), errors="coerce") or 0.5)
    return float(pd.to_numeric(row.get("p_away"), errors="coerce") or 0.5)


def _resolve_stake(row: pd.Series, bankroll: float) -> float:
    stake_amount = pd.to_numeric(row.get("stake_amount"), errors="coerce")
    if pd.notna(stake_amount) and float(stake_amount) > 0:
        return float(stake_amount)
    stake_pct = pd.to_numeric(row.get("stake_pct"), errors="coerce")
    if pd.notna(stake_pct) and float(stake_pct) > 0:
        return float(bankroll * float(stake_pct))
    return float(bankroll * 0.01)


def _choose_rows(df: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    count = min(len(df), max(1, count))
    rng = pd.Series(range(len(df)))
    indices = rng.sample(n=count, replace=False, random_state=seed).tolist()
    return df.iloc[indices].reset_index(drop=True)


def _synthetic_picks(meta_df: pd.DataFrame | None, n: int, bankroll: float, season: int, week: int) -> pd.DataFrame:
    if meta_df is None or meta_df.empty:
        rows = []
        for idx in range(n):
            prob = 0.5 + 0.1 * math.sin((season + week + idx) * 0.7)
            rows.append(
                {
                    "game_id": f"SYN_{season}_{week}_{idx:02d}",
                    "side": "HOME" if idx % 2 == 0 else "AWAY",
                    "p_home": prob,
                    "p_away": 1 - prob,
                    "stake_amount": bankroll * 0.01,
                }
            )
        return pd.DataFrame(rows)

    working = meta_df.copy()
    working["side"] = ["HOME" if i % 2 == 0 else "AWAY" for i in range(len(working))]
    working["stake_amount"] = bankroll * 0.01
    return _choose_rows(working, n, seed=_make_seed(season, week))


def _generate_bets(
    season: int,
    week: int,
    bankroll: float,
    picks_df: pd.DataFrame | None,
    meta_df: pd.DataFrame | None,
    clv_map: Dict[Tuple[str, str], float],
    n_bets: int,
) -> List[Bet]:
    base = picks_df if picks_df is not None and not picks_df.empty else None
    if base is not None:
        active = base.copy()
        if "stake_amount" in active.columns:
            active = active[pd.to_numeric(active.get("stake_amount"), errors="coerce").fillna(0) > 0]
        if active.empty and "stake_pct" in base.columns:
            active = base[pd.to_numeric(base.get("stake_pct"), errors="coerce").fillna(0) > 0]
        if active.empty:
            active = base.copy()
        chosen = _choose_rows(active, n_bets, seed=_make_seed(season, week))
    else:
        chosen = _synthetic_picks(meta_df, n_bets, bankroll, season, week)

    bets: List[Bet] = []
    if chosen.empty:
        return bets

    for row in chosen.itertuples(index=False):
        series = pd.Series(row._asdict())
        selection = _normalise_selection(series.get("side"))
        game_id = str(series.get("game_id", f"SYN_{season}_{week}_{len(bets):02d}"))
        selection_prob = _extract_open_prob(series)
        open_prob = max(1e-3, min(1.0 - 1e-3, selection_prob))
        close_prob = clv_map.get((game_id, selection), open_prob)
        stake = max(1.0, _resolve_stake(series, bankroll))
        outcome = _deterministic_outcome(
            open_prob,
            season=season,
            week=week,
            game_id=game_id,
            selection=selection,
        )
        bet = Bet(
            game_id=game_id,
            selection=selection,
            stake=stake,
            open_prob=open_prob,
            close_prob=close_prob,
            outcome=outcome,
        )
        bets.append(bet)
    return bets


def _bankroll_curve(start: float, bets: Sequence[Bet]) -> List[float]:
    bankroll = start
    curve = [float(bankroll)]
    for bet in bets:
        bankroll -= bet.stake
        bankroll += bet.payout()
        curve.append(float(bankroll))
    return curve


def _summarise_week(bets: Sequence[Bet], bankroll_start: float) -> dict[str, Any]:
    if not bets:
        return {
            "bets": 0,
            "roi": 0.0,
            "win_pct": 0.0,
            "ece": 0.0,
            "clv_bps_mean": 0.0,
            "max_drawdown": 0.0,
            "herfindahl": 0.0,
            "bankroll": bankroll_start,
            "unit_return": 0.0,
            "positive_clv_pct": 0.0,
        }

    payouts = [bet.payout() for bet in bets]
    stakes = [bet.stake for bet in bets]
    outcomes = [bet.outcome for bet in bets]
    open_probs = [bet.open_prob for bet in bets]
    close_probs = [bet.close_prob for bet in bets]

    roi = metrics_lib.roi(payouts, stakes)
    win_pct = metrics_lib.win_rate(sum(outcomes), len(outcomes))
    ece = metrics_lib.expected_calibration_error(open_probs, outcomes)
    clv_bps = metrics_lib.clv_basis_points(open_probs, close_probs)
    curve = _bankroll_curve(bankroll_start, bets)
    drawdown = metrics_lib.max_drawdown(curve)
    weights = [stake / bankroll_start for stake in stakes if bankroll_start]
    herfindahl = metrics_lib.herfindahl_index(weights)
    unit_return = sum(bet.profit() for bet in bets) / len(bets)
    positive_clv_pct = float(sum(1 for o, c in zip(open_probs, close_probs) if c > o) / len(bets))

    return {
        "bets": len(bets),
        "roi": roi,
        "win_pct": win_pct,
        "ece": ece,
        "clv_bps_mean": clv_bps,
        "max_drawdown": drawdown,
        "herfindahl": herfindahl,
        "bankroll": curve[-1],
        "unit_return": unit_return,
        "positive_clv_pct": positive_clv_pct,
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
    bets_per_week = int(backtest_cfg.get("bets_per_week", 8))

    meta_df = _load_artifact_df("artifacts/meta/final_probs_*.parquet")
    picks_df = _load_artifact_df("artifacts/portfolio/picks_week_*.parquet")
    clv_df = _load_artifact_df("artifacts/market/clv_*.parquet")

    store_path = pathlib.Path(store_cfg.get("duckdb_path", "./artifacts/store/a22a_metrics.duckdb"))
    ensure_store(store_path)
    clv_map = _build_clv_map(clv_df)

    per_week: List[dict[str, Any]] = []
    global_payouts: List[float] = []
    global_stakes: List[float] = []
    global_outcomes: List[int] = []
    global_open_probs: List[float] = []
    global_close_probs: List[float] = []

    running_bankroll = bankroll_start
    for season in seasons:
        for week in weeks:
            bets = _generate_bets(
                season=season,
                week=week,
                bankroll=running_bankroll,
                picks_df=picks_df,
                meta_df=meta_df,
                clv_map=clv_map,
                n_bets=bets_per_week,
            )
            summary = _summarise_week(bets, running_bankroll)
            summary.update({"season": season, "week": week})
            per_week.append(summary)
            _append_store(summary, store_path)

            global_payouts.extend(bet.payout() for bet in bets)
            global_stakes.extend(bet.stake for bet in bets)
            global_outcomes.extend(bet.outcome for bet in bets)
            global_open_probs.extend(bet.open_prob for bet in bets)
            global_close_probs.extend(bet.close_prob for bet in bets)
            running_bankroll = float(summary.get("bankroll", running_bankroll))

    total_roi = metrics_lib.roi(global_payouts, global_stakes)
    total_win_pct = metrics_lib.win_rate(sum(global_outcomes), len(global_outcomes)) if global_outcomes else 0.0
    total_ece = metrics_lib.expected_calibration_error(global_open_probs, global_outcomes) if global_outcomes else 0.0
    total_clv = metrics_lib.clv_basis_points(global_open_probs, global_close_probs) if global_close_probs else 0.0
    total_bankroll = running_bankroll

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons,
        "weeks": weeks,
        "results": per_week,
        "totals": {
            "bets": int(sum(item.get("bets", 0) for item in per_week)),
            "roi": total_roi,
            "win_pct": total_win_pct,
            "ece": total_ece,
            "clv_bps_mean": total_clv,
            "bankroll_end": total_bankroll,
        },
        "artifacts": {
            "meta_final_probs": bool(meta_df is not None and not meta_df.empty),
            "portfolio_picks": bool(picks_df is not None and not picks_df.empty),
            "market_clv": bool(clv_df is not None and not clv_df.empty),
        },
        "timings": {"runtime_s": round(time.time() - start, 3)},
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_path = ARTIFACT_DIR / f"{SUMMARY_PREFIX}{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"[backtest] seasons={len(seasons)} weeks={len(weeks)} rows={len(per_week)} runtime={time.time() - start:.2f}s"
    )
    print(f"[backtest] wrote {output_path.relative_to(pathlib.Path('.'))}")
    return payload, output_path


def main() -> None:
    run_backtest()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
