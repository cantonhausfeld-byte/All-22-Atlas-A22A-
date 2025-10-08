"""Closing line value computation."""

from __future__ import annotations

import math
import pathlib
import time
from typing import Any, Dict

import pandas as pd
import yaml

from a22a.metrics.market import basis_points_delta, implied_from_american, pairwise_remove_vig

CONFIG_PATH = pathlib.Path("configs/defaults.yaml")
ARTIFACT_DIR = pathlib.Path("artifacts/market")

MARKET_SIDES = {
    "h2h": ("home", "away"),
    "spreads": ("home", "away"),
    "totals": ("over", "under"),
}

SPREAD_SIGMA = 13.5
TOTAL_SIGMA = 9.0
TOTAL_BASELINE = 45.0


def _load_config(path: pathlib.Path = CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _latest_snapshot() -> pathlib.Path | None:
    if not ARTIFACT_DIR.exists():
        return None
    candidates = sorted(ARTIFACT_DIR.glob("snapshots_*.parquet"))
    if candidates:
        return candidates[-1]
    return None


def _load_snapshot(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _spread_probability(line: float, selection: str) -> float:
    if selection == "home":
        return _norm_cdf(-float(line) / SPREAD_SIGMA)
    if selection == "away":
        return _norm_cdf(float(line) / SPREAD_SIGMA)
    return math.nan


def _total_probability(line: float, selection: str) -> float:
    z = (float(line) - TOTAL_BASELINE) / TOTAL_SIGMA
    if selection == "over":
        return 1.0 - _norm_cdf(z)
    if selection == "under":
        return _norm_cdf(z)
    return math.nan


def _blend_prob(price_prob: float, model_prob: float, weight: float = 0.65) -> float:
    if not math.isfinite(model_prob):
        return float(price_prob)
    price_prob = float(price_prob)
    blended = weight * price_prob + (1.0 - weight) * model_prob
    return float(min(max(blended, 0.0), 1.0))


def _compute_fair_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df["fair_prob"] = pd.Series(dtype=float)
        return df
    df = df.copy()
    df["price_prob"] = implied_from_american(df["price"].to_numpy())
    df["fair_prob"] = df["price_prob"].astype(float)

    grouped = df.groupby(["game_id", "book", "market", "ts"], dropna=False)
    for (_, _, market, _), group in grouped:
        sides = MARKET_SIDES.get(market)
        if not sides:
            continue
        if not all(side in group["selection"].values for side in sides):
            continue
        raw = [
            float(group.loc[group["selection"] == side, "price_prob"].iloc[0])
            for side in sides
        ]
        adjusted = pairwise_remove_vig(raw)
        for side, prob in zip(sides, adjusted, strict=False):
            idx = group.index[group["selection"] == side]
            df.loc[idx, "fair_prob"] = float(prob)

    for idx, row in df.iterrows():  # small tables, explicit loop acceptable
        market = row.get("market")
        selection = row.get("selection")
        line = float(row.get("line", 0.0) or 0.0)
        model_prob = math.nan
        if market == "spreads":
            model_prob = _spread_probability(line, str(selection))
        elif market == "totals":
            model_prob = _total_probability(line, str(selection))
        if math.isfinite(model_prob):
            df.at[idx, "fair_prob"] = _blend_prob(df.at[idx, "fair_prob"], model_prob)

    return df


def _compute_clv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "book",
                "market",
                "selection",
                "open_price",
                "close_price",
                "open_line",
                "close_line",
                "open_prob",
                "close_prob",
                "clv_bps",
                "synthetic",
            ]
        )

    df = df.sort_values("ts")
    records: list[dict[str, Any]] = []
    grouped = df.groupby(["game_id", "book", "market", "selection"], dropna=False)

    for (game_id, book, market, selection), group in grouped:
        group = group.sort_values("ts")
        if group["fair_prob"].isna().all():
            continue
        open_row = group.iloc[0]
        close_row = group.iloc[-1]
        open_prob = float(open_row["fair_prob"])
        close_prob = float(close_row["fair_prob"])
        clv = float(basis_points_delta([close_prob], [open_prob])[0])
        synthetic_flag = bool(group["synthetic"].all()) if "synthetic" in group.columns else False
        records.append(
            {
                "game_id": game_id,
                "book": book,
                "market": market,
                "selection": selection,
                "open_price": float(open_row["price"]),
                "close_price": float(close_row["price"]),
                "open_line": float(open_row.get("line", 0.0) or 0.0),
                "close_line": float(close_row.get("line", 0.0) or 0.0),
                "open_prob": open_prob,
                "close_prob": close_prob,
                "clv_bps": clv,
                "synthetic": synthetic_flag,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "game_id",
                "book",
                "market",
                "selection",
                "open_price",
                "close_price",
                "open_line",
                "close_line",
                "open_prob",
                "close_prob",
                "clv_bps",
                "synthetic",
            ]
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    cfg = _load_config()
    snapshot_path = _latest_snapshot()
    if snapshot_path is None:
        print("[clv] no snapshots found; run `make market` first")
        return
    df = _load_snapshot(snapshot_path)
    df = _compute_fair_probabilities(df)
    clv_df = _compute_clv(df)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = ARTIFACT_DIR / f"clv_{stamp}.parquet"
    clv_df.to_parquet(out_path, index=False)
    print(f"[clv] wrote {out_path.name} rows={len(clv_df)} from={snapshot_path.name}")


if __name__ == "__main__":
    main()

