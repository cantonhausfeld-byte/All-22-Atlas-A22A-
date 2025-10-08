"""Market ingestion entrypoint."""

from __future__ import annotations

import pathlib
import time
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

from a22a.metrics.market import basis_points_delta, implied_from_american, pairwise_remove_vig

from .clients import ClientResult, collect_provider_payloads
from .normalize import NORMALIZED_COLUMNS, normalize_payload

CONFIG_PATH = pathlib.Path("configs/defaults.yaml")
ARTIFACT_DIR = pathlib.Path("artifacts/market")
META_DIR = pathlib.Path("artifacts/meta")

MARKET_SIDE_MAP: Dict[str, tuple[str, str]] = {
    "h2h": ("home", "away"),
    "spreads": ("home", "away"),
    "totals": ("over", "under"),
}


def _load_config(path: pathlib.Path = CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_market_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    market_cfg = dict(cfg.get("market", {}))
    providers_cfg = market_cfg.get("providers", {})
    providers: list[str] = [
        name
        for name, enabled in providers_cfg.items()
        if bool(enabled)
    ]
    if not providers:
        providers = ["theodds", "sgo"]
    leagues = market_cfg.get("leagues", ["NFL"])
    markets = market_cfg.get("markets", ["h2h", "spreads", "totals"])
    books = market_cfg.get(
        "books_allowlist",
        ["pinnacle", "circa", "betmgm", "fanduel", "draftkings"],
    )
    poll_window_hours = int(market_cfg.get("poll_window_hours", 168))
    rate_limit = int(market_cfg.get("rate_limit_per_min", 30))
    return {
        "providers": providers,
        "leagues": leagues,
        "markets": markets,
        "books": books,
        "poll_window_hours": poll_window_hours,
        "rate_limit_per_min": rate_limit,
    }


def _load_final_probabilities() -> pd.DataFrame:
    if not META_DIR.exists():
        return pd.DataFrame(columns=["game_id", "p_home", "p_away"])
    candidates = sorted(META_DIR.glob("final_probs_*.parquet"))
    if not candidates:
        return pd.DataFrame(columns=["game_id", "p_home", "p_away"])
    latest = candidates[-1]
    try:
        df = pd.read_parquet(latest)
    except Exception:
        return pd.DataFrame(columns=["game_id", "p_home", "p_away"])
    if "p_home" not in df.columns and "p_calibrated" in df.columns:
        df = df.rename(columns={"p_calibrated": "p_home"})
    if "p_away" not in df.columns and "p_home" in df.columns:
        df["p_away"] = (1.0 - df["p_home"]).clip(0.0, 1.0)
    expected_cols = {"game_id", "p_home", "p_away"}
    missing = expected_cols - set(df.columns)
    if missing:
        for col in missing:
            df[col] = 0.5
    return df[list(expected_cols)]


def _compute_implied(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["implied_prob"] = []
        return df
    df = df.copy()
    df["implied_raw"] = implied_from_american(df["price"].to_numpy())
    df["implied_prob"] = df["implied_raw"].astype(float)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for market, sides in MARKET_SIDE_MAP.items():
        mask = df["market"] == market
        if not mask.any():
            continue
        grouped = df.loc[mask].groupby(["game_id", "book", "ts"], sort=False)
        for _, group in grouped:
            selections = list(group["selection"])
            if not all(side in selections for side in sides):
                continue
            raw_vals = [
                group.loc[group["selection"] == side, "implied_raw"].iloc[0]
                for side in sides
            ]
            adjusted = pairwise_remove_vig(raw_vals)
            for side, prob in zip(sides, adjusted, strict=False):
                idx = group.index[group["selection"] == side]
                df.loc[idx, "implied_prob"] = float(prob)
    return df


def _apply_fair_deltas(df: pd.DataFrame, fair: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lookup = {row["game_id"]: (row["p_home"], row["p_away"]) for _, row in fair.iterrows()}
    fair_values: list[float] = []
    for _, row in df.iterrows():
        pair = lookup.get(row["game_id"]) if lookup else None
        if pair and row["selection"] in {"home", "away"}:
            fair_val = pair[0] if row["selection"] == "home" else pair[1]
        elif row["selection"] in {"home", "away"}:
            fair_val = 0.5
        else:
            fair_val = np.nan
        fair_values.append(float(fair_val) if not np.isnan(fair_val) else np.nan)
    df["fair_prob"] = fair_values
    reference = np.where(np.isnan(df["fair_prob"]), df["implied_prob"], df["fair_prob"])
    df["fair_delta_bps"] = basis_points_delta(df["implied_prob"], reference)
    df.loc[df["fair_prob"].isna(), "fair_delta_bps"] = 0.0
    return df


def _build_snapshot(results: Iterable[ClientResult]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        normalized = normalize_payload(result.provider, result.payload, synthetic=result.synthetic)
        if normalized.empty:
            continue
        frames.append(normalized)
    if not frames:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    cfg = _load_config()
    market_cfg = _resolve_market_config(cfg)
    results = collect_provider_payloads(
        market_cfg["providers"],
        leagues=market_cfg["leagues"],
        markets=market_cfg["markets"],
        books=market_cfg["books"],
        poll_window_hours=market_cfg["poll_window_hours"],
        rate_limit_per_min=market_cfg["rate_limit_per_min"],
    )
    snapshot = _build_snapshot(results)
    snapshot = _compute_implied(snapshot)
    fair_probs = _load_final_probabilities()
    snapshot = _apply_fair_deltas(snapshot, fair_probs)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = ARTIFACT_DIR / f"snapshots_{stamp}.parquet"
    snapshot.to_parquet(path, index=False)
    provider_names = ",".join(sorted({r.provider for r in results})) if results else "none"
    print(
        "[market] wrote",
        path.name,
        f"rows={len(snapshot)}",
        f"providers={provider_names}",
    )


if __name__ == "__main__":
    main()
