"""Closing line value computation."""

from __future__ import annotations

import pathlib
import time
from typing import Any, Dict

import pandas as pd
import yaml

from a22a.metrics.market import basis_points_delta, implied_from_american

CONFIG_PATH = pathlib.Path("configs/defaults.yaml")
ARTIFACT_DIR = pathlib.Path("artifacts/market")


def _load_config(path: pathlib.Path = CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_clv_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    clv_cfg = dict(cfg.get("clv", {}))
    return {
        "use_closing": bool(clv_cfg.get("use_closing", True)),
        "min_snapshots_per_game": int(clv_cfg.get("min_snapshots_per_game", 2)),
    }


def _latest_snapshot() -> pathlib.Path | None:
    if not ARTIFACT_DIR.exists():
        return None
    candidates = sorted(ARTIFACT_DIR.glob("snapshots_*.parquet"))
    if candidates:
        return candidates[-1]
    candidates = sorted(ARTIFACT_DIR.glob("snapshots_*.csv"))
    return candidates[-1] if candidates else None


def _load_snapshot(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        return df
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def _ensure_implied(df: pd.DataFrame) -> pd.DataFrame:
    if "implied_prob" in df.columns:
        return df
    df = df.copy()
    df["implied_prob"] = implied_from_american(df["price"].to_numpy())
    return df


def _compute_clv(df: pd.DataFrame, *, min_snapshots: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouped = df.groupby(["provider", "game_id", "book", "market", "selection"], dropna=False)
    for (provider, game_id, book, market, selection), group in grouped:
        group = group.sort_values("ts")
        if len(group) < min_snapshots:
            continue
        open_row = group.iloc[0]
        close_row = group.iloc[-1]
        clv = float(basis_points_delta([close_row["implied_prob"]], [open_row["implied_prob"]])[0])
        records.append(
            {
                "provider": provider,
                "game_id": game_id,
                "book": book,
                "market": market,
                "selection": selection,
                "open_price": float(open_row["price"]),
                "close_price": float(close_row["price"]),
                "open_implied": float(open_row["implied_prob"]),
                "close_implied": float(close_row["implied_prob"]),
                "open_ts": open_row["ts"],
                "close_ts": close_row["ts"],
                "snapshots": int(len(group)),
                "synthetic": bool(group.get("synthetic", pd.Series([False])).all()),
                "clv_bps": clv,
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "provider",
                "game_id",
                "book",
                "market",
                "selection",
                "open_price",
                "close_price",
                "open_implied",
                "close_implied",
                "open_ts",
                "close_ts",
                "snapshots",
                "synthetic",
                "clv_bps",
            ]
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    cfg = _load_config()
    clv_cfg = _resolve_clv_config(cfg)
    snapshot_path = _latest_snapshot()
    if snapshot_path is None:
        print("[clv] no snapshots found; run `make market` first")
        return
    df = _load_snapshot(snapshot_path)
    df = _ensure_implied(df)
    clv_df = _compute_clv(df, min_snapshots=clv_cfg["min_snapshots_per_game"])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = ARTIFACT_DIR / f"clv_{stamp}.parquet"
    clv_df.to_parquet(out_path, index=False)
    print(f"[clv] wrote {out_path.name} rows={len(clv_df)} from={snapshot_path.name}")


if __name__ == "__main__":
    main()
