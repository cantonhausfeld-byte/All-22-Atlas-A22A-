"""Feature engineering scaffold for Phase 3."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "defaults.yaml"
FEATURE_DIR = Path("data") / "features"
REFERENCE_PATH = FEATURE_DIR / "reference.parquet"


def load_config() -> Dict[str, object]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def lazy_sources(datasets: Iterable[str]) -> Dict[str, pl.LazyFrame]:
    schemas = {
        "team_games": {"season": pl.Int64, "week": pl.Int64, "game_id": pl.Utf8, "team_id": pl.Utf8},
        "team_strength": {"season": pl.Int64, "week": pl.Int64, "team_id": pl.Utf8},
    }
    tables: Dict[str, pl.LazyFrame] = {}
    for name in datasets:
        tables[name] = pl.LazyFrame(schema=schemas.get(name, {}))
    return tables


def leakage_guard(frame: pl.DataFrame, required: Sequence[str]) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Leakage guard failed; missing columns -> {sorted(missing)}")
    duplicates = frame.group_by(["game_id", "team_id"]).len()
    if not duplicates.filter(pl.col("len") > 1).is_empty():
        raise ValueError("Leakage guard failed; duplicate game/team rows detected")


def compute_psi(reference: pl.DataFrame, current: pl.DataFrame, bins: int) -> Dict[str, float]:
    if reference.is_empty() or current.is_empty():
        return {}
    scores: Dict[str, float] = {}
    numeric = [column for column, dtype in current.schema.items() if dtype.is_numeric()]
    for column in numeric:
        ref = reference[column].to_numpy()
        cur = current[column].to_numpy()
        hist_ref, edges = np.histogram(ref, bins=bins)
        hist_cur, _ = np.histogram(cur, bins=edges)
        ref_pct = np.maximum(hist_ref / hist_ref.sum(), 1e-6)
        cur_pct = np.maximum(hist_cur / hist_cur.sum(), 1e-6)
        scores[column] = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return scores


def build_features(config: Dict[str, object]) -> pl.LazyFrame:
    sources = lazy_sources(["team_games", "team_strength"])
    team_games = sources["team_games"].with_columns(pl.lit(0.0).alias("margin"))
    team_strength = sources["team_strength"].with_columns(pl.lit(0.0).alias("theta_mean"))
    features = (
        team_games
        .join(team_strength, on=["season", "week"], how="left")
        .with_columns(pl.lit(0.0).alias("target"))
    )
    return features


def materialise(features: pl.LazyFrame, config: Dict[str, object]) -> Dict[str, object]:
    feature_cfg = config.get("feature_store", {})
    required = feature_cfg.get("leakage_guard_columns", [])
    bins = int(feature_cfg.get("psi_bins", 10))

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = features.collect()
    leakage_guard(frame, required)

    if REFERENCE_PATH.exists():
        reference = pl.read_parquet(REFERENCE_PATH)
        psi = compute_psi(reference, frame, bins)
    else:
        psi = {}
    frame.write_parquet(REFERENCE_PATH)
    return {"rows": frame.height, "psi": psi, "path": REFERENCE_PATH}


def main() -> None:
    config = load_config()
    features = build_features(config)
    artefacts = materialise(features, config)
    print(
        f"[features] wrote reference table to {artefacts['path']} "
        f"with {artefacts['rows']} rows"
    )
    if artefacts["psi"]:
        summary = ", ".join(f"{k}={v:.4f}" for k, v in artefacts["psi"].items())
        print(f"[features] psi drift summary -> {summary}")
    else:
        print("[features] psi drift summary -> n/a (no reference)")


if __name__ == "__main__":
    main()
