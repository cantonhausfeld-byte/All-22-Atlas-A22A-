"""Feature engineering pipeline for phases 3–5."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl

from a22a.models import team_strength as team_strength_module

STAGED_DIR = Path("staged")
FEATURE_DIR = Path("data") / "features"
REFERENCE_PATH = FEATURE_DIR / "reference.parquet"


def _ensure_team_strength() -> Path:
    output = STAGED_DIR / "team_strength.parquet"
    if not output.exists():
        team_strength_module.main()
    return output


def _scan_staged(name: str) -> pl.LazyFrame:
    path = STAGED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Expected staged dataset {path}; run `make ingest`." )
    return pl.scan_parquet(path)


def leakage_guard(features: pl.DataFrame) -> None:
    required = {"game_id", "team_id", "season", "week", "target"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Leakage guard failed: missing {sorted(missing)}")
    duplicates = features.group_by(["game_id", "team_id"]).len().filter(pl.col("len") > 1)
    if duplicates.height > 0:
        raise ValueError("Leakage guard failed: duplicate game/team rows detected")
    if (features["week"] <= 0).any():
        raise ValueError("Leakage guard failed: invalid week values detected")


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    hist_ref, edges = np.histogram(reference, bins=bins)
    hist_cur, _ = np.histogram(current, bins=edges)
    ref_pct = np.maximum(hist_ref / hist_ref.sum(), 1e-6)
    cur_pct = np.maximum(hist_cur / hist_cur.sum(), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_psi(reference: pl.DataFrame, current: pl.DataFrame) -> Dict[str, float]:
    if reference.is_empty() or current.is_empty():
        return {}
    numeric_columns = [
        column
        for column, dtype in current.schema.items()
        if dtype.is_numeric() and column not in {"season", "week", "target"}
    ]
    scores: Dict[str, float] = {}
    for column in numeric_columns:
        scores[column] = _psi(reference[column].to_numpy(), current[column].to_numpy())
    return scores


def build_feature_view() -> pl.LazyFrame:
    _ensure_team_strength()

    team_games = _scan_staged("team_games")
    pbp = _scan_staged("pbp")
    drives = _scan_staged("drives")
    team_strength = _scan_staged("team_strength")

    pbp_features = (
        pbp.with_columns(pl.col("posteam").alias("team_id"))
        .group_by(["season", "week", "game_id", "team_id"])
        .agg(
            pl.len().alias("plays"),
            pl.col("yards_gained").mean().alias("avg_yards_gained"),
            pl.col("yards_gained").max().alias("max_yards_gained"),
            pl.col("success").mean().alias("success_rate"),
            pl.col("scoring_margin").mean().alias("avg_scoring_margin_post"),
        )
    )

    drive_features = (
        drives.group_by(["season", "week", "game_id", "team_id"])
        .agg(
            pl.sum("points").alias("drive_points"),
            pl.mean("points").alias("avg_drive_points"),
            pl.sum("yards").alias("drive_yards"),
            (pl.sum("points") / pl.sum("plays")).alias("points_per_play"),
        )
        .with_columns(
            pl.col("drive_points").fill_null(0.0),
            pl.col("avg_drive_points").fill_null(0.0),
            pl.col("drive_yards").fill_null(0.0),
            pl.col("points_per_play").fill_null(0.0),
        )
    )

    lagged_team = (
        team_games.with_columns(
            pl.col("margin").shift(1).over("team_id").alias("lag_margin"),
            pl.col("win").shift(1).over("team_id").alias("lag_win"),
            pl.col("points_for").shift(1).over("team_id").alias("lag_points_for"),
            pl.col("points_against").shift(1).over("team_id").alias("lag_points_against"),
        )
        .with_columns(
            pl.col("lag_margin").fill_null(0.0),
            pl.col("lag_win").fill_null(0.0),
            pl.col("lag_points_for").fill_null(0.0),
            pl.col("lag_points_against").fill_null(0.0),
            pl.col("lag_margin").rolling_mean(window_size=3).over("team_id").fill_null(0.0).alias("rolling_margin_3"),
        )
    )

    features = (
        lagged_team
        .join(pbp_features, on=["season", "week", "game_id", "team_id"], how="left")
        .join(drive_features, on=["season", "week", "game_id", "team_id"], how="left")
        .join(team_strength, on=["season", "week", "team_id"], how="left")
    )

    opponent_strength = team_strength.rename({
        "team_id": "opponent_id",
        "theta_mean": "opponent_theta_mean",
        "theta_ci_lower": "opponent_theta_ci_lower",
        "theta_ci_upper": "opponent_theta_ci_upper",
        "samples": "opponent_strength_samples",
    })

    features = (
        features.join(opponent_strength, on=["season", "week", "opponent_id"], how="left")
        .with_columns(
            pl.col("theta_mean").fill_null(0.0),
            pl.col("opponent_theta_mean").fill_null(0.0),
            (pl.col("theta_mean") - pl.col("opponent_theta_mean")).alias("theta_diff"),
            pl.col("win").cast(pl.Float64).alias("target"),
        )
    )

    return features


def _materialize_weekly(features: pl.DataFrame) -> List[Path]:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for (season, week), frame in features.group_by(["season", "week"], maintain_order=True):
        season_dir = FEATURE_DIR / f"season={season}"
        season_dir.mkdir(parents=True, exist_ok=True)
        path = season_dir / f"week={int(week):02d}.parquet"
        frame.write_parquet(path)
        written.append(path)
    return written


def materialize(features: pl.LazyFrame) -> Dict[str, object]:
    current = features.collect()
    leakage_guard(current)
    psi_scores: Dict[str, float]
    if REFERENCE_PATH.exists():
        reference = pl.read_parquet(REFERENCE_PATH)
        psi_scores = compute_psi(reference, current)
    else:
        psi_scores = {}
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    current.write_parquet(REFERENCE_PATH)
    paths = _materialize_weekly(current)
    return {"paths": paths, "psi": psi_scores, "rows": current.height}


def main() -> None:
    STAGED_DIR.mkdir(exist_ok=True)
    features = build_feature_view()
    artefacts = materialize(features)
    psi_summary = ", ".join(f"{k}={v:.4f}" for k, v in sorted(artefacts["psi"].items())) or "n/a"
    print(f"[features] materialised {artefacts['rows']} rows -> {len(artefacts['paths'])} weekly files")
    print(f"[features] psi drift summary: {psi_summary}")


if __name__ == "__main__":
    main()
