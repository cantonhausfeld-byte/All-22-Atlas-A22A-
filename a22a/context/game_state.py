"""Phase 10 — dynamic game context engine.

This module derives incremental game-state features on a drive-by-drive basis
from staged data and exposes a partial simulation package compatible with the
Phase 6 simulator.  The implementation emphasises deterministic, low-latency
updates suitable for in-game usage.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import polars as pl
import yaml

from a22a.data import sample_data

CONFIG_PATH = pathlib.Path("configs/defaults.yaml")
PARTIAL_SIM_KEYS = {
    "game_id",
    "team_id",
    "score_diff",
    "expected_pace",
    "fatigue",
    "momentum",
    "timeouts",
    "state_vector",
    "aggressiveness_hint",
}


# ---------------------------------------------------------------------------
# Configuration helpers


def _load_config(path: pathlib.Path = CONFIG_PATH) -> Dict[str, Any]:
    if path.exists():
        return yaml.safe_load(path.read_text())
    return {}


def _paths(cfg: Mapping[str, Any]) -> tuple[pathlib.Path, float, float]:
    paths = cfg.get("paths", {})
    context_cfg = cfg.get("context", {})
    staged_root = pathlib.Path(paths.get("staged", "./data/staged"))
    latency_budget = float(context_cfg.get("update_latency_budget_s", 0.5))
    partial_budget = float(context_cfg.get("partial_sim_max_ms", 200))
    return staged_root, latency_budget, partial_budget


# ---------------------------------------------------------------------------
# Loading staged data (with sample fallback)


def _list_parquet(root: pathlib.Path, name: str) -> list[pathlib.Path]:
    base = root / name
    if not base.exists():
        return []
    return sorted(base.rglob("*.parquet"))


def _load_table(root: pathlib.Path, name: str) -> pl.DataFrame:
    files = _list_parquet(root, name)
    if files:
        return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    loader = getattr(sample_data, f"sample_{name}")
    return loader()


def _polars_to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    try:
        pdf = frame.to_pandas(use_pyarrow_extension_array=True)
    except ModuleNotFoundError:
        pdf = pd.DataFrame(frame.to_dicts())
    try:
        return pdf.convert_dtypes(dtype_backend="pyarrow")
    except (KeyError, TypeError, ValueError):
        return pdf.convert_dtypes()


# ---------------------------------------------------------------------------
# Feature engineering helpers


def _timeout_proxy(clock: pd.Series) -> pd.Series:
    # Assume each side loses a timeout every 10 minutes of elapsed game time.
    clock_clean = clock.fillna(1800.0)
    elapsed = 3600 - clock_clean.clip(lower=0, upper=3600)
    decay = np.floor(elapsed / 600.0)
    values = np.clip(3 - decay, 0, 3).astype(int)
    return pd.Series(values, index=clock.index)


def _momentum(points: pd.Series, window: int = 3) -> pd.Series:
    return points.rolling(window=window, min_periods=1).sum()


def _lead_volatility(lead: pd.Series, window: int = 3) -> pd.Series:
    return lead.rolling(window=window, min_periods=1).std().fillna(0.0)


def _plays_fatigue(plays: pd.Series) -> pd.Series:
    return plays.cumsum()


def _build_drive_features(
    pbp: pd.DataFrame,
    drives: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    pbp = pbp.copy()
    drives = drives.copy()

    first_play = (
        pbp.sort_values(["season", "week", "game_id", "drive", "play_id"])
        .groupby(["game_id", "drive", "posteam"], as_index=False)
        .agg(
            time_remaining=("game_seconds_remaining", "max"),
            score_diff=("score_differential", "max"),
            field_position=("yardline_100", "max"),
            pass_plays=("pass", "sum"),
            total_plays=("play_id", "count"),
        )
    )

    drives = drives.merge(
        first_play,
        left_on=["game_id", "drive_number", "posteam"],
        right_on=["game_id", "drive", "posteam"],
        how="left",
    )

    drives["pace_s_per_play"] = drives["drive_time_seconds"].astype(float) / drives[
        "drive_play_count"
    ].clip(lower=1)

    drives = drives.sort_values(["game_id", "posteam", "drive_number"]).reset_index(drop=True)
    plays_cumsum = drives.groupby(["game_id", "posteam"])["drive_play_count"].transform("cumsum")
    drives["plays_since_start"] = plays_cumsum.astype(float)
    drives["fatigue_proxy"] = plays_cumsum.astype(float)
    drives["current_lead"] = drives["posteam_score"].astype(float) - drives[
        "defteam_score"
    ].astype(float)

    drives["lead_volatility"] = drives.groupby("game_id")["current_lead"].transform(
        lambda s: s.rolling(window=3, min_periods=1).std().fillna(0.0)
    )

    drives["momentum_proxy"] = drives.groupby(["game_id", "posteam"])["drive_points"].transform(
        lambda s: s.rolling(window=3, min_periods=1).sum()
    )
    drives["momentum_proxy"] = drives["momentum_proxy"].astype(float)

    drives["seconds_remaining"] = drives["time_remaining"].astype(float)
    drives["timeouts_off"] = _timeout_proxy(drives["seconds_remaining"])
    drives["timeouts_def"] = drives.groupby("game_id")["timeouts_off"].shift(1).fillna(3).astype(int)

    drives["pass_rate"] = (
        drives["pass_plays"].astype(float) / drives["total_plays"].clip(lower=1)
    )
    league_pass = drives["pass_rate"].mean()
    drives["aggressiveness_hint"] = drives["pass_rate"].fillna(0.0) - league_pass

    if not games.empty and {"game_id", "home_team", "away_team"}.issubset(games.columns):
        coach_lookup = {}
        if {"home_coach", "away_coach"}.issubset(games.columns):
            for row in games.itertuples(index=False):
                coach_lookup[(row.game_id, row.home_team)] = getattr(row, "home_coach", None) or row.home_team
                coach_lookup[(row.game_id, row.away_team)] = getattr(row, "away_coach", None) or row.away_team
        drives["coach_id"] = drives.apply(
            lambda r: coach_lookup.get((r["game_id"], r["posteam"]), r["posteam"]),
            axis=1,
        )
    else:
        drives["coach_id"] = drives["posteam"]

    drives.rename(columns={"posteam": "team_id"}, inplace=True)

    columns = [
        "game_id",
        "drive_id",
        "drive_number",
        "team_id",
        "coach_id",
        "season",
        "week",
        "current_lead",
        "lead_volatility",
        "pace_s_per_play",
        "fatigue_proxy",
        "momentum_proxy",
        "timeouts_off",
        "timeouts_def",
        "plays_since_start",
        "seconds_remaining",
        "field_position",
        "score_diff",
        "aggressiveness_hint",
    ]
    return drives[columns].sort_values(["game_id", "drive_number", "team_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Partial simulation hook


def build_partial_sim_package(row: pd.Series | Mapping[str, Any]) -> Dict[str, Any]:
    """Create a structured package consumable by the Phase 6 simulator.

    Parameters
    ----------
    row:
        A single drive snapshot (``pd.Series`` or mapping) containing the
        engineered context features.

    Returns
    -------
    dict
        A dictionary with the fields required by downstream simulators.
    """

    if not isinstance(row, pd.Series):
        row = pd.Series(row)

    package = {
        "game_id": row["game_id"],
        "team_id": row["team_id"],
        "score_diff": float(row.get("current_lead", 0.0)),
        "expected_pace": float(row.get("pace_s_per_play", 30.0)),
        "fatigue": float(row.get("fatigue_proxy", 0.0)),
        "momentum": float(row.get("momentum_proxy", 0.0)),
        "timeouts": {
            "offense": int(row.get("timeouts_off", 3)),
            "defense": int(row.get("timeouts_def", 3)),
        },
        "state_vector": {
            "time_remaining": float(row.get("seconds_remaining", 1800.0)),
            "field_position": float(row.get("field_position", 50.0)),
            "game_importance": 1.0 + float(row.get("week", 1)) / 18.0,
        },
        "aggressiveness_hint": float(row.get("aggressiveness_hint", 0.0)),
    }

    missing = PARTIAL_SIM_KEYS.difference(package.keys())
    if missing:
        raise KeyError(f"partial sim package missing keys: {missing}")
    return package


# ---------------------------------------------------------------------------
# Main execution flow


def run_phase10(cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pathlib.Path, float]:
    staged_root, latency_budget, partial_budget = _paths(cfg)
    start = time.time()

    pbp = _polars_to_pandas(_load_table(staged_root, "pbp"))
    drives = _polars_to_pandas(_load_table(staged_root, "drives"))
    games = _polars_to_pandas(_load_table(staged_root, "games"))

    features = _build_drive_features(pbp, drives, games)
    latency = time.time() - start

    out_dir = pathlib.Path("artifacts/context")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"state_{stamp}.parquet"
    try:
        features.to_parquet(out_path, index=False)
    except Exception:  # pragma: no cover
        out_path = out_path.with_suffix(".csv")
        features.to_csv(out_path, index=False)

    fatigue_check = features.groupby("team_id")["fatigue_proxy"].diff().fillna(0)
    fatigue_monotonic = (fatigue_check >= -1e-6).all()

    scoring_diff = features.groupby("team_id")["momentum_proxy"].diff().fillna(0)
    momentum_positive = (scoring_diff >= -5).any()

    print(
        f"[context] wrote {out_path} in {latency:.3f}s (budget {latency_budget}s)"
    )
    print(
        f"[context] fatigue monotonic={fatigue_monotonic} momentum_positive={momentum_positive}"
    )

    # Partial sim benchmark on first row to ensure latency adherence.
    if not features.empty:
        hook_start = time.time()
        package = build_partial_sim_package(features.iloc[0])
        hook_latency = (time.time() - hook_start) * 1000.0
        print(
            f"[context] partial-sim hook keys={sorted(package.keys())} latency={hook_latency:.1f}ms "
            f"(budget {partial_budget}ms)"
        )
    else:
        hook_latency = 0.0

    return features, out_path, latency


def main() -> None:
    cfg = _load_config()
    features, out_path, latency = run_phase10(cfg)
    print(
        f"[context] state rows={len(features)} features={list(features.columns)} latency={latency:.3f}s"
    )


if __name__ == "__main__":
    main()
