"""Phase 8 Unit Effectiveness Ratings (UER) bootstrap stubs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
import yaml

from a22a.units import synergy

ARTIFACT_DIR = Path("artifacts/uer")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_FILE = ARTIFACT_DIR / "uer_week_stub.parquet"
UER_AXES = ["off_pass", "off_rush", "def_pass", "def_rush"]


@dataclass(frozen=True)
class UERConfig:
    recency_half_life_weeks: float
    ridge_lambda: float
    min_snaps_threshold: int

    @classmethod
    def from_mapping(cls, data: dict[str, float]) -> "UERConfig":
        return cls(
            recency_half_life_weeks=float(data.get("recency_half_life_weeks", 6)),
            ridge_lambda=float(data.get("ridge_lambda", 10.0)),
            min_snaps_threshold=int(data.get("min_snaps_threshold", 50)),
        )


def _load_config(path: str | Path = "configs/defaults.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text())


def _toy_snap_log() -> pd.DataFrame:
    """Create a toy RAPM-style snapshot table."""

    rows = []
    for unit_id in ("TEAM_A_OFF", "TEAM_B_DEF"):
        for axis in UER_AXES:
            rows.append(
                {
                    "unit_id": unit_id,
                    "axis": axis,
                    "value": 0.05 if "off" in axis else -0.03,
                    "snaps": 60 if unit_id == "TEAM_A_OFF" else 45,
                    "weeks_ago": 1 if unit_id == "TEAM_A_OFF" else 3,
                    "opponent_strength": 0.01,
                }
            )
    return pd.DataFrame(rows)


def _recency_weight(weeks_ago: float, half_life: float) -> float:
    return 0.5 ** (weeks_ago / max(half_life, 1e-6))


def _apply_recency_weights(df: pd.DataFrame, config: UERConfig) -> pd.DataFrame:
    weighted = df.copy()
    weighted["weight"] = weighted["weeks_ago"].apply(
        lambda w: _recency_weight(w, config.recency_half_life_weeks)
    )
    return weighted


def _opponent_adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Simple opponent adjustment placeholder."""

    adjusted = df.copy()
    adjusted["adjusted_value"] = adjusted["value"] - adjusted["opponent_strength"]
    return adjusted


def _ridge_shrink(mean: float, weight_sum: float, config: UERConfig) -> float:
    denom = weight_sum + config.ridge_lambda
    if denom == 0:
        return 0.0
    return mean * (weight_sum / denom)


def _ci_half_width(snaps: int) -> float:
    return 1.96 * (0.1 / math.sqrt(max(snaps, 1)))


def _estimate_axis_rating(df: pd.DataFrame, config: UERConfig) -> tuple[float, float, float]:
    if df.empty:
        return 0.0, 0.0, 0.0

    weight_sum = df["weight"].sum()
    weighted_mean = (df["adjusted_value"] * df["weight"]).sum() / max(weight_sum, 1e-6)
    shrunk = _ridge_shrink(weighted_mean, weight_sum, config)

    total_snaps = int(df["snaps"].sum())
    if total_snaps < config.min_snaps_threshold:
        shrink_factor = total_snaps / max(config.min_snaps_threshold, 1)
        shrunk *= shrink_factor
    else:
        shrink_factor = 1.0

    half_width = _ci_half_width(total_snaps) + (1 - shrink_factor) * 0.05
    return shrunk, shrunk - half_width, shrunk + half_width


def _compute_uer_table(df: pd.DataFrame, config: UERConfig) -> pd.DataFrame:
    results = []
    for unit_id, group in df.groupby("unit_id"):
        row: dict[str, float | str | int] = {"unit_id": unit_id, "snaps": int(group["snaps"].sum())}
        for axis in UER_AXES:
            axis_group = group[group["axis"] == axis]
            mean, ci_low, ci_high = _estimate_axis_rating(axis_group, config)
            row[f"{axis}_mean"] = mean
            row[f"{axis}_ci_low"] = ci_low
            row[f"{axis}_ci_high"] = ci_high
        results.append(row)
    return pd.DataFrame(results)


def run(config_path: str | Path = "configs/defaults.yaml", *, snapshots: Optional[pd.DataFrame] = None) -> pl.DataFrame:
    """Entry point for ``make uer``."""

    raw = _load_config(config_path)
    uer_cfg = UERConfig.from_mapping(raw.get("uer", {}))
    base = snapshots.copy() if snapshots is not None else _toy_snap_log()
    weighted = _apply_recency_weights(base, uer_cfg)
    adjusted = _opponent_adjust(weighted)

    synergy.offensive_line_qb_synergy()
    synergy.qb_wr_synergy()
    synergy.front_scheme_synergy()

    table_pd = _compute_uer_table(adjusted, uer_cfg)
    table_pl = pl.from_pandas(table_pd)
    table_pl.write_parquet(ARTIFACT_FILE)

    print("[uer] table written to", ARTIFACT_FILE)
    return table_pl


def main() -> None:
    run()


if __name__ == "__main__":
    main()
