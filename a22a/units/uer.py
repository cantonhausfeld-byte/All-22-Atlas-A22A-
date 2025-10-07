"""Phase 8 Unit Effectiveness Ratings (UER).

The ratings approximate a ridge-regularised adjusted plus-minus system at the
team-week level.  The implementation is intentionally light-weight while still
showing the full set of modelling hooks expected later in the project: feature
engineering (including TTT normalisation), opponent adjustments, recency
shrinkage, and synergy scaffolds.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import yaml

from a22a.units import synergy

ARTIFACT_DIR = Path("artifacts/uer")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
UER_AXES = ["off_pass", "off_rush", "def_pass", "def_rush"]


@dataclass(frozen=True)
class UERConfig:
    recency_half_life_weeks: float
    ridge_lambda: float
    min_snaps_threshold: int
    ttt_reference: float

    @classmethod
    def from_mapping(cls, data: dict[str, float]) -> "UERConfig":
        return cls(
            recency_half_life_weeks=float(data.get("recency_half_life_weeks", 6)),
            ridge_lambda=float(data.get("ridge_lambda", 10.0)),
            min_snaps_threshold=int(data.get("min_snaps_threshold", 50)),
            ttt_reference=float(data.get("ttt_reference", 2.6)),
        )


def _load_config(path: str | Path = "configs/defaults.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text())


def _toy_snap_log() -> pd.DataFrame:
    """Create a deterministic toy snapshot table."""

    rng = np.random.default_rng(2208)
    rows = []
    teams = ["TEAM_A", "TEAM_B", "TEAM_C", "TEAM_D"]
    base_epa = {"off_pass": 0.15, "off_rush": 0.06, "def_pass": -0.10, "def_rush": -0.04}
    base_success = {"off_pass": 0.53, "off_rush": 0.50, "def_pass": 0.45, "def_rush": 0.47}
    for team in teams:
        for axis in UER_AXES:
            rows.append(
                {
                    "unit_id": f"{team}_{axis}",
                    "team": team,
                    "axis": axis,
                    "epa_per_play": base_epa[axis] + rng.normal(0, 0.02),
                    "success_rate": base_success[axis] + rng.normal(0, 0.015),
                    "ttt": 2.55 + rng.normal(0, 0.18),
                    "opponent_epa_allowed": rng.normal(0, 0.015),
                    "opponent_success_allowed": rng.normal(0, 0.02),
                    "snaps": int(rng.integers(35, 75)),
                    "weeks_ago": int(rng.integers(0, 8)),
                    "injury_factor": rng.uniform(0.9, 1.05),
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


def _feature_engineering(df: pd.DataFrame, config: UERConfig) -> pd.DataFrame:
    engineered = df.copy()
    engineered["success_delta"] = engineered.groupby("axis")["success_rate"].transform(
        lambda s: s - s.mean()
    )
    engineered["epa_adj"] = engineered["epa_per_play"] - engineered["opponent_epa_allowed"]
    engineered["success_adj"] = (
        engineered["success_rate"] - engineered["opponent_success_allowed"]
    )
    engineered["injury_adj"] = engineered["injury_factor"].fillna(1.0)

    engineered["base_value"] = 0.6 * engineered["epa_adj"] + 0.4 * engineered["success_adj"]

    # Time-to-throw normalisation only affects passing axes.
    pass_mask = engineered["axis"].isin(["off_pass", "def_pass"])
    ttt_ref = config.ttt_reference
    engineered.loc[pass_mask, "ttt_normalised"] = (
        engineered.loc[pass_mask, "ttt"].fillna(ttt_ref) - ttt_ref
    )
    engineered.loc[~pass_mask, "ttt_normalised"] = 0.0
    engineered.loc[pass_mask, "base_value"] -= 0.12 * engineered.loc[pass_mask, "ttt_normalised"]

    # Defensive metrics are framed so negative EPA is good.
    def_mask = engineered["axis"].str.startswith("def_")
    engineered.loc[def_mask, "base_value"] *= -1

    engineered["value"] = engineered["base_value"] * engineered["injury_adj"]
    return engineered


def _opponent_adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Apply opponent adjustments to the engineered value."""

    adjusted = df.copy()
    adjusted["adjusted_value"] = adjusted["value"]
    return adjusted


def _solve_ridge(axis_df: pd.DataFrame, config: UERConfig) -> dict[str, tuple[float, float, float]]:
    if axis_df.empty:
        return {}

    units = axis_df["unit_id"].unique().tolist()
    idx_map = {unit: i for i, unit in enumerate(units)}
    n_obs = len(axis_df)
    n_units = len(units)

    X = np.zeros((n_obs, n_units))
    y = axis_df["adjusted_value"].to_numpy()
    w = axis_df["weight"].to_numpy()
    snaps = axis_df.groupby("unit_id")["snaps"].sum().to_dict()

    for row_idx, (_, row) in enumerate(axis_df.iterrows()):
        X[row_idx, idx_map[row["unit_id"]]] = 1.0

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w

    lhs = Xw.T @ Xw + config.ridge_lambda * np.eye(n_units)
    rhs = Xw.T @ yw
    try:
        inv_lhs = np.linalg.inv(lhs)
    except np.linalg.LinAlgError:
        inv_lhs = np.linalg.pinv(lhs)
    beta = inv_lhs @ rhs

    residuals = y - X @ beta
    dof = max(float(w.sum()) - n_units, 1.0)
    sigma2 = float(np.dot(w * residuals, residuals) / dof)
    cov = inv_lhs * sigma2
    std_err = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    results: dict[str, tuple[float, float, float]] = {}
    for unit, idx in idx_map.items():
        total_snaps = snaps.get(unit, 0)
        snap_factor = min(1.0, total_snaps / max(config.min_snaps_threshold, 1))
        mean = float(beta[idx]) * snap_factor
        half_width = 1.96 * std_err[idx]
        if snap_factor < 1.0:
            half_width *= 1.0 / max(snap_factor, 1e-3)
        results[unit] = (mean, mean - half_width, mean + half_width)
    return results


def _compute_uer_table(df: pd.DataFrame, config: UERConfig) -> pd.DataFrame:
    rows: dict[str, dict[str, float | str | int]] = {}
    snap_totals = df.groupby("unit_id")["snaps"].sum().to_dict()
    for axis in UER_AXES:
        axis_df = df[df["axis"] == axis]
        axis_results = _solve_ridge(axis_df, config)
        for unit_id, stats in axis_results.items():
            row = rows.setdefault(
                unit_id,
                {"unit_id": unit_id, "snaps": int(snap_totals.get(unit_id, 0))},
            )
            mean, ci_low, ci_high = stats
            row[f"{axis}_mean"] = mean
            row[f"{axis}_ci_low"] = ci_low
            row[f"{axis}_ci_high"] = ci_high
    table = pd.DataFrame(rows.values())
    for axis in UER_AXES:
        mean_col = f"{axis}_mean"
        low_col = f"{axis}_ci_low"
        high_col = f"{axis}_ci_high"
        if mean_col not in table.columns:
            table[mean_col] = 0.0
        if low_col not in table.columns:
            table[low_col] = table[mean_col]
        if high_col not in table.columns:
            table[high_col] = table[mean_col]
    return table


def _artifact_path(stamp: Optional[str] = None) -> Path:
    stem = stamp or datetime.now(timezone.utc).strftime("%Y%m%d")
    return ARTIFACT_DIR / f"uer_week_{stem}.parquet"


def _log_extremes(table: pl.DataFrame) -> None:
    for axis in UER_AXES:
        mean_col = f"{axis}_mean"
        ci_low = f"{axis}_ci_low"
        ci_high = f"{axis}_ci_high"
        if mean_col not in table.columns:
            continue
        top = table.sort(mean_col, descending=True).head(5)
        bottom = table.sort(mean_col, descending=False).head(5)
        print(f"[uer] axis={axis} top 5:")
        for row in top.iter_rows(named=True):
            mean = row.get(mean_col)
            low = row.get(ci_low)
            high = row.get(ci_high)
            if mean is None or low is None or high is None:
                continue
            print(f"  {row['unit_id']}: {mean:+.3f} ({low:+.3f}, {high:+.3f})")
        print(f"[uer] axis={axis} bottom 5:")
        for row in bottom.iter_rows(named=True):
            mean = row.get(mean_col)
            low = row.get(ci_low)
            high = row.get(ci_high)
            if mean is None or low is None or high is None:
                continue
            print(f"  {row['unit_id']}: {mean:+.3f} ({low:+.3f}, {high:+.3f})")


def run(
    config_path: str | Path = "configs/defaults.yaml",
    *,
    snapshots: Optional[pd.DataFrame] = None,
    stamp: Optional[str] = None,
) -> pl.DataFrame:
    """Entry point for ``make uer``."""

    raw = _load_config(config_path)
    uer_cfg = UERConfig.from_mapping(raw.get("uer", {}))
    base = snapshots.copy() if snapshots is not None else _toy_snap_log()
    engineered = _feature_engineering(base, uer_cfg)
    weighted = _apply_recency_weights(engineered, uer_cfg)
    adjusted = _opponent_adjust(weighted)

    # Synergy scaffolds (placeholders for future integration but invoked here).
    synergy.offensive_line_qb_synergy()
    synergy.qb_wr_synergy()
    synergy.front_scheme_synergy()

    table_pd = _compute_uer_table(adjusted, uer_cfg)
    table_pl = pl.from_pandas(table_pd)
    artifact_path = _artifact_path(stamp)
    table_pl.write_parquet(artifact_path)

    _log_extremes(table_pl)
    print("[uer] table written to", artifact_path)
    return table_pl


def main() -> None:
    run()


if __name__ == "__main__":
    main()
