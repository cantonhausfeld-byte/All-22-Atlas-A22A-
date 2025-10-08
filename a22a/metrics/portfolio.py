"""Portfolio metrics stubs used during bootstrap."""

from __future__ import annotations

import pandas as pd


def exposure_summary(df: pd.DataFrame) -> dict:
    """Return aggregate exposure information.

    Parameters
    ----------
    df:
        Portfolio dataframe containing ``stake_pct`` and ``exposure_pct``.
    """

    total_pct = float(df.get("exposure_pct", pd.Series(dtype=float)).max() or 0.0)
    total_amount = float(df.get("exposure_amount", pd.Series(dtype=float)).max() or 0.0)
    return {
        "total_pct": total_pct,
        "total_amount": total_amount,
    }


def concentration_summary(df: pd.DataFrame) -> dict:
    """Compute a simple Gini-style concentration measure."""

    stakes = df.get("stake_pct", pd.Series(dtype=float)).astype(float)
    if stakes.empty or stakes.sum() == 0:
        return {"gini": 0.0}
    sorted_stakes = stakes.sort_values().to_numpy()
    n = len(sorted_stakes)
    cumulative = sorted_stakes.cumsum()
    gini = 1 - (2 / (n - 1)) * ((n - cumulative / cumulative[-1]).sum()) if n > 1 else 0.0
    return {"gini": float(max(0.0, min(1.0, gini)))}


def turnover_summary(previous: pd.DataFrame | None, current: pd.DataFrame) -> dict:
    """Placeholder turnover metric that tracks changed games."""

    if previous is None or previous.empty:
        return {"changed": int(len(current))}
    prev_ids = set(previous.get("game_id", []))
    curr_ids = set(current.get("game_id", []))
    return {"changed": int(len(curr_ids.symmetric_difference(prev_ids)))}


__all__ = [
    "exposure_summary",
    "concentration_summary",
    "turnover_summary",
]
