"""Conformal prediction helpers for calibrated outputs and simulator quantiles."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def split_conformal_binary(
    p: pd.Series,
    y: pd.Series,
    coverage: float = 0.9,
) -> dict:
    """Return a simple split-conformal radius for binary probabilities."""

    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")

    residuals = np.abs(y.to_numpy(dtype=float) - p.to_numpy(dtype=float))
    quantile = float(np.quantile(residuals, coverage))
    return {"q": quantile, "coverage": float(coverage)}


def split_conformal_quantiles(
    samples: Iterable[float],
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> Tuple[float, float]:
    """Return lower and upper quantiles from simulator draws."""

    arr = np.asarray(list(samples), dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("samples must be a one-dimensional iterable with values")
    if not (0 <= q_low < q_high <= 1):
        raise ValueError("quantile bounds must satisfy 0 <= q_low < q_high <= 1")

    lo = float(np.quantile(arr, q_low))
    hi = float(np.quantile(arr, q_high))
    return lo, hi


__all__ = ["split_conformal_binary", "split_conformal_quantiles"]
