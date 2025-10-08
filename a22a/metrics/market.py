"""Helpers for working with betting market prices."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


def american_to_decimal(odds: Iterable[float]) -> np.ndarray:
    odds_arr = np.asarray(list(odds), dtype=float)
    pos_mask = odds_arr > 0
    dec = np.empty_like(odds_arr, dtype=float)
    dec[pos_mask] = 1.0 + odds_arr[pos_mask] / 100.0
    dec[~pos_mask] = 1.0 + 100.0 / np.abs(odds_arr[~pos_mask])
    return dec


def implied_from_american(odds: Iterable[float]) -> np.ndarray:
    decimal = american_to_decimal(odds)
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = np.where(decimal > 0, 1.0 / decimal, np.nan)
    return implied


def remove_vig(p_home: float, p_away: float, eps: float = 1e-9) -> Tuple[float, float]:
    total = float(p_home) + float(p_away) + eps
    return float(p_home) / total, float(p_away) / total


def pairwise_remove_vig(values: Sequence[float]) -> np.ndarray:
    if len(values) != 2:
        raise ValueError("pairwise_remove_vig expects exactly two values")
    adjusted = remove_vig(values[0], values[1])
    return np.asarray(adjusted, dtype=float)


def basis_points_delta(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    return 10000.0 * (arr_a - arr_b)


__all__ = [
    "american_to_decimal",
    "implied_from_american",
    "remove_vig",
    "pairwise_remove_vig",
    "basis_points_delta",
]
