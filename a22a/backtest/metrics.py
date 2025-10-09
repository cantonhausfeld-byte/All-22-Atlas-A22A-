"""Utility metrics for summarising backtests.

These helpers intentionally keep their inputs simple (plain sequences of floats)
so they can be reused by both bootstrap smoke tests and richer simulations.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _safe_sum(values: Iterable[float]) -> float:
    return float(sum(values))


def roi(payouts: Sequence[float], stakes: Sequence[float]) -> float:
    """Return on investment given payouts and stakes."""

    total_stake = _safe_sum(stakes)
    if total_stake == 0:
        return 0.0
    total_return = _safe_sum(payouts) - total_stake
    return total_return / total_stake


def win_rate(wins: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return wins / total


def expected_calibration_error(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Compute a tiny-bin ECE approximation.

    We bucket predictions into two coarse bins to keep things robust in the
    bootstrap setting.
    """

    if not probabilities or not outcomes:
        return 0.0
    low_bin = [p for p in probabilities if p < 0.5]
    low_outcomes = [o for p, o in zip(probabilities, outcomes) if p < 0.5]
    high_bin = [p for p in probabilities if p >= 0.5]
    high_outcomes = [o for p, o in zip(probabilities, outcomes) if p >= 0.5]

    def _bin_ece(bin_probs: Sequence[float], bin_outcomes: Sequence[int]) -> float:
        if not bin_probs:
            return 0.0
        avg_prob = sum(bin_probs) / len(bin_probs)
        avg_outcome = sum(bin_outcomes) / len(bin_outcomes)
        return abs(avg_prob - avg_outcome) * len(bin_probs) / len(probabilities)

    return _bin_ece(low_bin, low_outcomes) + _bin_ece(high_bin, high_outcomes)


def clv_basis_points(open_prices: Sequence[float], close_prices: Sequence[float]) -> float:
    """Compute average closing line value in basis points."""

    if not open_prices or not close_prices:
        return 0.0
    deltas = [(close - open_) * 10000 for open_, close in zip(open_prices, close_prices)]
    return sum(deltas) / len(deltas)


def max_drawdown(equity_curve: Sequence[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = (value - peak) / peak if peak else 0.0
        max_dd = min(max_dd, drawdown)
    return max_dd


def sharpe_like(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(variance)
    if std_dev == 0:
        return 0.0
    return mean_return / std_dev


def herfindahl_index(weights: Sequence[float]) -> float:
    """Compute the Herfindahl-Hirschman Index for a collection of weights."""

    if not weights:
        return 0.0
    total = sum(abs(w) for w in weights)
    if total == 0:
        return 0.0
    normalised = [abs(w) / total for w in weights if total]
    return sum(w ** 2 for w in normalised)


__all__ = [
    "roi",
    "win_rate",
    "expected_calibration_error",
    "clv_basis_points",
    "max_drawdown",
    "sharpe_like",
    "herfindahl_index",
]
