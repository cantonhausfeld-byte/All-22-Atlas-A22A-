"""Utility helpers for summarising player impact simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass
class ImpactDelta:
    """Container for an impact estimate with a confidence interval."""

    estimate: float
    lower: float
    upper: float

    def as_dict(self, suffix: str) -> Mapping[str, float]:
        return {
            f"delta_{suffix}": self.estimate,
            f"delta_{suffix}_ci_low": self.lower,
            f"delta_{suffix}_ci_high": self.upper,
        }


def _ensure_array(samples: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(samples), dtype=float)
    if arr.size == 0:
        raise ValueError("At least one sample is required to compute an impact delta")
    return arr


def summarize_delta(samples: Iterable[float], ci_level: float) -> ImpactDelta:
    """Return the mean and equal-tailed CI for a set of samples."""

    arr = _ensure_array(samples)
    alpha = (1 - ci_level) / 2
    lower = float(np.quantile(arr, alpha))
    upper = float(np.quantile(arr, 1 - alpha))
    estimate = float(arr.mean())
    return ImpactDelta(estimate=estimate, lower=lower, upper=upper)


def summarize_player_metric(win_samples: Iterable[float],
                            margin_samples: Iterable[float],
                            total_samples: Iterable[float],
                            ci_level: float) -> Mapping[str, float]:
    """Compute summary statistics for a player's simulated impact."""

    win_delta = summarize_delta(win_samples, ci_level)
    margin_delta = summarize_delta(margin_samples, ci_level)
    total_delta = summarize_delta(total_samples, ci_level)

    summary: dict[str, float] = {}
    summary.update(win_delta.as_dict("win_pct"))
    summary.update(margin_delta.as_dict("margin"))
    summary.update(total_delta.as_dict("total"))
    return summary


__all__ = [
    "ImpactDelta",
    "summarize_delta",
    "summarize_player_metric",
]
