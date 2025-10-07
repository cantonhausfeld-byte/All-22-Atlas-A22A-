"""Selection logic for Phase 7 decision stubs."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import pandas as pd

from a22a.metrics.selection import SelectionMetrics, evaluate_selection


class SelectionMode(str, Enum):
    """Selection policy enum."""

    TOP_K = "top_k"
    THRESHOLD = "threshold"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class SelectivityConfig:
    prob_threshold: float
    k_top: int
    mode: SelectionMode = SelectionMode.HYBRID


@dataclass(frozen=True)
class SelectivityResult:
    """Container for selection output."""

    selected_mask: list[bool]
    metrics: SelectionMetrics


def _threshold_mask(probs: Iterable[float], threshold: float) -> list[bool]:
    return [p >= threshold for p in probs]


def _top_k_mask(probs: Iterable[float], k: int) -> list[bool]:
    if k <= 0:
        return [False for _ in probs]
    series = pd.Series(list(probs))
    if series.empty:
        return []
    top_idx = series.sort_values(ascending=False).head(k).index
    return [i in set(top_idx) for i in range(len(series))]


def apply_selectivity(df: pd.DataFrame, config: SelectivityConfig, actual_col: str = "actual") -> SelectivityResult:
    """Apply configured selection policy and compute metrics."""

    if config.mode is SelectionMode.THRESHOLD:
        selected_mask = _threshold_mask(df["win_prob"], config.prob_threshold)
    elif config.mode is SelectionMode.TOP_K:
        selected_mask = _top_k_mask(df["win_prob"], config.k_top)
    else:
        threshold_mask = _threshold_mask(df["win_prob"], config.prob_threshold)
        topk_mask = _top_k_mask(df["win_prob"], config.k_top)
        selected_mask = [th and tk for th, tk in zip(threshold_mask, topk_mask)]

    metrics = evaluate_selection(df[actual_col], selected_mask, k=config.k_top)
    return SelectivityResult(selected_mask=selected_mask, metrics=metrics)
