"""Selection metrics utilities for decision engine stubs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SelectionMetrics:
    """Bundle of selection quality indicators."""

    precision_at_k: float
    coverage: float
    sample_size: int


def precision_at_k(actuals: Iterable[bool], k: int) -> float:
    """Compute precision@K on a boolean iterable.

    The iterable is assumed to be ordered by model confidence (descending).
    """

    actual_list = list(actuals)
    if k <= 0 or not actual_list:
        return 0.0
    top_k = actual_list[: k if k <= len(actual_list) else len(actual_list)]
    return sum(1 for val in top_k if val) / len(top_k)


def selection_coverage(selected_mask: Iterable[bool]) -> float:
    """Compute coverage as the fraction of items selected."""

    mask_list = list(selected_mask)
    if not mask_list:
        return 0.0
    return sum(1 for flag in mask_list if flag) / len(mask_list)


def evaluate_selection(actuals: Iterable[bool], selected_mask: Iterable[bool], *, k: int) -> SelectionMetrics:
    """Evaluate selection quality using simple metrics."""

    actual_list = list(actuals)
    mask_list = list(selected_mask)
    if len(actual_list) != len(mask_list):
        raise ValueError("actuals and selected_mask must be the same length")

    ordered_actuals = [act for act, flag in zip(actual_list, mask_list) if flag]
    prec = precision_at_k(ordered_actuals, k)
    cov = selection_coverage(mask_list)
    return SelectionMetrics(precision_at_k=prec, coverage=cov, sample_size=len(actual_list))
