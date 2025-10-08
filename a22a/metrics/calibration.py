"""Calibration metrics utilities used by the Phase 14 meta-learner."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def ece(p: pd.Series | np.ndarray, y: pd.Series | np.ndarray, bins: int = 10) -> float:
    """Expected calibration error (ECE) using equal-width bins."""

    probs = np.asarray(p, dtype=float)
    labels = np.asarray(y, dtype=float)
    if probs.shape != labels.shape:
        raise ValueError("Probability and label arrays must share the same shape")

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece_val = 0.0
    total = len(probs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if not np.any(mask):
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        weight = mask.sum() / total
        ece_val += abs(bin_acc - bin_conf) * weight
    return float(ece_val)


def brier_score(p: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    probs = np.asarray(p, dtype=float)
    labels = np.asarray(y, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def log_loss(p: pd.Series | np.ndarray, y: pd.Series | np.ndarray, eps: float = 1e-6) -> float:
    probs = np.asarray(p, dtype=float)
    labels = np.asarray(y, dtype=float)
    probs = np.clip(probs, eps, 1 - eps)
    return float(-(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)).mean())


def reliability_bins(
    p: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    bins: int = 10,
) -> List[Dict[str, float]]:
    """Return per-bin accuracy/volume pairs for reliability diagrams."""

    probs = np.asarray(p, dtype=float)
    labels = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    report: List[Dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if not np.any(mask):
            continue
        report.append(
            {
                "lower": float(lo),
                "upper": float(hi),
                "count": float(mask.sum()),
                "fraction": float(mask.sum() / probs.size),
                "confidence": float(probs[mask].mean()),
                "accuracy": float(labels[mask].mean()),
            }
        )
    return report


__all__ = ["ece", "brier_score", "log_loss", "reliability_bins"]
