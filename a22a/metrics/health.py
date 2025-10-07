"""Health model evaluation stubs for bootstrap phases 11–12."""

from __future__ import annotations

from typing import Dict

import numpy as np


def evaluate_health_predictions(probs: np.ndarray, outcomes: np.ndarray) -> Dict[str, float]:
    """Return deterministic placeholder calibration metrics for health models.

    The bootstrap intentionally avoids any betting odds usage and simply
    validates array shapes before emitting representative metric values.
    """

    if probs.shape != outcomes.shape:
        raise ValueError("probabilities and outcomes must align in shape")

    return {
        "expected_calibration_error": float(np.abs(probs - outcomes).mean()),
        "brier_score": float(np.mean((probs - outcomes) ** 2)),
    }


__all__ = ["evaluate_health_predictions"]
