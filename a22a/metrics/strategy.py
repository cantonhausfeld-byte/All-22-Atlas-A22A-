"""Strategy calibration metrics stubs."""

from __future__ import annotations

from typing import Iterable


def calibration_stub(pred: Iterable[float], actual: Iterable[float]) -> float:
    """Return a fixed calibration error estimate.

    The bootstrap implementation keeps an identical interface to the
    planned metric utilities so downstream callers can be wired without
    requiring probabilistic outputs yet.
    """

    return 0.05


__all__ = ["calibration_stub"]
