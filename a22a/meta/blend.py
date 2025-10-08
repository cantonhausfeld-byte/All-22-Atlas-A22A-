"""Blending and stacking helpers for the meta-learner stage."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def stack_logit(p_cols: Iterable[str], df: pd.DataFrame, seed: int = 14) -> pd.Series:
    """Combine probability columns with a simple logit-average stub.

    Parameters
    ----------
    p_cols:
        Iterable of column names in ``df`` that contain probability predictions.
    df:
        DataFrame holding the candidate probabilities.
    seed:
        Random seed reserved for future stochastic ensembling strategies.

    Returns
    -------
    pandas.Series
        Blended probabilities clipped to :math:`[0, 1]`.
    """

    if not p_cols:
        raise ValueError("p_cols must be non-empty for stacking")

    arr = df.loc[:, list(p_cols)].to_numpy(dtype=float)
    arr = np.clip(arr, 1e-6, 1 - 1e-6)
    logits = np.log(arr / (1 - arr))
    mean_logit = logits.mean(axis=1)
    blended = 1 / (1 + np.exp(-mean_logit))
    return pd.Series(blended, index=df.index, name="p_meta").clip(0.0, 1.0)


__all__ = ["stack_logit"]
