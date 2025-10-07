"""Synergy scaffolds for Phase 8 UER bootstrap."""
from __future__ import annotations

import pandas as pd


def offensive_line_qb_synergy() -> pd.DataFrame:
    """Placeholder OL×QB synergy surface."""

    return pd.DataFrame(
        {
            "pair": ["OL_stub+QB_stub"],
            "synergy_value": [0.0],
        }
    )


def qb_wr_synergy() -> pd.DataFrame:
    """Placeholder QB×WR synergy surface."""

    return pd.DataFrame(
        {
            "pair": ["QB_stub+WR_stub"],
            "synergy_value": [0.0],
        }
    )


def front_scheme_synergy() -> pd.DataFrame:
    """Placeholder front-seven × scheme matchup."""

    return pd.DataFrame(
        {
            "pair": ["Front_stub+Scheme_stub"],
            "synergy_value": [0.0],
        }
    )
