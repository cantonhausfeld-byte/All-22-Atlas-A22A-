"""Bootstrap weekly portfolio sizing logic.

The real implementation will ingest calibrated probabilities and simulation
outputs. For now we synthesize a deterministic slate and size wagers subject to
basic Kelly-derived risk caps.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from a22a.metrics import portfolio as portfolio_metrics


@dataclass
class PCfg:
    """Portfolio configuration parsed from ``configs/defaults.yaml``."""

    bankroll: float
    kelly_fraction: float
    max_stake_pct_per_bet: float
    max_weekly_exposure_pct: float
    max_game_exposure_pct: float
    corr_guard: bool
    corr_threshold: float


def kelly_even(prob: float, fraction: float) -> float:
    """Return fractional Kelly stake for an even-money payoff.

    Parameters
    ----------
    prob:
        Win probability.
    fraction:
        Fraction of full Kelly to deploy.
    """

    edge = 2 * prob - 1
    return max(0.0, edge) * fraction


def load_config() -> PCfg:
    cfg = yaml.safe_load(pathlib.Path("configs/defaults.yaml").read_text())
    return PCfg(**cfg["portfolio"])


def synthesize_slate(n_games: int = 12) -> pd.DataFrame:
    """Create a deterministic slate with a guaranteed coin-flip entry."""

    rng = np.random.default_rng(15)
    p_home = np.clip(0.5 + 0.2 * rng.normal(size=n_games), 0.01, 0.99)
    p_home[0] = 0.5  # ensure abstention case for tests
    df = pd.DataFrame({
        "game_id": [f"G{i:03d}" for i in range(n_games)],
        "p_home": p_home,
    })
    df["pick_side"] = np.where(df.p_home >= 0.5, "HOME", "AWAY")
    df["pick_prob"] = np.where(df.p_home >= 0.5, df.p_home, 1 - df.p_home)
    return df


def size_portfolio(df: pd.DataFrame, pcfg: PCfg) -> pd.DataFrame:
    exposure = 0.0
    stakes = []
    exposure_trace = []

    for _, row in df.iterrows():
        stake = min(
            kelly_even(row["pick_prob"], pcfg.kelly_fraction),
            pcfg.max_stake_pct_per_bet,
            pcfg.max_game_exposure_pct,
        )
        if exposure + stake > pcfg.max_weekly_exposure_pct:
            stake = 0.0
        exposure += stake
        stakes.append(round(stake, 4))
        exposure_trace.append(round(exposure, 4))

    sized = df.copy()
    sized["stake_pct"] = stakes
    sized["stake_amount"] = (sized["stake_pct"] * pcfg.bankroll).round(2)
    sized["exposure_pct"] = exposure_trace
    sized["exposure_amount"] = (sized["exposure_pct"] * pcfg.bankroll).round(2)
    return sized


def write_portfolio(df: pd.DataFrame, outdir: pathlib.Path) -> pathlib.Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"picks_week_{stamp}.parquet"
    try:
        df.to_parquet(out, index=False)
    except Exception:
        out = out.with_suffix(".csv")
        df.to_csv(out, index=False)
    return out


def main() -> None:
    t0 = time.time()
    pcfg = load_config()
    slate = synthesize_slate()
    sized = size_portfolio(slate, pcfg)

    out_path = write_portfolio(sized, pathlib.Path("artifacts/portfolio"))

    exposure_stats = portfolio_metrics.exposure_summary(sized)
    concentration_stats = portfolio_metrics.concentration_summary(sized)

    elapsed = time.time() - t0
    print(
        f"[portfolio] wrote {out_path.name} "
        f"exposure={exposure_stats['total_pct']:.3f} "
        f"concentration={concentration_stats['gini']:.3f} "
        f"in {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
