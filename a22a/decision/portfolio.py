"""Phase 7 decision engine implementation.

The decision engine ingests calibrated win probabilities produced upstream, Monte
Carlo distribution summaries from the Phase 6 simulator, and bankroll settings
from configuration.  Using those inputs it selects a portfolio of wagers and
sizes stakes under a fractional Kelly framework with multiple exposure caps.

The module intentionally keeps the data requirements light – helpers generate a
small deterministic stub slate that mimics the shape of future integrations so
tests can exercise the full flow without external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

from a22a.decision.selectivity import SelectivityConfig, SelectionMode, apply_selectivity
from a22a.metrics.selection import SelectionMetrics


ARTIFACT_DIR = Path("artifacts/picks")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DecisionConfig:
    """Configuration bundle for the decision engine."""

    k_top: int
    prob_threshold: float
    selection_mode: SelectionMode
    kelly_fraction: float
    max_stake_pct_per_bet: float
    max_stake_pct_per_game: float
    max_weekly_exposure_pct: float
    seed: int

    @classmethod
    def from_mapping(cls, data: dict[str, float]) -> "DecisionConfig":
        return cls(
            k_top=int(data.get("k_top", 5)),
            prob_threshold=float(data.get("prob_threshold", 0.6)),
            selection_mode=SelectionMode(data.get("selection_mode", "hybrid")),
            kelly_fraction=float(data.get("kelly_fraction", 0.25)),
            max_stake_pct_per_bet=float(data.get("max_stake_pct_per_bet", 0.02)),
            max_stake_pct_per_game=float(data.get("max_stake_pct_per_game", 0.03)),
            max_weekly_exposure_pct=float(data.get("max_weekly_exposure_pct", 0.15)),
            seed=int(data.get("seed", 42)),
        )


def _load_config(path: str | Path = "configs/defaults.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text())


def _stub_candidates() -> pd.DataFrame:
    """Create a deterministic stub slate used in tests.

    The stub includes the calibrated win probability, Phase 6 simulation moments,
    and historical outcomes for a tiny evaluation slice.
    """

    rng = np.random.default_rng(202401)
    slate = []
    games = [
        ("GB@CHI", "spread", ("CHI", 0.58, 2.4), ("GB", 0.42, -2.4)),
        ("KC@DEN", "moneyline", ("KC", 0.64, 5.6), ("DEN", 0.36, -5.6)),
        ("SF@SEA", "total_over", ("SF", 0.55, 3.1),),
        ("DAL@PHI", "spread", ("DAL", 0.51, 0.8), ("PHI", 0.49, -0.8)),
    ]
    for game_id, market, *sides in games:
        for side, prob, margin in sides:
            slate.append(
                {
                    "game_id": game_id,
                    "market": market,
                    "side": side,
                    "win_prob": prob,
                    "sim_margin_mean": margin,
                    "sim_margin_std": abs(margin) / 2 + 4,
                    "sim_fair_prob": 0.5 + np.sign(margin) * min(0.1, abs(margin) / 10),
                    "actual": bool(rng.uniform() < prob),
                }
            )
    return pd.DataFrame(slate)


def _stub_simulations(candidates: pd.DataFrame, *, draws: int = 256) -> pd.DataFrame:
    """Create Phase 6-style simulation draws for correlation estimation."""

    rng = np.random.default_rng(8675309)
    samples = []
    for game_id, group in candidates.groupby("game_id"):
        mean = group["sim_margin_mean"].mean()
        std = max(1.5, group["sim_margin_std"].mean())
        draws_arr = rng.normal(loc=mean, scale=std, size=draws)
        for idx, value in enumerate(draws_arr):
            samples.append({"game_id": game_id, "sample_id": idx, "margin": float(value)})
    return pd.DataFrame(samples)


def _kelly_fraction(prob: float, *, fraction: float) -> float:
    """Return the fractional Kelly stake for even odds."""

    edge = max(0.0, 2 * prob - 1)
    return edge * fraction


def _confidence_signal(prob: float, sim_fair_prob: float, sim_margin_mean: float) -> float:
    """Blend multiple confidence signals into a scalar rank value."""

    prob_edge = abs(prob - 0.5)
    sim_edge = abs(sim_fair_prob - 0.5)
    margin_edge = abs(sim_margin_mean) / 10.0
    return 0.6 * prob_edge + 0.3 * sim_edge + 0.1 * margin_edge


def _rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["confidence"] = ranked.apply(
        lambda row: _confidence_signal(
            row["win_prob"], row.get("sim_fair_prob", 0.5), row.get("sim_margin_mean", 0.0)
        ),
        axis=1,
    )
    return ranked.sort_values(["confidence", "win_prob"], ascending=False).reset_index(drop=True)


def _size_stakes(
    df: pd.DataFrame,
    selected_mask: list[bool],
    config: DecisionConfig,
    *,
    bankroll: float = 1.0,
    sim_draws: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Apply fractional Kelly sizing with exposure caps and correlation guard."""

    sized = df.copy()
    sized["selected"] = selected_mask
    sized["stake_pct"] = 0.0

    if sized["selected"].any():
        sized.loc[sized["selected"], "stake_pct"] = sized.loc[sized["selected"], "win_prob"].apply(
            lambda p: _kelly_fraction(p, fraction=config.kelly_fraction)
        )

    sized["stake_pct"] = sized["stake_pct"].clip(upper=config.max_stake_pct_per_bet)

    # Cap exposure per game (games may have multiple markets).
    for game_id, group in sized.groupby("game_id"):
        total = group["stake_pct"].sum()
        if total > config.max_stake_pct_per_game and total > 0:
            scale = config.max_stake_pct_per_game / total
            sized.loc[group.index, "stake_pct"] *= scale

    # Correlation guard: downscale total exposure when simulations show high
    # cross-game co-movement.
    corr_guard = _correlation_multiplier(sized, sim_draws)

    total_weekly = sized["stake_pct"].sum() * corr_guard
    if total_weekly > config.max_weekly_exposure_pct and total_weekly > 0:
        scale = config.max_weekly_exposure_pct / total_weekly
        sized["stake_pct"] *= scale

    sized["stake"] = sized["stake_pct"] * bankroll
    return sized


def _correlation_multiplier(df: pd.DataFrame, sim_draws: Optional[pd.DataFrame]) -> float:
    """Return a multiplier >= 1 accounting for correlated exposure.

    The multiplier is 1 + mean pairwise correlation of |margin| across the games
    with active stakes.  Missing or degenerate simulations fall back to 1.
    """

    active_games = df.loc[df["stake_pct"] > 0, "game_id"].unique().tolist()
    if not active_games or sim_draws is None or sim_draws.empty:
        return 1.0

    pivot = (
        sim_draws[sim_draws["game_id"].isin(active_games)]
        .assign(abs_margin=lambda d: d["margin"].abs())
        .pivot_table(index="sample_id", columns="game_id", values="abs_margin")
    )
    if pivot.shape[1] < 2:
        return 1.0

    corr = pivot.corr(min_periods=10).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if corr.empty:
        return 1.0

    # Average the off-diagonal correlations.
    mask = ~np.eye(len(corr), dtype=bool)
    if not mask.any():
        return 1.0
    mean_corr = float(np.nanmean(corr.to_numpy()[mask]))
    return max(1.0, 1.0 + max(0.0, mean_corr))


def _historical_metrics(
    df_ranked: pd.DataFrame,
    selected_mask: Iterable[bool],
    k: int,
) -> SelectionMetrics:
    """Evaluate precision and coverage on the historical slice."""

    selected_series = pd.Series(list(selected_mask))
    ordered_actuals = df_ranked.loc[selected_series.to_numpy(), "actual"].tolist()
    coverage = selected_series.mean() if len(selected_series) else 0.0
    if not ordered_actuals:
        return SelectionMetrics(precision_at_k=0.0, coverage=float(coverage), sample_size=len(df_ranked))
    top_k = ordered_actuals[: min(k, len(ordered_actuals))]
    precision = float(sum(top_k) / len(top_k)) if top_k else 0.0
    return SelectionMetrics(precision_at_k=precision, coverage=float(coverage), sample_size=len(df_ranked))


def _artifact_stem(stamp: Optional[str] = None) -> str:
    stamp_val = stamp or datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"picks_week_{stamp_val}"


def _write_artifacts(df: pd.DataFrame, stem: str) -> None:
    csv_path = ARTIFACT_DIR / f"{stem}.csv"
    json_path = ARTIFACT_DIR / f"{stem}.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    print(f"[decision] picks written to {csv_path}")
    print(f"[decision] picks written to {json_path}")


def run(
    config_path: str | Path = "configs/defaults.yaml",
    *,
    candidates: Optional[pd.DataFrame] = None,
    sim_draws: Optional[pd.DataFrame] = None,
    bankroll: float = 1.0,
    stamp: Optional[str] = None,
) -> pd.DataFrame:
    """Entry point used by ``make decision``."""

    raw = _load_config(config_path)
    decision_cfg = DecisionConfig.from_mapping(raw.get("decision", {}))

    base = candidates.copy() if candidates is not None else _stub_candidates()
    ranked = _rank_candidates(base)

    np.random.default_rng(decision_cfg.seed)

    select_cfg = SelectivityConfig(
        prob_threshold=decision_cfg.prob_threshold,
        k_top=decision_cfg.k_top,
        mode=decision_cfg.selection_mode,
    )
    select_result = apply_selectivity(ranked, select_cfg)

    if not any(select_result.selected_mask):
        stem = _artifact_stem(stamp)
        abstain_df = ranked.assign(selected=False, stake_pct=0.0, stake=0.0)
        _write_artifacts(abstain_df, stem)
        print(
            f"[decision] precision@{decision_cfg.k_top}: 0.000 | coverage: 0.000 | exposure: 0.00% | abstained"
        )
        return abstain_df

    draws = sim_draws if sim_draws is not None else _stub_simulations(ranked)
    sized = _size_stakes(ranked, select_result.selected_mask, decision_cfg, bankroll=bankroll, sim_draws=draws)

    metrics = _historical_metrics(ranked, select_result.selected_mask, decision_cfg.k_top)
    exposure = sized["stake_pct"].sum()
    abstentions = int(len(ranked) - sized["selected"].sum())

    stem = _artifact_stem(stamp)
    _write_artifacts(sized, stem)

    print(
        f"[decision] precision@{decision_cfg.k_top}: {metrics.precision_at_k:.3f} | "
        f"coverage: {metrics.coverage:.3f}"
    )
    print(f"[decision] total exposure: {exposure:.2%} | abstentions: {abstentions}")

    return sized


def main() -> None:
    run()


if __name__ == "__main__":
    main()
