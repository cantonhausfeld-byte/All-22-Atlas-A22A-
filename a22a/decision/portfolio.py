"""Phase 7 decision engine bootstrap stubs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from a22a.decision.selectivity import SelectivityConfig, apply_selectivity


ARTIFACT_DIR = Path("artifacts/picks")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_FILE = ARTIFACT_DIR / "picks_week_stub.csv"


@dataclass(frozen=True)
class DecisionConfig:
    k_top: int
    prob_threshold: float
    kelly_fraction: float
    max_stake_pct_per_bet: float
    max_stake_pct_per_game: float
    max_weekly_exposure_pct: float

    @classmethod
    def from_mapping(cls, data: dict[str, float]) -> "DecisionConfig":
        return cls(
            k_top=int(data.get("k_top", 5)),
            prob_threshold=float(data.get("prob_threshold", 0.6)),
            kelly_fraction=float(data.get("kelly_fraction", 0.25)),
            max_stake_pct_per_bet=float(data.get("max_stake_pct_per_bet", 0.02)),
            max_stake_pct_per_game=float(data.get("max_stake_pct_per_game", 0.03)),
            max_weekly_exposure_pct=float(data.get("max_weekly_exposure_pct", 0.15)),
        )


def _load_config(path: str | Path = "configs/defaults.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text())


def _stub_candidates() -> pd.DataFrame:
    """Create stub probability + outcome table (no odds usage)."""

    return pd.DataFrame(
        [
            {"game_id": "GB@CHI", "market": "spread", "team": "CHI", "win_prob": 0.58, "actual": False},
            {"game_id": "GB@CHI", "market": "spread", "team": "GB", "win_prob": 0.52, "actual": True},
            {"game_id": "KC@DEN", "market": "moneyline", "team": "KC", "win_prob": 0.61, "actual": True},
            {"game_id": "KC@DEN", "market": "moneyline", "team": "DEN", "win_prob": 0.39, "actual": False},
            {"game_id": "SF@SEA", "market": "total_over", "team": "SF", "win_prob": 0.57, "actual": False},
        ]
    )


def _size_stakes(df: pd.DataFrame, selected_mask: list[bool], config: DecisionConfig, *, bankroll: float = 1.0) -> pd.DataFrame:
    """Apply fractional Kelly sizing with exposure caps."""

    sized = df.copy()
    sized["selected"] = selected_mask
    sized["stake_pct"] = 0.0
    sized.loc[sized["selected"], "stake_pct"] = (
        sized.loc[sized["selected"], "win_prob"].apply(lambda p: max(0.0, (2 * p) - 1))
        * config.kelly_fraction
    )
    sized["stake_pct"] = sized["stake_pct"].clip(upper=config.max_stake_pct_per_bet)

    for game_id, group in sized.groupby("game_id"):
        total = group["stake_pct"].sum()
        if total > config.max_stake_pct_per_game and total > 0:
            scale = config.max_stake_pct_per_game / total
            sized.loc[group.index, "stake_pct"] *= scale

    total_weekly = sized["stake_pct"].sum()
    if total_weekly > config.max_weekly_exposure_pct and total_weekly > 0:
        scale = config.max_weekly_exposure_pct / total_weekly
        sized["stake_pct"] *= scale

    sized["stake"] = sized["stake_pct"] * bankroll
    return sized


def run(config_path: str | Path = "configs/defaults.yaml", *, candidates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Entry point used by ``make decision``."""

    raw = _load_config(config_path)
    decision_cfg = DecisionConfig.from_mapping(raw.get("decision", {}))
    data = candidates.copy() if candidates is not None else _stub_candidates()

    select_cfg = SelectivityConfig(
        prob_threshold=decision_cfg.prob_threshold, k_top=decision_cfg.k_top
    )
    select_result = apply_selectivity(data, select_cfg)
    sized = _size_stakes(data, select_result.selected_mask, decision_cfg)

    sized.to_csv(ARTIFACT_FILE, index=False)

    print("[decision] picks written to", ARTIFACT_FILE)
    print(
        f"[decision] precision@{decision_cfg.k_top}: {select_result.metrics.precision_at_k:.3f} | "
        f"coverage: {select_result.metrics.coverage:.3f}"
    )

    return sized


def main() -> None:
    run()


if __name__ == "__main__":
    main()
