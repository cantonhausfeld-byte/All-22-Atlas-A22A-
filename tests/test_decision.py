import pandas as pd

from a22a.decision.portfolio import DecisionConfig, _size_stakes
from a22a.decision.selectivity import SelectivityConfig, apply_selectivity


def test_abstains_for_coinflip_probs():
    cfg = DecisionConfig.from_mapping({})
    df = pd.DataFrame(
        [
            {"game_id": "A", "market": "spread", "team": "A", "win_prob": 0.51, "actual": False},
            {"game_id": "B", "market": "ml", "team": "B", "win_prob": 0.49, "actual": True},
        ]
    )
    select_cfg = SelectivityConfig(prob_threshold=cfg.prob_threshold, k_top=cfg.k_top)
    result = apply_selectivity(df, select_cfg)
    assert not any(result.selected_mask)


def test_exposure_caps_respected():
    cfg = DecisionConfig.from_mapping({
        "prob_threshold": 0.55,
        "k_top": 5,
        "kelly_fraction": 0.5,
        "max_stake_pct_per_bet": 0.02,
        "max_stake_pct_per_game": 0.03,
        "max_weekly_exposure_pct": 0.05,
    })
    df = pd.DataFrame(
        [
            {"game_id": "G1", "market": "spread", "team": "X", "win_prob": 0.70, "actual": True},
            {"game_id": "G1", "market": "total", "team": "Y", "win_prob": 0.68, "actual": False},
            {"game_id": "G2", "market": "ml", "team": "Z", "win_prob": 0.72, "actual": True},
        ]
    )
    select_cfg = SelectivityConfig(prob_threshold=cfg.prob_threshold, k_top=cfg.k_top)
    result = apply_selectivity(df, select_cfg)
    sized = _size_stakes(df, result.selected_mask, cfg)

    assert sized["stake_pct"].max() <= cfg.max_stake_pct_per_bet + 1e-9
    for _, group in sized.groupby("game_id"):
        assert group["stake_pct"].sum() <= cfg.max_stake_pct_per_game + 1e-9
    assert sized["stake_pct"].sum() <= cfg.max_weekly_exposure_pct + 1e-9
    assert sized.loc[~pd.Series(result.selected_mask), "stake_pct"].eq(0).all()
