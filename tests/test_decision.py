from textwrap import dedent

import pandas as pd

from a22a.decision.portfolio import DecisionConfig, _size_stakes, run
from a22a.decision.selectivity import SelectivityConfig, SelectionMode, apply_selectivity


def test_abstains_for_coinflip_probs():
    cfg = DecisionConfig.from_mapping({})
    df = pd.DataFrame(
        [
            {"game_id": "A", "market": "spread", "side": "A", "win_prob": 0.51, "actual": False},
            {"game_id": "B", "market": "ml", "side": "B", "win_prob": 0.49, "actual": True},
        ]
    )
    select_cfg = SelectivityConfig(prob_threshold=cfg.prob_threshold, k_top=cfg.k_top, mode=SelectionMode.HYBRID)
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
            {"game_id": "G1", "market": "spread", "side": "X", "win_prob": 0.70, "actual": True},
            {"game_id": "G1", "market": "total", "side": "Y", "win_prob": 0.68, "actual": False},
            {"game_id": "G2", "market": "ml", "side": "Z", "win_prob": 0.72, "actual": True},
        ]
    )
    select_cfg = SelectivityConfig(prob_threshold=cfg.prob_threshold, k_top=cfg.k_top, mode=SelectionMode.HYBRID)
    result = apply_selectivity(df, select_cfg)
    sized = _size_stakes(df, result.selected_mask, cfg)

    assert sized["stake_pct"].max() <= cfg.max_stake_pct_per_bet + 1e-9
    for _, group in sized.groupby("game_id"):
        assert group["stake_pct"].sum() <= cfg.max_stake_pct_per_game + 1e-9
    assert sized["stake_pct"].sum() <= cfg.max_weekly_exposure_pct + 1e-9
    assert sized.loc[~pd.Series(result.selected_mask), "stake_pct"].eq(0).all()


def test_correlation_guard_rescales_exposure():
    cfg = DecisionConfig.from_mapping(
        {
            "prob_threshold": 0.5,
            "k_top": 3,
            "kelly_fraction": 0.75,
            "max_stake_pct_per_bet": 0.1,
            "max_stake_pct_per_game": 0.2,
            "max_weekly_exposure_pct": 0.12,
            "selection_mode": "top_k",
            "seed": 7,
        }
    )
    df = pd.DataFrame(
        [
            {"game_id": "G1", "market": "ml", "side": "A", "win_prob": 0.72, "actual": True},
            {"game_id": "G2", "market": "ml", "side": "B", "win_prob": 0.73, "actual": False},
        ]
    )
    select_cfg = SelectivityConfig(prob_threshold=0.5, k_top=2, mode=SelectionMode.TOP_K)
    selection = apply_selectivity(df, select_cfg)
    draws = pd.DataFrame(
        [
            {"game_id": "G1", "sample_id": i, "margin": 5.0} for i in range(50)
        ]
        + [
            {"game_id": "G2", "sample_id": i, "margin": 5.0} for i in range(50)
        ]
    )
    sized = _size_stakes(df, selection.selected_mask, cfg, sim_draws=draws)
    assert sized["stake_pct"].sum() <= cfg.max_weekly_exposure_pct + 1e-9


def test_run_deterministic_with_seed(tmp_path):
    candidates = pd.DataFrame(
        [
            {
                "game_id": "HOU@TEN",
                "market": "spread",
                "side": "HOU",
                "win_prob": 0.62,
                "sim_margin_mean": 4.2,
                "sim_margin_std": 5.1,
                "sim_fair_prob": 0.61,
                "actual": True,
            },
            {
                "game_id": "HOU@TEN",
                "market": "spread",
                "side": "TEN",
                "win_prob": 0.38,
                "sim_margin_mean": -4.2,
                "sim_margin_std": 5.1,
                "sim_fair_prob": 0.39,
                "actual": False,
            },
            {
                "game_id": "MIA@BUF",
                "market": "ml",
                "side": "BUF",
                "win_prob": 0.57,
                "sim_margin_mean": 2.7,
                "sim_margin_std": 4.4,
                "sim_fair_prob": 0.58,
                "actual": False,
            },
        ]
    )
    draws = pd.DataFrame(
        [
            {"game_id": "HOU@TEN", "sample_id": i, "margin": 4.0 + (i % 3)}
            for i in range(64)
        ]
        + [
            {"game_id": "MIA@BUF", "sample_id": i, "margin": 3.0 - (i % 2)}
            for i in range(64)
        ]
    )

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        dedent(
            """
            decision:
              prob_threshold: 0.55
              k_top: 2
              kelly_fraction: 0.4
              max_stake_pct_per_bet: 0.08
              max_stake_pct_per_game: 0.15
              max_weekly_exposure_pct: 0.2
              selection_mode: hybrid
              seed: 123
            """
        ).strip()
    )

    first = run(
        config_path=cfg_path,
        candidates=candidates,
        sim_draws=draws,
        bankroll=100.0,
        stamp="unit",
    )
    second = run(
        config_path=cfg_path,
        candidates=candidates,
        sim_draws=draws,
        bankroll=100.0,
        stamp="unit",
    )

    pd.testing.assert_frame_equal(first, second)

