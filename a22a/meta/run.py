"""Entry-point orchestrating blending, calibration, and conformal control."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from a22a.meta.blend import stack_logit
from a22a.meta.calibrate import calibrate_probs
from a22a.meta.conformal import split_conformal_binary, split_conformal_quantiles
from a22a.metrics.calibration import brier_score, ece, log_loss, reliability_bins


def _load_config(path: str = "configs/defaults.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _synthetic_inputs(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "game_id": [f"G{i:04d}" for i in range(n)],
            "p_base": np.clip(0.55 + 0.2 * rng.standard_normal(size=n), 0.01, 0.99),
            "p_strength": np.clip(0.52 + 0.15 * rng.standard_normal(size=n), 0.01, 0.99),
            "p_sim": np.clip(0.50 + 0.18 * rng.standard_normal(size=n), 0.01, 0.99),
        }
    )
    # Toy outcome: correlated with the baseline probability.
    y = (df["p_base"] > 0.5).astype(int)
    return df, y


def main() -> None:
    start = time.time()
    cfg = _load_config()

    meta_cfg = cfg.get("meta", {})
    calibrate_cfg = cfg.get("calibrate", {})
    conformal_cfg = cfg.get("conformal", {})

    outdir = pathlib.Path("artifacts/meta")
    outdir.mkdir(parents=True, exist_ok=True)

    df, y = _synthetic_inputs(n=1024, seed=int(meta_cfg.get("seed", 14)))

    # Blend base predictors.
    blend_cols = ["p_base", "p_strength", "p_sim"]
    p_blend = stack_logit(blend_cols, df, seed=int(meta_cfg.get("seed", 14)))

    # Calibrate blended probability.
    method = calibrate_cfg.get("method", "isotonic")
    p_cal, meta_info = calibrate_probs(p_blend, y, method=method)

    # Conformal for win probability.
    coverage = float(conformal_cfg.get("coverage", 0.9))
    conf_binary = split_conformal_binary(p_cal, y, coverage=coverage)
    empirical_cov = float((np.abs(y.to_numpy() - p_cal.to_numpy()) <= conf_binary["q"]).mean())

    # Conformal quantiles for simulator margin/total draws (synthetic for now).
    rng = np.random.default_rng(int(meta_cfg.get("seed", 14)))
    margin_draws = rng.normal(loc=3.0, scale=7.0, size=2048)
    total_draws = rng.normal(loc=44.0, scale=10.0, size=2048)
    margin_qs = conformal_cfg.get("margin_quantiles", [0.05, 0.95])
    total_qs = conformal_cfg.get("total_quantiles", [0.05, 0.95])
    margin_interval = split_conformal_quantiles(margin_draws, q_low=margin_qs[0], q_high=margin_qs[1])
    total_interval = split_conformal_quantiles(total_draws, q_low=total_qs[0], q_high=total_qs[1])

    # Assemble final probabilities (home vs. away) and write artifacts.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    final_df = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "p_home": p_cal,
            "p_away": 1.0 - p_cal,
        }
    )
    final_path = outdir / f"final_probs_{stamp}.parquet"
    final_df.to_parquet(final_path, index=False)

    calib_report = {
        "ece": ece(p_cal, y, bins=int(calibrate_cfg.get("bins", 10))),
        "brier": brier_score(p_cal, y),
        "log_loss": log_loss(p_cal, y),
        "calibration_method": meta_info.get("method", method),
        "reliability_bins": reliability_bins(p_cal, y, bins=int(calibrate_cfg.get("bins", 10))),
        "conformal": {
            "binary": {"nominal": coverage, "empirical": empirical_cov, **conf_binary},
            "margin": {"low": margin_interval[0], "high": margin_interval[1], "quantiles": margin_qs},
            "total": {"low": total_interval[0], "high": total_interval[1], "quantiles": total_qs},
        },
    }
    report_path = outdir / f"calibration_report_{stamp}.json"
    report_path.write_text(json.dumps(calib_report, indent=2))

    print(
        "[meta] wrote final_probs_%s and calibration_report_%s" % (stamp, stamp)
    )
    print(
        f"[meta] conformal coverage nominal={coverage:.2%} empirical={empirical_cov:.2%}"
    )
    elapsed = time.time() - start
    print(f"[meta] completed in {elapsed:.2f}s → final_probs: {final_path}")


if __name__ == "__main__":
    main()
