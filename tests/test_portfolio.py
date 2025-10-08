import pathlib
import subprocess
import sys

import pandas as pd
import yaml


def _run_portfolio():
    result = subprocess.run(
        [sys.executable, "-m", "a22a.portfolio.optimize"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "wrote" in result.stdout
    outdir = pathlib.Path("artifacts/portfolio")
    assert outdir.exists(), "portfolio artifacts directory missing"
    outputs = sorted(outdir.glob("picks_week_*"))
    assert outputs, "no portfolio outputs produced"
    return outputs[-1], result.stdout


def _read_output(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path.with_suffix(".csv"))
    return pd.read_csv(path)


def test_portfolio_caps_and_abstention():
    path, _ = _run_portfolio()
    df = _read_output(path)
    cfg = yaml.safe_load(pathlib.Path("configs/defaults.yaml").read_text())["portfolio"]

    assert "stake_pct" in df.columns
    assert "exposure_pct" in df.columns

    assert float(df["stake_pct"].max()) <= cfg["max_stake_pct_per_bet"] + 1e-6
    assert float(df["stake_pct"].max()) <= cfg["max_game_exposure_pct"] + 1e-6
    assert float(df["exposure_pct"].max()) <= cfg["max_weekly_exposure_pct"] + 1e-6

    near_coin = df[df["pick_prob"].sub(0.5).abs() < 1e-8]
    assert not near_coin.empty
    assert (near_coin["stake_pct"] == 0).all()
