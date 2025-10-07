import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _read_output_path(output: str) -> Path:
    match = re.search(r"(artifacts/strategy/coach_adapt_[\d-]+\.(?:parquet|csv))", output)
    assert match, output
    return Path(match.group(1))


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def test_strategy_phase9_outputs():
    cmd = [sys.executable, "-m", "a22a.strategy.coach_adapt"]
    r1 = subprocess.run(cmd, capture_output=True, text=True)
    assert r1.returncode == 0, r1.stdout + r1.stderr
    out_path1 = _read_output_path(r1.stdout)
    df1 = _load_table(out_path1)

    required_cols = {
        "team_id",
        "coach_id",
        "agg_index",
        "ci_lo",
        "ci_hi",
        "tags",
    }
    assert required_cols.issubset(df1.columns), df1.columns
    assert df1["agg_index"].notna().all()
    assert (df1["ci_hi"] >= df1["ci_lo"]).all()

    mono = re.search(r"behind=([0-9.]+).*ahead=([0-9.]+)", r1.stdout, re.S)
    assert mono, r1.stdout
    behind = float(mono.group(1))
    ahead = float(mono.group(2))
    assert behind > ahead, r1.stdout

    # Deterministic output with fixed seed
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    out_path2 = _read_output_path(r2.stdout)
    df2 = _load_table(out_path2)

    compare_cols = ["team_id", "coach_id", "agg_index", "ci_lo", "ci_hi", "tags"]
    df1_sorted = df1.sort_values(compare_cols).reset_index(drop=True)
    df2_sorted = df2.sort_values(compare_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(df1_sorted[compare_cols], df2_sorted[compare_cols])
