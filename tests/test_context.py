import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from a22a.context.game_state import build_partial_sim_package


def _read_context_path(output: str) -> Path:
    match = re.search(r"(artifacts/context/state_[\d-]+\.(?:parquet|csv))", output)
    assert match, output
    return Path(match.group(1))


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def test_context_phase10_features():
    cmd = [sys.executable, "-m", "a22a.context.game_state"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    path = _read_context_path(result.stdout)
    df = _load_table(path)

    latency_match = re.search(r"in ([0-9.]+)s \(budget ([0-9.]+)s\)", result.stdout)
    assert latency_match, result.stdout
    latency = float(latency_match.group(1))
    budget = float(latency_match.group(2))
    assert latency <= budget + 0.1

    required_cols = {
        "game_id",
        "team_id",
        "current_lead",
        "fatigue_proxy",
        "momentum_proxy",
        "pace_s_per_play",
        "timeouts_off",
        "timeouts_def",
        "aggressiveness_hint",
    }
    assert required_cols.issubset(df.columns), df.columns

    for _, group in df.groupby("team_id"):
        diffs = group.sort_values("drive_number")["fatigue_proxy"].diff().fillna(0)
        assert (diffs >= -1e-6).all()

    momentum_change = (
        df.sort_values("drive_number")
        .groupby("team_id")["momentum_proxy"]
        .diff()
        .fillna(0)
    )
    assert (momentum_change > 0).any()

    first_row = df.iloc[0]
    start = time.time()
    package = build_partial_sim_package(first_row)
    elapsed_ms = (time.time() - start) * 1000
    assert set(package.keys()) == {
        "game_id",
        "team_id",
        "score_diff",
        "expected_pace",
        "fatigue",
        "momentum",
        "timeouts",
        "state_vector",
        "aggressiveness_hint",
    }
    assert elapsed_ms <= 200.0
    assert package["timeouts"]["offense"] >= 0
