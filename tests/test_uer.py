import pandas as pd
import polars as pl

from a22a.units.uer import UER_AXES, run


def test_uer_outputs_expected_axes():
    table = run()
    for axis in UER_AXES:
        assert f"{axis}_mean" in table.columns
        assert f"{axis}_ci_low" in table.columns
        assert f"{axis}_ci_high" in table.columns


def test_uer_handles_thin_samples():
    thin = pd.DataFrame(
        [
            {
                "unit_id": "THIN_UNIT",
                "axis": axis,
                "epa_per_play": 0.02,
                "success_rate": 0.48,
                "ttt": 2.6,
                "opponent_epa_allowed": 0.0,
                "opponent_success_allowed": 0.47,
                "snaps": 5,
                "weeks_ago": 1,
                "injury_factor": 1.0,
            }
            for axis in UER_AXES
        ]
    )
    table = run(snapshots=thin)
    assert table.height > 0
    for axis in UER_AXES:
        assert table.select(pl.col(f"{axis}_mean").is_not_nan().all()).item()
