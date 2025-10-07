import subprocess
import sys


def test_strategy_runs():
    r = subprocess.run(
        [sys.executable, "-m", "a22a.strategy.coach_adapt"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "coach_adapt" in r.stdout
    assert "wrote" in r.stdout
