import pathlib
import subprocess
import sys


def test_monitor_runs():
    result = subprocess.run(
        [sys.executable, "-m", "a22a.monitor.run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[monitor] wrote" in result.stdout
    outputs = list(pathlib.Path("artifacts/monitor").glob("health_*.json"))
    assert outputs, "monitor should write health artifacts"
