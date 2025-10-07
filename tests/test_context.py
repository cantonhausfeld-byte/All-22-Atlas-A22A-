import re
import subprocess
import sys


def test_context_runs_fast():
    r = subprocess.run(
        [sys.executable, "-m", "a22a.context.game_state"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "state_" in r.stdout
    match = re.search(r"in ([0-9.]+)s \(budget ([0-9.]+)s\)", r.stdout)
    assert match, r.stdout
    latency = float(match.group(1))
    budget = float(match.group(2))
    assert latency <= budget + 0.2
