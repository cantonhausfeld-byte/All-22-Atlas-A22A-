import os, sys, time, hashlib, re, pathlib, json, subprocess
from dataclasses import dataclass

BLOCKED_TOKENS = [
    "the-odds-api", "SportsgameOdds", "sportsbook", "oddsapi", "odds-api"
]

@dataclass
class SLOs:
    ingest_minutes_per_season: int = 5
    features_minutes: int = 2
    baseline_train_minutes_per_season: int = 3
    team_strength_minutes: int = 2
    sim_full_slate_seconds: int = 60

def hash_repo(root="a22a"):
    h = hashlib.sha256()
    for p in sorted(pathlib.Path(root).rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]

def static_scan_for_odds(root="a22a"):
    offenders = []
    for p in pathlib.Path(root).rglob("*.py"):
        if p.name == "doctor.py" and 'tools' in p.parts:
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if any(tok.lower() in txt.lower() for tok in BLOCKED_TOKENS):
            offenders.append(str(p))
    return offenders


def check_for_odds_imports(root="a22a"):
    offenders = []
    for p in pathlib.Path(root).rglob("*.py"):
        if p.name == "doctor.py" and "tools" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import", "from")) and "odds" in stripped.lower():
                offenders.append(str(p))
                break
    return offenders

def check_env_only_secrets():
    # Ensure no hardcoded API keys in repo
    offenders = []
    for p in pathlib.Path("a22a").rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"(OPENAI|ODDS|SPORTSGAMEODDS|WEATHER)_API_KEY\s*=", text):
            offenders.append(str(p))
    return offenders

def run_doctor(ci=False) -> bool:
    start = time.time()
    print("A22A Doctor — starting …")

    # Python version
    ok_py = sys.version_info >= (3, 11)
    print(f"[env] Python: {sys.version.split()[0]} (>=3.11 required) -> {'OK' if ok_py else 'FAIL'}")

    # Repo hash (determinism)
    run_id = hash_repo()
    print(f"[repro] code_hash={run_id}")

    # SLOs (placeholders from config/defaults.yaml if present)
    slos = SLOs()
    print(f"[slo] budgets: {slos.__dict__}")

    # Secrets policy (env only)
    offenders = check_env_only_secrets()
    if offenders:
        print("[secrets] FAIL: suspected hardcoded keys in:", offenders)
        return False
    else:
        print("[secrets] OK: no hardcoded API keys detected")

    # Static scan for odds leakage in core modules
    odds_hits = static_scan_for_odds("a22a")
    if odds_hits:
        print("[odds] FAIL: prohibited odds tokens found in:", odds_hits)
        return False
    else:
        print("[odds] OK: no prohibited odds usage in code")

    odds_import_hits = check_for_odds_imports("a22a")
    if odds_import_hits:
        print("[odds-imports] FAIL: odds-related imports detected in", odds_import_hits)
        return False
    else:
        print("[odds-imports] OK: no odds imports detected")

    # Decision module import guard
    decision_import_offenders = []
    for path in pathlib.Path("a22a/decision").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.strip().startswith(("import", "from")) and "odds" in line.lower():
                decision_import_offenders.append(str(path))
                break
    if decision_import_offenders:
        print("[decision] FAIL: odds-related imports detected in", decision_import_offenders)
        return False
    print("[decision] imports OK (no odds modules)")

    # Directory structure sanity
    expected_dirs = ["data", "a22a", "configs"]
    for d in expected_dirs:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)
    print("[fs] structure OK and writable")

    # Module presence report
    print("[modules] decision present:", pathlib.Path("a22a/decision").exists())
    print("[modules] units present:", pathlib.Path("a22a/units").exists())
    print("[modules] strategy present:", pathlib.Path("a22a/strategy").exists())
    print("[modules] context present:", pathlib.Path("a22a/context").exists())
    print("[modules] health present:", pathlib.Path("a22a/health").exists())
    print("[modules] roster present:", pathlib.Path("a22a/roster").exists())
    print("[modules] impact present:", pathlib.Path("a22a/impact").exists())
    print("[modules] meta present:", pathlib.Path("a22a/meta").exists())
    print("[modules] portfolio present:", pathlib.Path("a22a/portfolio").exists())

    # Quick runtime taps (best-effort)
    tap_targets = {
        "decision": 10,
        "portfolio": 10,
        "uer": 10,
        "strategy": 10,
        "context": 10,
        "injuries": 6,
        "depth": 6,
        "impact": 90,
        "meta": 20,
    }
    for label, timeout in tap_targets.items():
        try:
            tap_start = time.time()
            result = subprocess.run(
                ["make", label],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                text=True,
            )
            status = "OK" if result.returncode == 0 else f"exit={result.returncode}"
            elapsed = time.time() - tap_start
            print(f"[tap] make {label}: {status} in {elapsed:.2f}s")
            if result.stdout:
                preview = "\n".join(result.stdout.splitlines()[:5])
                print(f"[tap] output preview:\n{preview}")
        except subprocess.TimeoutExpired:
            print(f"[tap] make {label}: timeout after {timeout}s (non-fatal)")
        except FileNotFoundError:
            print(f"[tap] make {label}: make not available (non-fatal)")

    for namespace in ("health", "roster"):
        pkg_path = pathlib.Path(f"a22a/{namespace}")
        if not pkg_path.exists():
            continue
        offenders = []
        for path in pkg_path.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if line.strip().startswith(("import", "from")) and "odds" in line.lower():
                    offenders.append(str(path))
                    break
        if offenders:
            print(f"[{namespace}] FAIL: odds-related imports detected in {offenders}")
            return False
        print(f"[{namespace}] imports OK (no odds modules)")

    # Timing check placeholder
    dur = time.time() - start
    print(f"[doctor] completed in {dur:.2f}s")
    return ok_py

if __name__ == "__main__":
    ci = "--ci" in sys.argv
    ok = run_doctor(ci=ci)
    sys.exit(0 if ok else 1)
