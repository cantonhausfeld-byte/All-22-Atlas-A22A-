import os, sys, time, hashlib, re, pathlib, json
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

    # Directory structure sanity
    expected_dirs = ["data", "a22a", "configs"]
    for d in expected_dirs:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)
    print("[fs] structure OK and writable")

    # Timing check placeholder
    dur = time.time() - start
    print(f"[doctor] completed in {dur:.2f}s")
    return ok_py

if __name__ == "__main__":
    ci = "--ci" in sys.argv
    ok = run_doctor(ci=ci)
    sys.exit(0 if ok else 1)
