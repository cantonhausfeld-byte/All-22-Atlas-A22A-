"""
Phase 2: creates staged/ directories and no-op ingest until sources are wired.
"""
import os, pathlib, yaml

DEFAULTS = "configs/defaults.yaml"

def main():
    cfg = {}
    if pathlib.Path(DEFAULTS).exists():
        cfg = yaml.safe_load(open(DEFAULTS))
    staged = pathlib.Path(cfg.get("paths",{}).get("staged","./data/staged"))
    staged.mkdir(parents=True, exist_ok=True)
    (staged / ".keep").write_text("staged ready\n")
    print(f"[ingest] staged path ready at: {staged.resolve()}")
    print("[ingest] no-op complete (wire sources in Phase 2).")

if __name__ == "__main__":
    main()
