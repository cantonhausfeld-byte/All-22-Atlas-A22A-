"""
Phase 6: drive-level simulator stub with antithetic sampling and CI early-stop hooks.
"""
import numpy as np, time, yaml

def simulate_game(n=512, target_ci=0.3, rng=None):
    rng = rng or np.random.default_rng(123)
    # Placeholder: normal margin with antithetic pairs
    half = n//2
    z = rng.standard_normal(size=half)
    samples = np.concatenate([z, -z])  # antithetic
    margin = samples * 7.0  # ~7-pt sd placeholder
    mean = margin.mean()
    ci = 1.96 * margin.std(ddof=1) / np.sqrt(len(margin))
    return mean, ci

def main():
    t0 = time.time()
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    n = int(cfg["sim"]["sims_per_game"])
    target = float(cfg["sim"]["ci_width_target_margin"])
    mean, ci = simulate_game(n=n, target_ci=target)
    print(f"[sim] margin_mean={mean:.2f} ci≈{ci:.2f} (target {target})")
    print(f"[sim] done in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
