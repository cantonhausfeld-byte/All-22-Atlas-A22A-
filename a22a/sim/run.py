"""Simulation scaffolding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List


@dataclass
class SimulationConfig:
    num_draws: int = 16
    convergence_tol: float = 0.01


class EarlyStop(Exception):
    """Raised when the simulation meets early-stop criteria."""


def quasi_monte_carlo_sampler(num_draws: int) -> List[float]:
    return [idx / num_draws for idx in range(num_draws)]


def run_simulation(config: SimulationConfig, hook: Callable[[int, float], bool]) -> List[float]:
    draws = quasi_monte_carlo_sampler(config.num_draws)
    results: List[float] = []
    for idx, draw in enumerate(draws, start=1):
        results.append(draw)
        if hook(idx, draw):
            raise EarlyStop(f"Stopped after {idx} iterations")
    return results


def fair_ladder(results: Iterable[float]) -> List[float]:
    return sorted(results)


def main() -> None:
    config = SimulationConfig()

    def early_stop_hook(iteration: int, value: float) -> bool:
        return value >= (1.0 - config.convergence_tol)

    try:
        results = run_simulation(config, early_stop_hook)
    except EarlyStop as exc:
        print(f"[sim] early-stop triggered: {exc}")
        results = []
    ladder = fair_ladder(results)
    print("[sim] ladder placeholder:", ladder)


if __name__ == "__main__":
    main()
