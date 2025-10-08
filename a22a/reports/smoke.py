"""Quick smoke test for the reports package."""

from __future__ import annotations

from a22a.reports.compile import main as compile_reports


def run() -> None:
    compile_reports()


if __name__ == "__main__":
    run()
