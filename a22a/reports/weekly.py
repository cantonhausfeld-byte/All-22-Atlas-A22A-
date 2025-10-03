"""Weekly report scaffolding."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

REPORTS_DIR = Path("reports")


def render_weekly_summary() -> str:
    timestamp = datetime.utcnow().isoformat()
    summary = f"Week in review generated at {timestamp}. Awaiting data sources."
    return summary


def write_report(content: str) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / "weekly.md"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    content = render_weekly_summary()
    path = write_report(content)
    print(f"[report] weekly summary written to {path}")


if __name__ == "__main__":
    main()
