"""Unit-level analytics scaffolding (Phase 8 stubs)."""

__all__ = ["run_uer"]


def __getattr__(name: str):
    if name == "run_uer":
        from .uer import run as _run

        return _run
    raise AttributeError(name)
