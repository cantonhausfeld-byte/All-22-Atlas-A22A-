"""Reporting and analytics entrypoints for A22A."""

from importlib import metadata as _metadata

__all__ = ["__version__"]

try:
    __version__ = _metadata.version("a22a")
except _metadata.PackageNotFoundError:  # pragma: no cover - local dev fallback
    __version__ = "0+local"
