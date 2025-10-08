"""External odds provider clients."""

import importlib
from types import ModuleType
from typing import Any

from .base import ClientResponse

_MODULE_CACHE: dict[str, ModuleType] = {}


def _load(name: str) -> ModuleType:
    if name not in _MODULE_CACHE:
        _MODULE_CACHE[name] = importlib.import_module(f"{__name__}.{name}")
    return _MODULE_CACHE[name]


def __getattr__(item: str) -> Any:  # pragma: no cover - exercised via imports
    if item in {"theodds_client", "sgo_client"}:
        return _load(item)
    raise AttributeError(item)


__all__ = ["ClientResponse", "theodds_client", "sgo_client"]

