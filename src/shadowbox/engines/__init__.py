"""Pluggable slicing engines.

New engines: implement `SlicingEngine` from `base.py` and register here so
`get(name)` can resolve them.
"""

from __future__ import annotations

from shadowbox.engines.base import SlicingEngine
from shadowbox.engines.luminance import LuminanceEngine

_REGISTRY: dict[str, SlicingEngine] = {
    LuminanceEngine.name: LuminanceEngine(),
}


def get(name: str) -> SlicingEngine:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = sorted(_REGISTRY)
        raise KeyError(f"Unknown engine {name!r}. Available: {available}") from exc


def names() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["LuminanceEngine", "SlicingEngine", "get", "names"]
