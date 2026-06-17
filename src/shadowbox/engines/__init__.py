"""Pluggable slicing engines.

New engines: implement `SlicingEngine` from `base.py`, then register a
factory callable in `_FACTORIES` below. Instances are constructed lazily
on first use so importing the package doesn't pay for model loading.
"""

from __future__ import annotations

from collections.abc import Callable

from shadowbox.engines.base import SlicingEngine
from shadowbox.engines.depth import DepthAnythingBaseEngine, DepthAnythingEngine
from shadowbox.engines.luminance import LuminanceEngine
from shadowbox.engines.palette import PaletteEngine

_FACTORIES: dict[str, Callable[[], SlicingEngine]] = {
    LuminanceEngine.name: LuminanceEngine,
    DepthAnythingEngine.name: DepthAnythingEngine,
    DepthAnythingBaseEngine.name: DepthAnythingBaseEngine,
    PaletteEngine.name: PaletteEngine,
}
_INSTANCES: dict[str, SlicingEngine] = {}


def get(name: str) -> SlicingEngine:
    if name not in _INSTANCES:
        if name not in _FACTORIES:
            available = sorted(_FACTORIES)
            raise KeyError(f"Unknown engine {name!r}. Available: {available}")
        _INSTANCES[name] = _FACTORIES[name]()
    return _INSTANCES[name]


def names() -> list[str]:
    return sorted(_FACTORIES)


__all__ = [
    "DepthAnythingBaseEngine",
    "DepthAnythingEngine",
    "LuminanceEngine",
    "PaletteEngine",
    "SlicingEngine",
    "get",
    "names",
]
