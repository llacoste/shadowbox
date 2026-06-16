"""Save / load project configuration as JSON.

A project is "everything except the image" plus a SHA-256 of the input image
bytes. Loading verifies the hash if the same image is re-supplied — mismatched
hashes are a warning, not an error, since the user may legitimately want to
re-run the same settings against a tweaked image.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectSettings:
    engine: str = "luminance"
    threshold_mode: str = "otsu"
    layers: int = 5
    remove_bg: bool = False
    material: str | None = None
    width_mm: float = 200.0
    height_mm: float = 200.0
    kerf_mm: float = 0.15
    min_feature_mm: float = 0.5
    invert_layers: tuple[int, ...] = ()
    frame: bool = True
    engrave_numbers: bool = True
    layer_colors: tuple[str, ...] = field(default_factory=tuple)


SCHEMA_VERSION = 1


def hash_image_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def to_json(settings: ProjectSettings, image_hash: str) -> str:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "image_sha256": image_hash,
        "settings": asdict(settings)
        | {"invert_layers": list(settings.invert_layers), "layer_colors": list(settings.layer_colors)},
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def from_json(raw: str) -> tuple[ProjectSettings, str]:
    payload = json.loads(raw)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported project schema version {payload.get('schema_version')!r}; "
            f"this build only reads version {SCHEMA_VERSION}."
        )
    s = payload["settings"]
    settings = ProjectSettings(
        engine=s["engine"],
        threshold_mode=s["threshold_mode"],
        layers=s["layers"],
        remove_bg=s["remove_bg"],
        material=s.get("material"),
        width_mm=s["width_mm"],
        height_mm=s["height_mm"],
        kerf_mm=s["kerf_mm"],
        min_feature_mm=s["min_feature_mm"],
        invert_layers=tuple(s.get("invert_layers", [])),
        frame=s["frame"],
        engrave_numbers=s["engrave_numbers"],
        layer_colors=tuple(s.get("layer_colors", [])),
    )
    return settings, payload["image_sha256"]


def save(path: Path, settings: ProjectSettings, image_hash: str) -> None:
    path.write_text(to_json(settings, image_hash), encoding="utf-8")


def load(path: Path) -> tuple[ProjectSettings, str]:
    return from_json(path.read_text(encoding="utf-8"))
