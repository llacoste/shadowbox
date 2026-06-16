"""Material presets — kerf and thickness defaults for common laser stock.

Kerf is the width of material removed by the laser. Pieces cut at nominal
dimensions come out undersized by the kerf, so the composer offsets cut paths
outward by kerf/2 on each side. These defaults are conservative — your machine
and material batch will vary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    key: str
    label: str
    kerf_mm: float
    thickness_mm: float
    notes: str = ""


_MATERIALS: tuple[Material, ...] = (
    Material("birch_ply_3mm", "Birch plywood 3mm", 0.15, 3.0, "Most forgiving stock for shadow boxes."),
    Material("mdf_3mm", "MDF 3mm", 0.20, 3.0, "Cheap, paintable. Smells bad while cutting."),
    Material(
        "cast_acrylic_3mm",
        "Cast acrylic 3mm",
        0.20,
        3.0,
        "Glass-clear edges. Extruded acrylic burns differently — avoid.",
    ),
    Material("cast_acrylic_1_8in", 'Cast acrylic 1/8"', 0.20, 3.175, "Imperial stand-in for 3mm."),
    Material("cardstock_220gsm", "Cardstock 220gsm", 0.05, 0.3, "Cheap prototyping material. Fast cuts."),
    Material("chipboard_2mm", "Chipboard 2mm", 0.10, 2.0, "Light, rigid, paintable. Good for layered art."),
)


def list_materials() -> list[Material]:
    return list(_MATERIALS)


def get(key: str) -> Material:
    for mat in _MATERIALS:
        if mat.key == key:
            return mat
    raise KeyError(f"Unknown material preset: {key!r}. Available: {[m.key for m in _MATERIALS]}")
