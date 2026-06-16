from __future__ import annotations

import pytest

from shadowbox import materials


def test_list_materials_nonempty() -> None:
    presets = materials.list_materials()
    assert len(presets) >= 5
    keys = {m.key for m in presets}
    assert {"birch_ply_3mm", "cast_acrylic_3mm", "cardstock_220gsm"} <= keys


def test_get_known_preset() -> None:
    m = materials.get("birch_ply_3mm")
    assert m.kerf_mm == pytest.approx(0.15)
    assert m.thickness_mm == pytest.approx(3.0)


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError):
        materials.get("unobtainium")
