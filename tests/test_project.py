from __future__ import annotations

import pytest

from shadowbox.project import ProjectSettings, from_json, hash_image_bytes, to_json


def test_round_trip() -> None:
    settings = ProjectSettings(
        engine="luminance",
        threshold_mode="otsu",
        layers=4,
        remove_bg=True,
        material="birch_ply_3mm",
        width_mm=180.0,
        height_mm=180.0,
        kerf_mm=0.18,
        min_feature_mm=0.4,
        invert_layers=(2, 3),
        frame=True,
        engrave_numbers=False,
        layer_colors=("#fff", "#aaa", "#555", "#000"),
    )
    image_hash = hash_image_bytes(b"hello-world")
    raw = to_json(settings, image_hash)
    loaded, loaded_hash = from_json(raw)
    assert loaded == settings
    assert loaded_hash == image_hash


def test_unsupported_schema_version_rejected() -> None:
    raw = '{"schema_version": 99, "image_sha256": "x", "settings": {}}'
    with pytest.raises(ValueError, match="schema"):
        from_json(raw)
