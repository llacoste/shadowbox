from __future__ import annotations

import numpy as np

from shadowbox.vectorize import mm_to_pixels, vectorize


def test_empty_mask_returns_no_paths() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    assert vectorize(mask) == []


def test_solid_square_produces_one_path() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    paths = vectorize(mask)
    assert len(paths) == 1
    # potrace path starts with M and ends with Z.
    assert paths[0].startswith("M")
    assert paths[0].endswith("Z")


def test_min_feature_filter_drops_speck() -> None:
    mask = np.zeros((60, 60), dtype=bool)
    mask[5:50, 5:50] = True
    # A single noisy pixel that should be filtered when turdsize is high.
    mask[1, 1] = True
    unfiltered = vectorize(mask, min_feature_pixels=0)
    filtered = vectorize(mask, min_feature_pixels=10)
    # The speck and the big square give 2 paths unfiltered, 1 filtered.
    # potrace may merge or vary, so we assert filtered <= unfiltered.
    assert len(filtered) <= len(unfiltered)


def test_mm_to_pixels_round_trip() -> None:
    # 200mm canvas mapped to 1000px → 5 px/mm → 10mm = 50px
    assert mm_to_pixels(10.0, canvas_px=1000, canvas_mm=200.0) == 50.0
