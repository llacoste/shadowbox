from __future__ import annotations

import numpy as np
import pytest

from shadowbox.engines.luminance import LuminanceEngine


def _gradient(size: int = 64) -> np.ndarray:
    """Vertical luminance gradient — top row is 0, bottom row is 255."""
    row = np.linspace(0, 255, size).astype(np.uint8)
    return np.tile(row[:, None], (1, size))


def test_equal_bands_partition_a_gradient_into_equal_areas() -> None:
    engine = LuminanceEngine()
    img = _gradient(64)
    masks = engine.slice(img, n_layers=4, threshold_mode="equal")
    assert len(masks) == 4
    # Cumulative thresholds → strictly shrinking masks back→front.
    sums = [int(m.sum()) for m in masks]
    assert sums == sorted(sums, reverse=True)
    # Layer 1 holds everything; layer 4 holds only the darkest band.
    assert sums[0] > sums[-1]


def test_otsu_falls_back_on_constant_image() -> None:
    engine = LuminanceEngine()
    flat = np.full((32, 32), 128, dtype=np.uint8)
    masks = engine.slice(flat, n_layers=3, threshold_mode="otsu")
    assert len(masks) == 3
    # Constant image: degenerate, but engine must not crash.
    for m in masks:
        assert m.shape == flat.shape


def test_invert_layers_flips_specified_indices() -> None:
    engine = LuminanceEngine()
    img = _gradient(32)
    base = engine.slice(img, n_layers=3, threshold_mode="equal")
    inverted = engine.slice(img, n_layers=3, threshold_mode="equal", invert_layers=(2,))
    assert np.array_equal(inverted[0], base[0])
    assert np.array_equal(inverted[1], ~base[1])
    assert np.array_equal(inverted[2], base[2])


def test_kmeans_returns_correct_layer_count() -> None:
    engine = LuminanceEngine()
    img = _gradient(48)
    masks = engine.slice(img, n_layers=3, threshold_mode="kmeans")
    assert len(masks) == 3
    sums = [int(m.sum()) for m in masks]
    assert sums == sorted(sums, reverse=True)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="threshold_mode"):
        LuminanceEngine().slice(_gradient(32), n_layers=2, threshold_mode="bogus")


def test_zero_layers_rejected() -> None:
    with pytest.raises(ValueError, match="n_layers"):
        LuminanceEngine().slice(_gradient(32), n_layers=0)


def test_smoothing_removes_isolated_specks() -> None:
    """Single-pixel dark specks on a bright field should be erased at smoothing>=2."""
    # Bright canvas (gray=240) with isolated 1-pixel dark specks (gray=40).
    # The front layer's mask captures only dark pixels — each speck becomes a
    # tiny True island. Opening drops islands smaller than the disk radius.
    canvas = np.full((64, 64), 240, dtype=np.uint8)
    rng = np.random.default_rng(seed=7)
    ys = rng.integers(4, 60, size=150)
    xs = rng.integers(4, 60, size=150)
    canvas[ys, xs] = 40

    engine = LuminanceEngine()
    raw = engine.slice(canvas, n_layers=2, threshold_mode="equal", smoothing=0)
    cleaned = engine.slice(canvas, n_layers=2, threshold_mode="equal", smoothing=2)

    # Front (front-most) mask: dark pixels only. Raw should keep most specks
    # as True; cleaned should drop nearly all of them.
    front_raw = raw[-1]
    front_clean = cleaned[-1]
    assert front_raw.sum() > 100  # most specks present
    assert front_clean.sum() < front_raw.sum() / 3  # opening killed the islands
