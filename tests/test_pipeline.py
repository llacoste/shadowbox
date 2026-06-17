from __future__ import annotations

from shadowbox import pipeline
from shadowbox.project import ProjectSettings


def test_end_to_end_mountains(mountains_bytes: bytes) -> None:
    settings = ProjectSettings(layers=3, threshold_mode="equal", kerf_mm=0.0, width_mm=200, height_mm=200)
    result = pipeline.process(mountains_bytes, settings)
    assert len(result.layers) == 3
    for layer in result.layers:
        # svgwrite omits the XML declaration by default; the root element is enough
        # for browser/LightBurn import. `<svg ` is the load-bearing check.
        assert layer.svg.startswith("<svg")
        assert "viewBox" in layer.svg
    # Mountains is 128x128 — assert canvas plumbed through.
    assert result.canvas_px == (128, 128)
    assert result.canvas_mm == (200.0, 200.0)


def test_invert_layers_round_trips_to_svg(mountains_bytes: bytes) -> None:
    base = pipeline.process(
        mountains_bytes,
        ProjectSettings(layers=3, threshold_mode="equal", kerf_mm=0.0),
    )
    inverted = pipeline.process(
        mountains_bytes,
        ProjectSettings(layers=3, threshold_mode="equal", kerf_mm=0.0, invert_layers=(2,)),
    )
    assert base.layers[1].svg != inverted.layers[1].svg


def test_subpixel_kerf_is_clamped_to_one_pixel(line_art_bytes: bytes) -> None:
    # Tiny kerf relative to canvas resolution used to emit a noisy "no
    # compensation applied" note. We now clamp to 1 pixel and stay silent —
    # the user almost never cares about quarter-pixel kerf precision, and
    # any kerf > 0 should still affect the output.
    base = pipeline.process(line_art_bytes, ProjectSettings(layers=2, kerf_mm=0.0, width_mm=200.0))
    nudged = pipeline.process(line_art_bytes, ProjectSettings(layers=2, kerf_mm=0.0001, width_mm=200.0))
    assert not any("Kerf" in note for note in nudged.notes)
    # Even a sub-pixel kerf should produce a visibly different result (1px dilation).
    assert base.layers[0].svg != nudged.layers[0].svg


def test_fit_aspect_preserves_image_aspect(flower_bytes: bytes) -> None:
    # The flower fixture is 128x128 → square, no change expected.
    # Build a non-square test image inline using PIL and feed it in.
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (300, 100), (50, 50, 50)).save(buf, format="PNG")
    image_bytes = buf.getvalue()

    fit = pipeline.process(image_bytes, ProjectSettings(layers=2, width_mm=200, height_mm=200, kerf_mm=0))
    stretched = pipeline.process(
        image_bytes,
        ProjectSettings(layers=2, width_mm=200, height_mm=200, kerf_mm=0, fit_aspect=False),
    )
    # In fit mode the height shrinks to preserve the 3:1 aspect → 200x66.67mm.
    assert fit.canvas_mm[0] == 200
    assert fit.canvas_mm[1] < 100
    # In stretch mode we honor the literal dims.
    assert stretched.canvas_mm == (200, 200)
