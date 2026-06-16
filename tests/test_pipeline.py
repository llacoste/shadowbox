from __future__ import annotations

from shadowbox import pipeline
from shadowbox.project import ProjectSettings


def test_end_to_end_mountains(mountains_bytes: bytes) -> None:
    settings = ProjectSettings(layers=3, threshold_mode="equal", kerf_mm=0.0)
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


def test_kerf_zero_when_subpixel_emits_note(line_art_bytes: bytes) -> None:
    # Tiny kerf relative to canvas resolution → sub-pixel → no compensation but a note.
    settings = ProjectSettings(layers=2, kerf_mm=0.0001, width_mm=200.0)
    result = pipeline.process(line_art_bytes, settings)
    assert any("Kerf" in note for note in result.notes)
