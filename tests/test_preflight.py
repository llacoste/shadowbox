from __future__ import annotations

from shadowbox.preflight import analyze


def test_solid_layer_flagged() -> None:
    warnings = analyze(
        [["M0 0L10 0L10 10L0 10Z"], []],
        canvas_px=(100, 100),
        canvas_mm=(200.0, 200.0),
        min_feature_mm=0.5,
        image_aspect=1.0,
    )
    codes = {(w.layer, w.code) for w in warnings}
    assert (2, "solid_layer") in codes


def test_aspect_mismatch_flagged() -> None:
    warnings = analyze(
        [["M0 0L10 0L10 10L0 10Z"]],
        canvas_px=(100, 100),
        canvas_mm=(200.0, 100.0),  # output 2:1 vs image 1:1
        min_feature_mm=0.5,
        image_aspect=1.0,
    )
    assert any(w.code == "aspect_mismatch" for w in warnings)


def test_tiny_features_flagged() -> None:
    # 100px maps to 200mm → 0.5 px/mm. min_feature_mm=2 → 1px threshold.
    # A 1-px-wide path falls below that.
    warnings = analyze(
        [["M0 0L0.5 0L0.5 0.5L0 0.5Z"]],
        canvas_px=(100, 100),
        canvas_mm=(200.0, 200.0),
        min_feature_mm=2.0,
        image_aspect=1.0,
    )
    assert any(w.code == "tiny_features" for w in warnings)


def test_clean_input_no_warnings() -> None:
    warnings = analyze(
        [["M5 5L95 5L95 95L5 95Z"]],
        canvas_px=(100, 100),
        canvas_mm=(200.0, 200.0),
        min_feature_mm=0.5,
        image_aspect=1.0,
    )
    # Aspect matches, layer has a path, paths are not tiny — should be clean.
    assert warnings == []
