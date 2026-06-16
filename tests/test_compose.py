from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from shadowbox.compose import CUT_COLOR, ENGRAVE_COLOR, FRAME_COLOR, compose_layer


def _parse(svg: str) -> ET.Element:
    # svgwrite emits an <svg xmlns="http://www.w3.org/2000/svg"> — strip the
    # namespace for easier attribute reads in assertions.
    svg_stripped = re.sub(r"\sxmlns(?::\w+)?=\"[^\"]+\"", "", svg, count=2)
    return ET.fromstring(svg_stripped)


def test_dimensions_in_mm() -> None:
    svg = compose_layer(
        ["M0 0L10 0L10 10L0 10Z"],
        canvas_px=(100, 100),
        canvas_mm=(200.0, 200.0),
        layer_index=1,
        total_layers=3,
    )
    root = _parse(svg)
    assert root.get("width") == "200.0000mm"
    assert root.get("height") == "200.0000mm"
    assert root.get("viewBox") == "0 0 100 100"


def test_color_semantics() -> None:
    svg = compose_layer(
        ["M5 5L95 5L95 95L5 95Z"],
        canvas_px=(100, 100),
        canvas_mm=(200.0, 200.0),
        layer_index=1,
        total_layers=2,
    )
    root = _parse(svg)
    cuts = root.find(".//g[@id='cuts']")
    frame = root.find(".//g[@id='frame']")
    label = root.find(".//g[@id='layer-number']")
    assert cuts is not None and cuts.get("stroke") == CUT_COLOR
    assert frame is not None and frame.get("stroke") == FRAME_COLOR
    assert label is not None and label.get("fill") == ENGRAVE_COLOR


def test_engraved_number_reads_index_over_total() -> None:
    svg = compose_layer(
        [],
        canvas_px=(100, 100),
        canvas_mm=(200.0, 200.0),
        layer_index=2,
        total_layers=5,
    )
    root = _parse(svg)
    text = root.find(".//g[@id='layer-number']/text")
    assert text is not None
    assert text.text == "2/5"


def test_frame_toggle_off_omits_frame() -> None:
    svg = compose_layer(
        [],
        canvas_px=(50, 50),
        canvas_mm=(100.0, 100.0),
        layer_index=1,
        total_layers=1,
        frame=False,
    )
    assert 'id="frame"' not in svg


def test_engrave_toggle_off_omits_number() -> None:
    svg = compose_layer(
        [],
        canvas_px=(50, 50),
        canvas_mm=(100.0, 100.0),
        layer_index=1,
        total_layers=1,
        engrave_number=False,
    )
    assert 'id="layer-number"' not in svg


def test_invalid_dimensions_raise() -> None:
    with pytest.raises(ValueError):
        compose_layer([], canvas_px=(0, 100), canvas_mm=(100.0, 100.0), layer_index=1, total_layers=1)
