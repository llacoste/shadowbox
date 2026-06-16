"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mountains_bytes() -> bytes:
    return (FIXTURES / "mountains.png").read_bytes()


@pytest.fixture
def line_art_bytes() -> bytes:
    return (FIXTURES / "line-art.png").read_bytes()


@pytest.fixture
def flower_bytes() -> bytes:
    return (FIXTURES / "flower-photo.jpg").read_bytes()
