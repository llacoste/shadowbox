from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shadowbox.web.app import app

FIXTURE = Path(__file__).parent / "fixtures" / "mountains.png"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_index_renders(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "shadowbox" in r.text
    assert "Upload" in r.text


def test_materials_endpoint(client: TestClient) -> None:
    r = client.get("/materials.json")
    assert r.status_code == 200
    body = r.json()
    assert any(m["key"] == "birch_ply_3mm" for m in body)


def test_upload_returns_token(client: TestClient) -> None:
    with FIXTURE.open("rb") as f:
        r = client.post("/upload", files={"image": ("mountains.png", f, "image/png")})
    assert r.status_code == 200
    assert "data-token=" in r.text


def test_process_returns_layer_html(client: TestClient) -> None:
    with FIXTURE.open("rb") as f:
        upload = client.post("/upload", files={"image": ("mountains.png", f, "image/png")})
    token = _extract_token(upload.text)

    r = client.post(
        "/process",
        data={"token": token, "layers": "3", "threshold_mode": "equal", "kerf_mm": "0"},
    )
    assert r.status_code == 200
    # Three layer cards + a stacking preview.
    assert r.text.count("Layer 1 / 3") == 1
    assert "stack-preview" in r.text


def test_download_returns_zip_with_layers(client: TestClient) -> None:
    with FIXTURE.open("rb") as f:
        upload = client.post("/upload", files={"image": ("mountains.png", f, "image/png")})
    token = _extract_token(upload.text)

    r = client.post(
        "/download",
        data={"token": token, "layers": "2", "threshold_mode": "equal", "kerf_mm": "0"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = zf.namelist()
    assert "layer_1.svg" in names or "layer_01.svg" in names
    assert "preflight.txt" in names


def test_expired_token_404s(client: TestClient) -> None:
    r = client.post(
        "/process",
        data={"token": "nonexistent", "layers": "3", "threshold_mode": "equal", "kerf_mm": "0"},
    )
    assert r.status_code == 404


def _extract_token(html: str) -> str:
    import re

    m = re.search(r'data-token="([0-9a-f]+)"', html)
    assert m, f"no token in: {html!r}"
    return m.group(1)
