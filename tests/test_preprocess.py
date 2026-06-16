from __future__ import annotations

from PIL import Image

from shadowbox.preprocess import preprocess


def test_alpha_crop_strips_transparent_margin() -> None:
    img = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    for x in range(10, 30):
        for y in range(10, 20):
            img.putpixel((x, y), (255, 0, 0, 255))
    cropped = preprocess(img, remove_bg=False, auto_crop=True)
    assert cropped.size == (20, 10)


def test_opaque_image_passes_through() -> None:
    img = Image.new("RGB", (32, 32), (128, 128, 128))
    out = preprocess(img, remove_bg=False, auto_crop=True)
    assert out.size == (32, 32)


def test_fully_transparent_image_is_returned_unchanged() -> None:
    img = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    out = preprocess(img, remove_bg=False, auto_crop=True)
    assert out.size == (40, 40)


def test_remove_bg_can_be_stubbed(monkeypatch) -> None:
    """We don't want CI to hit the U²-Net model. Stub `rembg.remove`."""
    captured: list[Image.Image] = []

    def fake_remove(im: Image.Image) -> Image.Image:
        captured.append(im)
        # Return an RGBA copy so the alpha-crop branch has something to do.
        rgba = im.convert("RGBA")
        return rgba

    import rembg

    monkeypatch.setattr(rembg, "remove", fake_remove)

    img = Image.new("RGB", (32, 32), (200, 200, 200))
    out = preprocess(img, remove_bg=True, auto_crop=False)
    assert captured and out.mode == "RGBA"
