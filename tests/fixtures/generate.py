"""Regenerate the committed fixture PNGs.

Run as:
    python tests/fixtures/generate.py

Outputs are deterministic — same script + same numpy = same bytes — which is
what we want for committed test fixtures. Keep each image small (~100x100 px,
single-digit KB) so the repo stays light.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).parent


def mountains(size: int = 128) -> Image.Image:
    """Stacked silhouettes — the canonical shadow-box subject."""
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    # Three overlapping triangle silhouettes, each darker than the last.
    ranges = [
        ((10, size, size - 10, int(size * 0.4), size + 10, size), 200),
        ((size * 0.1, size, size * 0.55, int(size * 0.55), size * 0.9, size), 140),
        ((size * 0.3, size, size * 0.7, int(size * 0.25), size * 1.1, size), 60),
    ]
    for poly, shade in ranges:
        draw.polygon([(poly[0], poly[1]), (poly[2], poly[3]), (poly[4], poly[5])], fill=shade)
    return img.convert("RGB")


def line_art(size: int = 128) -> Image.Image:
    """High-contrast geometric drawing — exercises the equal-band path."""
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    for i, r in enumerate(range(size // 2 - 6, 6, -10)):
        shade = 0 if i % 2 == 0 else 200
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=shade, width=3)
    return img.convert("RGB")


def busy_photo(size: int = 128) -> Image.Image:
    """Synthetic 'subject + busy background' — exercises --remove-bg flows.

    Since rembg's model expects realistic photos, this fixture won't actually
    extract cleanly. The web tests stub remove_bg; this is just a sanity input
    that survives the alpha-crop branch when bg removal is off.
    """
    rng = np.random.default_rng(seed=42)
    # Noise background + a darker centered blob.
    bg = (rng.uniform(120, 220, size=(size, size, 3))).astype(np.uint8)
    img = Image.fromarray(bg).filter(ImageFilter.GaussianBlur(radius=2))
    draw = ImageDraw.Draw(img)
    cx, cy, r = size // 2, size // 2, size // 3
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(30, 30, 30))
    draw.ellipse((cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2), fill=(140, 140, 140))
    return img


def main() -> None:
    mountains().save(HERE / "mountains.png", optimize=True)
    line_art().save(HERE / "line-art.png", optimize=True)
    busy_photo().save(HERE / "flower-photo.jpg", quality=85)
    print(f"Wrote 3 fixtures to {HERE}")


if __name__ == "__main__":
    main()
