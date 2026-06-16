"""Image preprocessing — background removal and alpha-aware auto-crop.

Runs before any engine. Engines see a clean, tightly-cropped image regardless
of what the user uploaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    pass


def preprocess(
    image: Image.Image,
    *,
    remove_bg: bool = False,
    auto_crop: bool = True,
) -> Image.Image:
    """Return a preprocessed copy of `image`.

    - `remove_bg=True` runs `rembg` (U²-Net) and replaces the background with
      transparency. ~170MB model cached on first call (already baked into the
      Docker image at build time).
    - `auto_crop=True` (default) crops to the alpha bounding box when the
      image carries an alpha channel. No-op for fully-opaque images.
    """
    out = image
    if remove_bg:
        out = _remove_background(out)
    if auto_crop:
        out = _alpha_tight_crop(out)
    return out


def _remove_background(image: Image.Image) -> Image.Image:
    # Local import — keeps `import shadowbox` cheap and avoids loading the
    # onnxruntime / model on cold start of the CLI.
    from rembg import remove

    return remove(image)


def _alpha_tight_crop(image: Image.Image) -> Image.Image:
    if image.mode not in ("RGBA", "LA"):
        return image
    alpha = image.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        # Fully transparent — nothing to crop, just return the input untouched
        # so the engine downstream produces an empty result rather than crash.
        return image
    return image.crop(bbox)
