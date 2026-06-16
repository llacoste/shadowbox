"""Image preprocessing — background removal, alpha-aware auto-crop, smoothing.

Runs before any engine. Engines see a clean, tightly-cropped image regardless
of what the user uploaded.
"""

from __future__ import annotations

from PIL import Image, ImageFilter

# Smoothing presets — Gaussian sigma is expressed as a fraction of the image's
# longest dimension, so the effect is consistent across resolutions. Photos
# need real smoothing or thresholding turns every cloud and snowflake into a
# laser-uncuttable speck; flat illustrations want little or none.
_SMOOTHING_SIGMA_FRAC: dict[int, float] = {
    0: 0.0,
    1: 0.005,
    2: 0.012,
    3: 0.025,
}


def preprocess(
    image: Image.Image,
    *,
    remove_bg: bool = False,
    auto_crop: bool = True,
    smoothing: int = 2,
) -> Image.Image:
    """Return a preprocessed copy of `image`.

    - `remove_bg=True` runs `rembg` (U²-Net) and replaces the background with
      transparency. ~170MB model cached on first call (already baked into the
      Docker image at build time).
    - `auto_crop=True` (default) crops to the alpha bounding box when the
      image carries an alpha channel. No-op for fully-opaque images.
    - `smoothing` ∈ {0,1,2,3}: scales how aggressively we blur before the
      engine sees the image. Default 2 ("medium") makes photographic inputs
      cuttable; flat illustrations are fine with 0 or 1.
    """
    out = image
    if remove_bg:
        out = _remove_background(out)
    if auto_crop:
        out = _alpha_tight_crop(out)
    if smoothing > 0:
        out = _smooth(out, smoothing)
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


def _smooth(image: Image.Image, level: int) -> Image.Image:
    """Resolution-aware Gaussian blur, sized by `level` ∈ {1,2,3}."""
    frac = _SMOOTHING_SIGMA_FRAC.get(level, _SMOOTHING_SIGMA_FRAC[2])
    if frac <= 0:
        return image
    longest = max(image.size)
    radius = max(0.5, longest * frac)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))
