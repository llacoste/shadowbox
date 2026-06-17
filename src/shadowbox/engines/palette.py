"""Color-quantization slicer for stylized illustrations.

Photos belong to the depth engine; high-contrast silhouettes belong to
luminance. Stylized art — illustrations with distinct color regions — needs
something else: cluster pixels in perceptual color space and let each cluster
become a layer. That's the algorithmic analog of what an artist does when
they manually trace each region of an illustration into its own laser layer.

Pipeline:
1. Convert RGB to CIELAB (perceptually uniform — Euclidean distance approximates
   how different two colors look to a viewer).
2. K-means with `n_layers` clusters on a random subsample of pixels (full image
   would be slow and adds no benefit for a few thousand-cluster centers).
3. Assign every pixel to its nearest cluster center.
4. Sort clusters by lightness (L*) descending. The back layer carries the
   lightest cluster (usually the background), the front layer carries the
   darkest (usually the inked detail).
5. Each cluster's binary mask is one layer. Optional morphology cleanup at
   `smoothing>=1` removes per-pixel speckle from anti-aliased edges.

Masks are EXCLUSIVE — each pixel belongs to exactly one layer. That matches
how a stacked color-shadow-box is built: each panel carries only its own
color region; layers in front cover layers behind where they overlap.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from PIL import Image, ImageFilter
from scipy.cluster.vq import kmeans2
from scipy.ndimage import binary_closing, binary_opening
from skimage.color import rgb2lab

_MORPHOLOGY_BY_LEVEL: dict[int, tuple[int, int]] = {
    0: (0, 0),
    1: (1, 1),
    2: (2, 2),
    3: (4, 3),
}
_KMEANS_SAMPLE_LIMIT = 50_000


class PaletteEngine:
    name: ClassVar[str] = "palette"

    def slice(
        self,
        image: np.ndarray,
        n_layers: int,
        *,
        threshold_mode: str = "otsu",  # accepted for protocol parity; not used
        invert_layers: tuple[int, ...] = (),
        smoothing: int = 2,
    ) -> list[np.ndarray]:
        del threshold_mode
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        rgb = _ensure_rgb(image)
        h, w = rgb.shape[:2]

        # Pre-blur the image proportional to the smoothing level. Without this,
        # antialiased color edges in stylized illustrations get clustered into
        # spurious tiny groups, turning every color boundary into laser-thin
        # speckle. The blur is mild — just enough to collapse the AA noise.
        if smoothing > 0:
            pre_sigma = 0.4 + 0.5 * smoothing
            rgb = np.asarray(
                Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=pre_sigma)),
                dtype=np.uint8,
            )

        # rgb2lab wants float input in [0, 1]. The resulting L* ∈ [0, 100],
        # a* / b* roughly in [-128, 127].
        lab = rgb2lab(rgb.astype(np.float32) / 255.0).reshape(-1, 3).astype(np.float32)

        # Subsample for k-means: clustering 1M+ pixels gains nothing vs. 50k.
        rng = np.random.default_rng(seed=0)
        if lab.shape[0] > _KMEANS_SAMPLE_LIMIT:
            idx = rng.choice(lab.shape[0], _KMEANS_SAMPLE_LIMIT, replace=False)
            sample = lab[idx]
        else:
            sample = lab

        seeds = _spread_seeds(sample, n_layers)
        centers, _ = kmeans2(sample, seeds, minit="matrix", seed=0)

        # Assign every pixel to its nearest center via direct distance calc
        # (cheaper than scipy.cluster.vq.vq for small k and large pixel counts).
        # Shape: (pixels, k) → argmin along axis=1.
        diff = lab[:, None, :] - centers[None, :, :]
        labels = np.argmin(np.einsum("pkc,pkc->pk", diff, diff), axis=1)
        label_map = labels.reshape(h, w)

        # Back layer = lightest cluster (typically the background); front =
        # darkest (typically the inked detail).
        order = np.argsort(-centers[:, 0])  # descending L*

        masks = [label_map == order[i] for i in range(n_layers)]
        opening_r, closing_r = _MORPHOLOGY_BY_LEVEL.get(smoothing, _MORPHOLOGY_BY_LEVEL[2])
        if opening_r or closing_r:
            masks = [_cleanup(m, opening_r, closing_r) for m in masks]

        invert_set = {i - 1 for i in invert_layers}
        return [np.logical_not(m) if i in invert_set else m for i, m in enumerate(masks)]


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1).astype(np.uint8)
    if image.shape[2] == 4:
        # Drop alpha — pixel color stays the same once we've already cropped
        # against alpha in preprocess.
        return image[:, :, :3].astype(np.uint8)
    if image.shape[2] == 3:
        return image.astype(np.uint8)
    raise ValueError(f"unsupported image shape {image.shape}")


def _spread_seeds(sample: np.ndarray, k: int) -> np.ndarray:
    """Pick k seeds spread along the lightness axis so kmeans2 doesn't collapse.

    scipy's `minit="++"` is k-means++; we use deterministic L*-spaced seeds so
    the same image always produces the same layering — important for project
    save/load reproducibility.
    """
    sorted_by_l = sample[np.argsort(sample[:, 0])]
    if k == 1:
        return sorted_by_l[len(sorted_by_l) // 2 : len(sorted_by_l) // 2 + 1]
    quantile_idx = np.linspace(0, len(sorted_by_l) - 1, k).astype(int)
    return sorted_by_l[quantile_idx].astype(np.float32)


def _cleanup(mask: np.ndarray, opening_r: int, closing_r: int) -> np.ndarray:
    out = mask
    if opening_r > 0:
        out = binary_opening(out, structure=_disk(opening_r))
    if closing_r > 0:
        out = binary_closing(out, structure=_disk(closing_r))
    return out


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius
