"""Monocular-depth slicer powered by Depth Anything v2 (ONNX).

For photographic input, luminance binning gives nonsense: bright snow ends
up "front", dark sky ends up "back", and texture survives every smoothing
level. This engine asks a vision model where each pixel sits in z, then
quantizes the depth map into N back-to-front layers — which is what a
shadow box actually wants.

Two model sizes are baked into the Docker image:
- Small (~50MB) — engine name "depth". Fast, good default.
- Base  (~195MB) — engine name "depth-hq". Sharper edges on outdoor scenes.

Quantization strategy is "foreground-biased" by default: layer 1 (back) holds
the entire silhouette (everything passing a permissive cutoff), and the
remaining N-1 layers concentrate on the subject's depth range — giving real
variation between layers instead of N near-copies. Set threshold_mode=equal
or kmeans to use linear quantization across the full depth range.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image, ImageFilter
from skimage.filters import threshold_multiotsu, threshold_otsu

_MODEL_PATHS: dict[str, str] = {
    "small": "/models/depth_anything_v2_small_fp16.onnx",
    "base": "/models/depth_anything_v2_base_fp16.onnx",
}
ENV_MODEL_PATH = "SHADOWBOX_DEPTH_MODEL"

# DPTImageProcessor defaults from the model's preprocessor_config.json.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_TARGET_SIZE = 518
_MULTIPLE_OF = 14

# Gaussian sigma applied to the depth map before quantization. Levels match
# the engine-side `smoothing` parameter so users have one knob.
_DEPTH_BLUR_BY_LEVEL: dict[int, float] = {0: 0.0, 1: 0.8, 2: 1.6, 3: 3.0}


class DepthAnythingEngine:
    """Default depth engine — uses the Small model."""

    name: ClassVar[str] = "depth"
    model_size: ClassVar[str] = "small"

    def __init__(self, model_path: str | None = None):
        # Lazy-import onnxruntime so `import shadowbox.engines` is cheap and
        # works in environments without the model.
        import onnxruntime as ort

        env_override = os.environ.get(ENV_MODEL_PATH)
        path = Path(model_path or env_override or _MODEL_PATHS[self.model_size])
        if not path.exists():
            raise FileNotFoundError(
                f"Depth model not found at {path}. The Docker image bakes one at "
                f"{_MODEL_PATHS[self.model_size]}; for local runs set {ENV_MODEL_PATH} "
                f"to a Depth Anything v2 ONNX file."
            )
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # mute INFO/WARNING noise on session init
        self._session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._input_dtype = np.float16 if "float16" in self._session.get_inputs()[0].type else np.float32

    def slice(
        self,
        image: np.ndarray,
        n_layers: int,
        *,
        threshold_mode: str = "otsu",
        invert_layers: tuple[int, ...] = (),
        smoothing: int = 2,
    ) -> list[np.ndarray]:
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        depth = self._estimate_depth(image)  # uint8; 0=far, 255=near
        depth = _smooth_depth(depth, smoothing)
        thresholds = _thresholds(depth, n_layers, threshold_mode)
        thresholds = sorted(thresholds)  # ascending; layer 0 uses the smallest
        masks = [depth >= t for t in thresholds]

        invert_set = {i - 1 for i in invert_layers}
        return [np.logical_not(m) if i in invert_set else m for i, m in enumerate(masks)]

    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Run inference; return a uint8 HxW depth map where 255 = closest."""
        if image.ndim == 2:
            rgb = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            rgb = image[:, :, :3]
        else:
            rgb = image
        orig_h, orig_w = rgb.shape[:2]

        # Aspect-preserving resize with multiple-of-14 dims (model requirement).
        resize_w, resize_h = _model_input_size(orig_w, orig_h)
        pil = Image.fromarray(rgb.astype(np.uint8)).resize((resize_w, resize_h), Image.Resampling.BICUBIC)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = np.transpose(arr, (2, 0, 1))[None, ...].astype(self._input_dtype)

        depth_raw = self._session.run(None, {self._input_name: tensor})[0]
        depth_map = np.asarray(depth_raw).squeeze().astype(np.float32)

        # Resize back to the source resolution so masks align with the input.
        depth_pil = Image.fromarray(depth_map).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        depth_resized = np.asarray(depth_pil, dtype=np.float32)

        lo, hi = float(depth_resized.min()), float(depth_resized.max())
        if hi - lo < 1e-6:
            return np.full(depth_resized.shape, 128, dtype=np.uint8)
        normalized = (depth_resized - lo) / (hi - lo) * 255.0
        return normalized.astype(np.uint8)


class DepthAnythingBaseEngine(DepthAnythingEngine):
    """Higher-quality depth — uses the Base model (~195MB)."""

    name: ClassVar[str] = "depth-hq"
    model_size: ClassVar[str] = "base"


def _model_input_size(w: int, h: int, target: int = _TARGET_SIZE, multiple: int = _MULTIPLE_OF) -> tuple[int, int]:
    """Resize so the longer side is `target` and both dims are multiples of 14."""
    scale = target / max(w, h)
    rw = max(multiple, _round_to_multiple(round(w * scale), multiple))
    rh = max(multiple, _round_to_multiple(round(h * scale), multiple))
    return rw, rh


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)


def _smooth_depth(depth: np.ndarray, level: int) -> np.ndarray:
    sigma = _DEPTH_BLUR_BY_LEVEL.get(level, _DEPTH_BLUR_BY_LEVEL[2])
    if sigma <= 0:
        return depth
    blurred = Image.fromarray(depth).filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(blurred, dtype=np.uint8)


def _thresholds(depth: np.ndarray, n_layers: int, mode: str) -> list[int]:
    """Compute N back→front cutoff values for `depth >= t` masking.

    Default "otsu" mode is foreground-biased: layer 0 covers the entire image
    (back panel), layer 1 captures everything denser than Otsu's foreground
    break (the subject silhouette), and layers 2..N-1 spread evenly through
    the foreground depth range. This gives real variation between layers
    instead of N near-copies of the same outline.

    "equal" and "kmeans" treat the depth range as a single domain — equal
    bands or k-means breaks across [min, max] — which is fine for synthetic
    inputs but produces clustered, similar-looking layers on natural photos.
    """
    if n_layers == 1:
        return [int(np.median(depth))]
    if mode == "equal":
        lo, hi = int(depth.min()), int(depth.max())
        if hi <= lo:
            return [128] * n_layers
        step = (hi - lo) / n_layers
        return [round(lo + step * i) for i in range(n_layers)]
    if mode == "otsu":
        return _foreground_biased_thresholds(depth, n_layers)
    if mode == "kmeans":
        from scipy.cluster.vq import kmeans2

        flat = depth.reshape(-1).astype(np.float32)
        seeds = np.linspace(flat.min(), flat.max(), n_layers).astype(np.float32)
        centers, _ = kmeans2(flat, seeds, minit="matrix", seed=0)
        centers = sorted(int(c) for c in centers)
        midpoints = [round((centers[i] + centers[i + 1]) / 2) for i in range(len(centers) - 1)]
        return [int(depth.min()), *midpoints]
    raise ValueError(f"Unknown threshold_mode {mode!r}. Expected: equal|otsu|kmeans")


def _foreground_biased_thresholds(depth: np.ndarray, n_layers: int) -> list[int]:
    """Layer 0 = entire silhouette; layers 1..N-1 spread across the foreground."""
    lo = int(depth.min())
    hi = int(depth.max())
    if hi - lo < n_layers:
        # Depth range too narrow for distinct bands; fall back to equal.
        step = max(1, (hi - lo) / n_layers)
        return [round(lo + step * i) for i in range(n_layers)]

    try:
        fg_break = int(threshold_otsu(depth))
    except ValueError:
        fg_break = int((lo + hi) / 2)
    fg_break = max(lo + 1, min(hi - 1, fg_break))

    # When N is very small, behave gracefully.
    if n_layers == 2:
        return [lo, fg_break]

    # Reserve layer 0 for the back panel; distribute layers 1..N-1 across
    # [fg_break, hi]. Use Otsu again on the foreground subset for natural
    # interior breaks; fall back to equal spacing if Otsu can't.
    fg_pixels = depth[depth >= fg_break]
    interior_count = n_layers - 2  # interior breaks between fg_break and hi
    interior_breaks: list[int]
    if interior_count <= 0:
        interior_breaks = []
    else:
        try:
            ms = threshold_multiotsu(fg_pixels, classes=interior_count + 1)
            interior_breaks = [int(b) for b in ms]
        except ValueError:
            step = (hi - fg_break) / (interior_count + 1)
            interior_breaks = [round(fg_break + step * (i + 1)) for i in range(interior_count)]

    return [lo, fg_break, *interior_breaks]
