"""Monocular-depth slicer powered by Depth Anything v2 (Small, ONNX).

For photographic input, luminance binning gives nonsense: bright snow ends
up "front", dark sky ends up "back", and texture survives every smoothing
level. This engine asks a vision model where each pixel sits in z, then
quantizes the depth map into N back-to-front layers — which is what a
shadow box actually wants.

Model: Depth Anything v2 Small, fp16-quantized ONNX (~50MB). Baked into the
Docker image at build time so first-run latency is just ONNX session init,
not a model download.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image
from skimage.filters import threshold_multiotsu

DEFAULT_MODEL_PATH = "/models/depth_anything_v2_small_fp16.onnx"
ENV_MODEL_PATH = "SHADOWBOX_DEPTH_MODEL"

# DPTImageProcessor defaults from the model's preprocessor_config.json.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_TARGET_SIZE = 518
_MULTIPLE_OF = 14


class DepthAnythingEngine:
    name: ClassVar[str] = "depth"

    def __init__(self, model_path: str | None = None):
        # Lazy-import onnxruntime so `import shadowbox.engines` is cheap and
        # works in environments without the model. We only pay the import cost
        # when someone actually instantiates the engine.
        import onnxruntime as ort

        path = Path(model_path or os.environ.get(ENV_MODEL_PATH, DEFAULT_MODEL_PATH))
        if not path.exists():
            raise FileNotFoundError(
                f"Depth model not found at {path}. The Docker image bakes it at "
                f"{DEFAULT_MODEL_PATH}; for local runs set {ENV_MODEL_PATH} to the "
                f"ONNX file path."
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
        smoothing: int = 2,  # accepted for protocol parity; the depth engine doesn't use it
    ) -> list[np.ndarray]:
        del smoothing  # mark intentionally unused
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        depth = self._estimate_depth(image)  # uint8 [0,255]; 0=far, 255=near
        # Back layer holds everything *at-or-beyond* a far threshold; front
        # layer holds only the nearest pixels. depth higher = closer, so for
        # `back→front` ordering we want thresholds DESCENDING: layer 0 keeps
        # pixels with depth >= a loose (low) value (most of the image),
        # layer N-1 keeps only the closest stuff.
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
        # HWC → NCHW
        tensor = np.transpose(arr, (2, 0, 1))[None, ...].astype(self._input_dtype)

        depth_raw = self._session.run(None, {self._input_name: tensor})[0]
        # Model returns shape (1, H, W) or (1, 1, H, W).
        depth_map = np.asarray(depth_raw).squeeze().astype(np.float32)

        # Resize back to the source resolution so masks align with the input.
        depth_pil = Image.fromarray(depth_map).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        depth_resized = np.asarray(depth_pil, dtype=np.float32)

        # Normalize to [0, 255]. Higher raw values = closer for this model.
        lo, hi = float(depth_resized.min()), float(depth_resized.max())
        if hi - lo < 1e-6:
            return np.full(depth_resized.shape, 128, dtype=np.uint8)
        normalized = (depth_resized - lo) / (hi - lo) * 255.0
        return normalized.astype(np.uint8)


def _model_input_size(w: int, h: int, target: int = _TARGET_SIZE, multiple: int = _MULTIPLE_OF) -> tuple[int, int]:
    """Resize so the longer side is `target` and both dims are multiples of 14."""
    scale = target / max(w, h)
    rw = max(multiple, _round_to_multiple(round(w * scale), multiple))
    rh = max(multiple, _round_to_multiple(round(h * scale), multiple))
    return rw, rh


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)


def _thresholds(depth: np.ndarray, n_layers: int, mode: str) -> list[int]:
    """Pick N-1 interior break points plus a low edge, then return N values."""
    if n_layers == 1:
        return [int(np.median(depth))]
    if mode == "equal":
        lo, hi = int(depth.min()), int(depth.max())
        if hi <= lo:
            return [128] * n_layers
        step = (hi - lo) / n_layers
        return [round(lo + step * i) for i in range(n_layers)]
    if mode == "otsu":
        try:
            breaks = threshold_multiotsu(depth, classes=n_layers).astype(int).tolist()
        except ValueError:
            return _thresholds(depth, n_layers, "equal")
        # threshold_multiotsu returns N-1 interior breaks; prepend min for the
        # back-layer threshold (loosest cutoff).
        return [int(depth.min()), *breaks]
    if mode == "kmeans":
        from scipy.cluster.vq import kmeans2

        flat = depth.reshape(-1).astype(np.float32)
        seeds = np.linspace(flat.min(), flat.max(), n_layers).astype(np.float32)
        centers, _ = kmeans2(flat, seeds, minit="matrix", seed=0)
        centers = sorted(int(c) for c in centers)
        midpoints = [round((centers[i] + centers[i + 1]) / 2) for i in range(len(centers) - 1)]
        return [int(depth.min()), *midpoints]
    raise ValueError(f"Unknown threshold_mode {mode!r}. Expected: equal|otsu|kmeans")
