# shadowbox

Turn an image into N stacked SVGs sized for laser-cutting a shadow box.

Input: an image (a silhouette, illustration, photo, or line art) + a layer count.
Output: N SVG files, back-to-front, each carrying the cut paths for one layer of the stack — sized in millimeters, with a registration frame, kerf compensation, and color-coded operations that drop straight into LightBurn / xTool / Glowforge.

```
docker run --rm -v "$(pwd):/work" ghcr.io/llacoste/shadowbox:latest \
    slice INPUT.png \
        --layers 5 \
        --material birch_ply_3mm \
        --width-mm 200 --height-mm 200 \
        --output ./out/
```

Six SVGs land in `out/` (`layer_01.svg` … `layer_05.svg` plus a `preflight.txt`).

Or run the web UI:

```
docker run --rm -p 8000:8000 ghcr.io/llacoste/shadowbox:latest serve
# open http://localhost:8000
```

## What it does

The pipeline is:

1. **Load** the image.
2. **Preprocess** — optional `rembg` background removal (`--remove-bg`), alpha-aware auto-crop.
3. **Slice** — the luminance engine partitions the grayscale image into N brightness bands (equal-spaced, Otsu multi-level, or k-means). Each band becomes a binary mask describing where material remains on that layer.
4. **Vectorize** — `potracer` turns each mask into clean Bezier paths. A min-feature filter drops specks smaller than what the laser can reliably cut.
5. **Compose** — each layer becomes an SVG with the cut paths, an outer registration frame so layers align when stacked, kerf compensation (offset paths by `kerf/2` so finished pieces match nominal dimensions), and an engraved layer number in the corner.
6. **Preflight** — warnings for solid layers, floating islands, tiny features, aspect-ratio mismatches.

Color semantics in the output (matches LightBurn defaults):
- `#FF0000` (red) — interior cuts
- `#0000FF` (blue) — outer frame (cut last, after interior, so the workpiece doesn't fall free mid-cut)
- `#000000` (black) — engraved layer number

## Engines

V1 ships one engine — **luminance** — with three threshold modes:

| Mode | When it's right |
|---|---|
| `otsu` (default) | Finds natural breakpoints. Best for most photo / illustration inputs. |
| `equal` | Equal-spaced brightness bands. Best for synthetic gradients or when you want predictable, linear depth. |
| `kmeans` | K-means clustering of luminance values. Best for posterized / few-tone images. |

The architecture supports more engines (depth-ML, contour-based, color-quantization) but they're out of scope for v1.

## Materials

Built-in presets carry kerf defaults so you don't have to look them up:

```
just slice INPUT.png --material birch_ply_3mm   # kerf 0.15mm
just slice INPUT.png --material cast_acrylic_3mm # kerf 0.20mm
just slice INPUT.png --material cardstock_220gsm # kerf 0.05mm
```

`shadowbox materials list` prints the full set.

## Web UI

Two-step flow:

1. **Prepare.** Drag an image, optionally toggle background removal. The server returns a grayscale preview and a histogram; threshold sliders let you tune where the bands fall.
2. **Generate.** Pick material, dimensions, kerf, min feature size, and per-layer color swatches. The server returns inline SVG previews, a CSS-perspective stacking preview, and a preflight panel. Save the project config as JSON to come back to it later.

## Development

Everything runs inside the Docker image:

```
just build           # docker build
just slice <args>    # forward args to the CLI in the image
just serve           # FastAPI on http://localhost:8000
just test            # pytest in the image
just lint
just format-check
just typecheck
just shell           # /bin/bash inside the image
```

Releases go through CI: bump `[project].version` in `pyproject.toml`, merge to `master`, and the Deploy stage publishes `ghcr.io/llacoste/shadowbox:vX.Y.Z` + `:latest` and cuts a GitHub Release.

## Input guidance

For best results, feed shadowbox a **subject-focused image** — a silhouette, illustration, or photo of an object on a uniform background. Busy photos work with `--remove-bg` (the U²-Net model is baked into the Docker image, no first-run download). The reference image at the top of this README is a *photo of a finished build* — that's a bad input. The right input to recreate it would be the underlying daisy illustration.

## License

MIT.
