"""FastAPI + HTMX web UI for shadowbox.

Single-process model: uploaded images live in a process-local dict keyed by
a short token, with a 15-minute TTL. Don't run uvicorn with multiple workers
unless you front it with sticky sessions — the cache won't be shared.
"""

from __future__ import annotations

import io
import json
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from shadowbox import materials, pipeline
from shadowbox.project import (
    ProjectSettings,
    from_json,
    hash_image_bytes,
    to_json,
)

_HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))

app = FastAPI(title="shadowbox", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")


@dataclass
class _Upload:
    raw: bytes
    filename: str
    uploaded_at: float


_UPLOAD_TTL_S = 15 * 60
_uploads: dict[str, _Upload] = {}


def _gc_uploads() -> None:
    now = time.time()
    dead = [k for k, v in _uploads.items() if now - v.uploaded_at > _UPLOAD_TTL_S]
    for k in dead:
        _uploads.pop(k, None)


def _get_upload(token: str) -> _Upload:
    _gc_uploads()
    upload = _uploads.get(token)
    if upload is None:
        raise HTTPException(status_code=404, detail="upload expired or unknown — re-upload the image")
    return upload


def _parse_inverts(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(",") if x.strip())


def _resolve_kerf(material_key: str, kerf_mm: float | None) -> float:
    if kerf_mm is not None and kerf_mm >= 0:
        return kerf_mm
    if material_key and material_key != "custom":
        return materials.get(material_key).kerf_mm
    return 0.15


def _settings_from_form(
    *,
    layers: int,
    engine: str,
    threshold_mode: str,
    remove_bg: bool,
    material_key: str,
    width_mm: float,
    height_mm: float,
    kerf_mm: float | None,
    min_feature_mm: float,
    invert_layers: str,
    frame: bool,
    engrave_numbers: bool,
    layer_colors: str,
) -> ProjectSettings:
    return ProjectSettings(
        engine=engine,
        threshold_mode=threshold_mode,
        layers=layers,
        remove_bg=remove_bg,
        material=material_key if material_key and material_key != "custom" else None,
        width_mm=width_mm,
        height_mm=height_mm,
        kerf_mm=_resolve_kerf(material_key, kerf_mm),
        min_feature_mm=min_feature_mm,
        invert_layers=_parse_inverts(invert_layers),
        frame=frame,
        engrave_numbers=engrave_numbers,
        layer_colors=tuple(c.strip() for c in layer_colors.split(",") if c.strip()),
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"materials": materials.list_materials()},
    )


@app.get("/materials.json")
async def materials_json() -> JSONResponse:
    return JSONResponse(
        [
            {"key": m.key, "label": m.label, "kerf_mm": m.kerf_mm, "thickness_mm": m.thickness_mm, "notes": m.notes}
            for m in materials.list_materials()
        ]
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, image: UploadFile = File(...)) -> HTMLResponse:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    token = uuid.uuid4().hex
    _uploads[token] = _Upload(raw=raw, filename=image.filename or "upload.png", uploaded_at=time.time())
    _gc_uploads()
    return templates.TemplateResponse(
        request,
        "_upload_ack.html",
        {"token": token, "filename": image.filename, "size_kb": len(raw) // 1024},
    )


@app.post("/process", response_class=HTMLResponse)
async def process(
    request: Request,
    token: str = Form(...),
    layers: int = Form(5),
    engine: str = Form("luminance"),
    threshold_mode: str = Form("otsu"),
    remove_bg: bool = Form(False),
    material_key: str = Form(""),
    width_mm: float = Form(200.0),
    height_mm: float = Form(200.0),
    kerf_mm: float | None = Form(None),
    min_feature_mm: float = Form(0.5),
    invert_layers: str = Form(""),
    frame: bool = Form(True),
    engrave_numbers: bool = Form(True),
    layer_colors: str = Form(""),
) -> HTMLResponse:
    upload = _get_upload(token)
    settings = _settings_from_form(
        layers=layers,
        engine=engine,
        threshold_mode=threshold_mode,
        remove_bg=remove_bg,
        material_key=material_key,
        width_mm=width_mm,
        height_mm=height_mm,
        kerf_mm=kerf_mm,
        min_feature_mm=min_feature_mm,
        invert_layers=invert_layers,
        frame=frame,
        engrave_numbers=engrave_numbers,
        layer_colors=layer_colors,
    )
    result = pipeline.process(upload.raw, settings)
    # Default per-layer colors if the user didn't pick: a gentle gray ramp.
    colors = list(settings.layer_colors)
    while len(colors) < result.layers[-1].total:
        i = len(colors)
        shade = 220 - int(160 * i / max(1, result.layers[-1].total - 1))
        colors.append(f"rgb({shade},{shade},{shade})")
    return templates.TemplateResponse(
        request,
        "_results.html",
        {
            "result": result,
            "settings": settings,
            "colors": colors,
            "token": token,
        },
    )


@app.post("/download")
async def download(
    token: str = Form(...),
    layers: int = Form(5),
    engine: str = Form("luminance"),
    threshold_mode: str = Form("otsu"),
    remove_bg: bool = Form(False),
    material_key: str = Form(""),
    width_mm: float = Form(200.0),
    height_mm: float = Form(200.0),
    kerf_mm: float | None = Form(None),
    min_feature_mm: float = Form(0.5),
    invert_layers: str = Form(""),
    frame: bool = Form(True),
    engrave_numbers: bool = Form(True),
    layer_colors: str = Form(""),
) -> StreamingResponse:
    upload = _get_upload(token)
    settings = _settings_from_form(
        layers=layers,
        engine=engine,
        threshold_mode=threshold_mode,
        remove_bg=remove_bg,
        material_key=material_key,
        width_mm=width_mm,
        height_mm=height_mm,
        kerf_mm=kerf_mm,
        min_feature_mm=min_feature_mm,
        invert_layers=invert_layers,
        frame=frame,
        engrave_numbers=engrave_numbers,
        layer_colors=layer_colors,
    )
    result = pipeline.process(upload.raw, settings)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        width = len(str(result.layers[-1].total))
        for layer in result.layers:
            zf.writestr(f"layer_{layer.index:0{width}d}.svg", layer.svg)
        preflight_text = "\n".join(
            f"[{w.severity.upper()}] layer {w.layer or '*'} {w.code}: {w.message}" for w in result.warnings
        )
        zf.writestr("preflight.txt", preflight_text + ("\n" if preflight_text else ""))
    buf.seek(0)
    base = Path(upload.filename).stem or "shadowbox"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{base}-layers.zip"'},
    )


@app.post("/project/save")
async def project_save(
    token: str = Form(...),
    layers: int = Form(5),
    engine: str = Form("luminance"),
    threshold_mode: str = Form("otsu"),
    remove_bg: bool = Form(False),
    material_key: str = Form(""),
    width_mm: float = Form(200.0),
    height_mm: float = Form(200.0),
    kerf_mm: float | None = Form(None),
    min_feature_mm: float = Form(0.5),
    invert_layers: str = Form(""),
    frame: bool = Form(True),
    engrave_numbers: bool = Form(True),
    layer_colors: str = Form(""),
) -> StreamingResponse:
    upload = _get_upload(token)
    settings = _settings_from_form(
        layers=layers,
        engine=engine,
        threshold_mode=threshold_mode,
        remove_bg=remove_bg,
        material_key=material_key,
        width_mm=width_mm,
        height_mm=height_mm,
        kerf_mm=kerf_mm,
        min_feature_mm=min_feature_mm,
        invert_layers=invert_layers,
        frame=frame,
        engrave_numbers=engrave_numbers,
        layer_colors=layer_colors,
    )
    body = to_json(settings, hash_image_bytes(upload.raw))
    base = Path(upload.filename).stem or "shadowbox"
    return StreamingResponse(
        io.BytesIO(body.encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{base}-shadowbox.json"'},
    )


@app.post("/project/load", response_class=HTMLResponse)
async def project_load(
    request: Request,
    project: UploadFile = File(...),
) -> HTMLResponse:
    raw = (await project.read()).decode("utf-8")
    try:
        settings, image_hash = from_json(raw)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid project file: {exc}") from exc
    return templates.TemplateResponse(
        request,
        "_settings_form.html",
        {
            "settings": settings,
            "image_hash": image_hash,
            "materials": materials.list_materials(),
        },
    )
