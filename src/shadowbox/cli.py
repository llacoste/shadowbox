"""Typer CLI: `shadowbox slice ...`, `shadowbox project save|load`, `shadowbox materials list`, `shadowbox serve`."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from shadowbox import engines, materials, pipeline
from shadowbox.project import ProjectSettings, hash_image_bytes, load, save

app = typer.Typer(no_args_is_help=True, help=__doc__, add_completion=False)
project_app = typer.Typer(no_args_is_help=True, help="Save / load full project configurations.")
materials_app = typer.Typer(no_args_is_help=True, help="Built-in laser-material presets.")
app.add_typer(project_app, name="project")
app.add_typer(materials_app, name="materials")


def _resolve_kerf(material: str | None, kerf_mm: float | None) -> float:
    if kerf_mm is not None:
        return kerf_mm
    if material is not None:
        return materials.get(material).kerf_mm
    return 0.15  # birch_ply_3mm default


def _emit(result: pipeline.PipelineResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    width = len(str(result.layers[-1].total))
    for layer in result.layers:
        target = output_dir / f"layer_{layer.index:0{width}d}.svg"
        target.write_text(layer.svg, encoding="utf-8")
    preflight_text = "\n".join(
        f"[{w.severity.upper()}] layer {w.layer or '*'} {w.code}: {w.message}" for w in result.warnings
    )
    (output_dir / "preflight.txt").write_text(preflight_text + ("\n" if preflight_text else ""), encoding="utf-8")
    for note in result.notes:
        typer.echo(f"note: {note}", err=True)


def _exit_on_errors(result: pipeline.PipelineResult, force: bool) -> None:
    errors = [w for w in result.warnings if w.severity == "error"]
    if errors and not force:
        for e in errors:
            typer.echo(f"error: layer {e.layer} {e.code}: {e.message}", err=True)
        typer.echo("Refusing to proceed — pass --force to ignore.", err=True)
        raise typer.Exit(code=1)


@app.command(name="slice")
def slice_cmd(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, metavar="INPUT"),
    layers: int = typer.Option(5, "--layers", "-n", min=1, help="Number of stacked layers."),
    output: Path = typer.Option(Path("./out"), "--output", "-o", help="Directory for the SVGs."),
    engine: str = typer.Option("luminance", "--engine", help=f"Slicing engine: {engines.names()}"),
    threshold_mode: str = typer.Option("otsu", "--threshold-mode", help="equal | otsu | kmeans"),
    remove_bg: bool = typer.Option(False, "--remove-bg/--no-remove-bg", help="Run rembg first."),
    smoothing: int = typer.Option(
        2, "--smoothing", min=0, max=3, help="0=off 1=low 2=medium (default) 3=high. Photos need at least 2."
    ),
    fit_aspect: bool = typer.Option(
        True,
        "--fit-aspect/--stretch",
        help="Default fits image aspect inside the width/height bounding box; --stretch distorts to the literal mm.",
    ),
    material: str | None = typer.Option(None, "--material", help="Preset name; supplies kerf if --kerf-mm is unset."),
    width_mm: float = typer.Option(200.0, "--width-mm"),
    height_mm: float = typer.Option(200.0, "--height-mm"),
    kerf_mm: float | None = typer.Option(None, "--kerf-mm", help="Override the material preset's kerf."),
    min_feature_mm: float = typer.Option(0.5, "--min-feature-mm"),
    invert_layers: str = typer.Option("", "--invert-layers", help="Comma-separated 1-based layer indices to invert."),
    frame: bool = typer.Option(True, "--frame/--no-frame"),
    engrave_numbers: bool = typer.Option(True, "--engrave-numbers/--no-engrave-numbers"),
    force: bool = typer.Option(False, "--force", help="Ignore preflight errors."),
) -> None:
    """Slice an image into N layered SVGs."""
    parsed_inverts = tuple(int(x) for x in invert_layers.split(",") if x.strip())
    settings = ProjectSettings(
        engine=engine,
        threshold_mode=threshold_mode,
        layers=layers,
        remove_bg=remove_bg,
        smoothing=smoothing,
        fit_aspect=fit_aspect,
        material=material,
        width_mm=width_mm,
        height_mm=height_mm,
        kerf_mm=_resolve_kerf(material, kerf_mm),
        min_feature_mm=min_feature_mm,
        invert_layers=parsed_inverts,
        frame=frame,
        engrave_numbers=engrave_numbers,
    )
    raw = input_path.read_bytes()
    result = pipeline.process(raw, settings)
    _exit_on_errors(result, force)
    _emit(result, output)
    typer.echo(f"Wrote {len(result.layers)} layer(s) to {output}")
    if result.warnings:
        typer.echo(f"{len(result.warnings)} preflight notes — see {output / 'preflight.txt'}")


@project_app.command("save")
def project_save(
    to: Path = typer.Option(..., "--to", help="Path to write the project JSON to."),
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False),
    layers: int = typer.Option(5, "--layers"),
    engine: str = typer.Option("luminance", "--engine"),
    threshold_mode: str = typer.Option("otsu", "--threshold-mode"),
    remove_bg: bool = typer.Option(False, "--remove-bg/--no-remove-bg"),
    smoothing: int = typer.Option(2, "--smoothing", min=0, max=3),
    fit_aspect: bool = typer.Option(True, "--fit-aspect/--stretch"),
    material: str | None = typer.Option(None, "--material"),
    width_mm: float = typer.Option(200.0, "--width-mm"),
    height_mm: float = typer.Option(200.0, "--height-mm"),
    kerf_mm: float | None = typer.Option(None, "--kerf-mm"),
    min_feature_mm: float = typer.Option(0.5, "--min-feature-mm"),
    invert_layers: str = typer.Option("", "--invert-layers"),
    frame: bool = typer.Option(True, "--frame/--no-frame"),
    engrave_numbers: bool = typer.Option(True, "--engrave-numbers/--no-engrave-numbers"),
) -> None:
    """Persist a slicing configuration for re-use."""
    parsed_inverts = tuple(int(x) for x in invert_layers.split(",") if x.strip())
    settings = ProjectSettings(
        engine=engine,
        threshold_mode=threshold_mode,
        layers=layers,
        remove_bg=remove_bg,
        smoothing=smoothing,
        fit_aspect=fit_aspect,
        material=material,
        width_mm=width_mm,
        height_mm=height_mm,
        kerf_mm=_resolve_kerf(material, kerf_mm),
        min_feature_mm=min_feature_mm,
        invert_layers=parsed_inverts,
        frame=frame,
        engrave_numbers=engrave_numbers,
    )
    save(to, settings, hash_image_bytes(image.read_bytes()))
    typer.echo(f"Wrote {to}")


@project_app.command("load")
def project_load(
    project_file: Path = typer.Argument(..., exists=True, dir_okay=False),
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False),
    output: Path = typer.Option(Path("./out"), "--output", "-o"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Slice an image using settings from a saved project file."""
    settings, expected_hash = load(project_file)
    raw = image.read_bytes()
    actual_hash = hash_image_bytes(raw)
    if actual_hash != expected_hash:
        typer.echo(
            f"warning: image hash {actual_hash[:12]} != project hash {expected_hash[:12]} — proceeding anyway",
            err=True,
        )
    result = pipeline.process(raw, settings)
    _exit_on_errors(result, force)
    _emit(result, output)
    typer.echo(f"Wrote {len(result.layers)} layer(s) to {output}")


@materials_app.command("list")
def materials_list() -> None:
    """Print built-in material presets."""
    for mat in materials.list_materials():
        line = f"{mat.key:24}  kerf={mat.kerf_mm:.2f}mm  thickness={mat.thickness_mm:.2f}mm  {mat.notes}"
        typer.echo(line.rstrip())


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Run the FastAPI web UI."""
    try:
        import uvicorn  # local import — keeps `shadowbox --help` cheap when the web extra isn't installed
    except ImportError as exc:  # pragma: no cover
        typer.echo("The 'web' extra is not installed. Re-install with `pip install .[web]`.", err=True)
        raise typer.Exit(code=2) from exc
    uvicorn.run("shadowbox.web.app:app", host=host, port=port, log_level="info")


def main() -> None:  # pragma: no cover -- entrypoint shim
    app(prog_name="shadowbox")


if __name__ == "__main__":  # pragma: no cover
    main()
    sys.exit(0)
