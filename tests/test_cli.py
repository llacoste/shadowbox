from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from shadowbox.cli import app


def test_materials_list_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["materials", "list"])
    assert result.exit_code == 0
    assert "birch_ply_3mm" in result.stdout


def test_slice_emits_svgs(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "mountains.png"
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "slice",
            str(fixture),
            "--layers",
            "3",
            "--threshold-mode",
            "equal",
            "--kerf-mm",
            "0",
            "--output",
            str(out),
            "--no-engrave-numbers",  # avoids font shape variance in goldens
        ],
    )
    assert result.exit_code == 0, result.stdout
    svgs = sorted(out.glob("layer_*.svg"))
    assert len(svgs) == 3
    assert (out / "preflight.txt").exists()


def test_project_round_trip(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "mountains.png"
    project = tmp_path / "p.json"
    runner = CliRunner()
    save = runner.invoke(
        app,
        [
            "project",
            "save",
            "--to",
            str(project),
            "--image",
            str(fixture),
            "--layers",
            "2",
            "--threshold-mode",
            "equal",
            "--kerf-mm",
            "0",
        ],
    )
    assert save.exit_code == 0, save.stdout
    assert project.exists()

    out = tmp_path / "out"
    load = runner.invoke(
        app,
        ["project", "load", str(project), "--image", str(fixture), "--output", str(out)],
    )
    assert load.exit_code == 0, load.stdout
    assert len(list(out.glob("layer_*.svg"))) == 2
