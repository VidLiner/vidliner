"""Exercise the portable canvas workflow through the public CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tests.conftest import make_recipe
from vidliner.cli.main import app
from vidliner.domain.workflow import WorkflowDocument
from vidliner.fixtures import build_dataset

runner = CliRunner()


def test_export_validate_and_preview_without_generation(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vidliner.pipeline.runner import JobRunner

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("workflow export must not execute the job")

    monkeypatch.setattr(JobRunner, "execute", forbidden)
    build_dataset(workspace / "data/source", count=2, annotations=True)
    recipe = workspace / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(make_recipe("./data/source", candidates=1)), encoding="utf-8")
    target = workspace / "canvas/plan.json"
    arguments = [
        "workflow",
        "export",
        str(recipe),
        "--workspace",
        str(workspace),
        "-o",
        str(target),
        "--json",
    ]
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    document = WorkflowDocument.model_validate_json(target.read_text())
    assert document.capability_bindings
    assert len(document.nodes) == json.loads(result.output)["nodes"]
    assert not list((workspace / "runs").glob("*/manifest.json"))
    assert runner.invoke(app, ["workflow", "validate", str(target)]).exit_code == 0
    preview = workspace / "canvas/preview.html"
    result = runner.invoke(app, ["workflow", "preview", str(target), "-o", str(preview)])
    assert result.exit_code == 0, result.output
    assert "VIDLINER" in preview.read_text()
    assert runner.invoke(app, arguments).exit_code == 2
    assert runner.invoke(app, [*arguments, "--force"]).exit_code == 0


@pytest.mark.parametrize("command", ["validate", "preview"])
def test_invalid_workflow_returns_cli_error(tmp_path: Path, command: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version":"unknown"}', encoding="utf-8")
    arguments = ["workflow", command, str(path)]
    if command == "preview":
        arguments += ["-o", str(tmp_path / "preview.html")]
    result = runner.invoke(app, arguments)
    assert result.exit_code == 2
    assert "GRAPH_INVALID" in result.output


def test_schema_and_catalog_are_machine_readable() -> None:
    assert json.loads(runner.invoke(app, ["workflow", "schema"]).output)["type"] == "object"
    assert json.loads(runner.invoke(app, ["workflow", "catalog"]).output)
