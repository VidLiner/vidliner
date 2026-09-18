"""CLI tests: every documented command is exercised through the Typer application."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import make_recipe
from vidliner.cli.main import app

runner = CliRunner()


def _json_only(text: str) -> str:
    """Extract the JSON document from command output that may carry trailing guidance lines."""
    start = text.index("{")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise AssertionError(f"no complete JSON document in output: {text!r}")


@pytest.fixture(name="workspace_dir")
def workspace_dir_fixture(tmp_path: Path) -> Path:
    """A freshly initialised workspace, created through the CLI itself."""
    root = tmp_path / "cli-workspace"
    result = runner.invoke(app, ["init", str(root)])
    assert result.exit_code == 0, result.output
    # Give the workspace a dataset so every later command has something to work with.
    from vidliner.fixtures import build_dataset

    build_dataset(root / "data" / "source", count=4, annotations=True)
    return root


def _recipe_file(root: Path, name: str = "recipe.yaml", **overrides: object) -> Path:
    import yaml

    document = make_recipe("./data/source", candidates=1, **overrides)  # type: ignore[arg-type]
    path = root / name
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------- #


def test_init_creates_a_workspace(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    result = runner.invoke(app, ["init", str(root)])
    assert result.exit_code == 0
    assert (root / "runtime.yaml").is_file()
    assert (root / "state.db").is_file()
    assert (root / "artifacts").is_dir()
    assert "initialised workspace" in result.output


def test_init_reports_json(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    result = runner.invoke(app, ["init", str(root), "--json"])
    assert result.exit_code == 0
    payload = json.loads(_json_only(result.output))
    assert payload["workspace"] == str(root)
    assert payload["backends"]


def test_init_keeps_an_existing_profile(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    runner.invoke(app, ["init", str(root)])
    marker = (root / "runtime.yaml").read_text()
    (root / "runtime.yaml").write_text(marker + "\n# edited\n", encoding="utf-8")
    result = runner.invoke(app, ["init", str(root)])
    assert result.exit_code == 0
    assert "# edited" in (root / "runtime.yaml").read_text()


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #


def test_inspect_reports_the_dataset(workspace_dir: Path) -> None:
    result = runner.invoke(
        app,
        [
            "inspect",
            str(workspace_dir / "data" / "source"),
            "--format",
            "coco-instance",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "samples   4" in result.output
    assert "classes   car=" in result.output


def test_inspect_json_mode(workspace_dir: Path) -> None:
    result = runner.invoke(
        app,
        ["inspect", str(workspace_dir / "data" / "source"), "--format", "coco-instance", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(_json_only(result.output))
    assert payload["samples"] == 4
    assert payload["annotated_samples"] == 4
    assert payload["classes"] == {"car": 4}


def test_inspect_reports_a_missing_directory(workspace_dir: Path) -> None:
    result = runner.invoke(app, ["inspect", str(workspace_dir / "nope")])
    assert result.exit_code == 2
    assert "error" in result.output


# --------------------------------------------------------------------------- #
# recipe
# --------------------------------------------------------------------------- #


def test_recipe_init_and_validate(workspace_dir: Path) -> None:
    target = workspace_dir / "starter.yaml"
    assert runner.invoke(app, ["recipe", "init", str(target)]).exit_code == 0
    assert target.is_file()
    result = runner.invoke(app, ["recipe", "validate", str(target)])
    assert result.exit_code == 0, result.output
    assert "is valid" in result.output


def test_recipe_init_refuses_to_overwrite(workspace_dir: Path) -> None:
    target = workspace_dir / "starter.yaml"
    runner.invoke(app, ["recipe", "init", str(target)])
    result = runner.invoke(app, ["recipe", "init", str(target)])
    assert result.exit_code == 2
    assert runner.invoke(app, ["recipe", "init", str(target), "--force"]).exit_code == 0


def test_recipe_validate_reports_invalid_input(workspace_dir: Path) -> None:
    import yaml

    bad = workspace_dir / "bad.yaml"
    bad.write_text(yaml.safe_dump({"recipe": "bad"}), encoding="utf-8")
    result = runner.invoke(app, ["recipe", "validate", str(bad)])
    assert result.exit_code == 2
    assert "RECIPE_INVALID" in result.output


def test_recipe_schema_is_emitted(workspace_dir: Path) -> None:
    result = runner.invoke(app, ["recipe", "schema"])
    assert result.exit_code == 0
    schema = json.loads(result.output)
    assert schema["title"] == "Recipe"
    assert "dataset" in schema["properties"]


def test_recipe_schema_can_be_written_to_a_file(workspace_dir: Path) -> None:
    target = workspace_dir / "recipe.schema.json"
    result = runner.invoke(app, ["recipe", "schema", "--output", str(target)])
    assert result.exit_code == 0
    assert json.loads(target.read_text())["title"] == "Recipe"


def test_recipe_show_and_json(workspace_dir: Path) -> None:
    path = _recipe_file(workspace_dir)
    result = runner.invoke(app, ["recipe", "show", str(path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["recipe"] == "test-recipe"


def test_recipe_validate_json_mode(workspace_dir: Path) -> None:
    path = _recipe_file(workspace_dir)
    result = runner.invoke(app, ["recipe", "validate", str(path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["required_capabilities"]


# --------------------------------------------------------------------------- #
# plan, run, jobs, job, qa, export
# --------------------------------------------------------------------------- #


def test_plan_before_running(workspace_dir: Path) -> None:
    path = _recipe_file(workspace_dir)
    result = runner.invoke(app, ["plan", str(path), "--workspace", str(workspace_dir)])
    assert result.exit_code == 0, result.output
    assert "graph" in result.output
    assert "samples    4" in result.output
    assert (workspace_dir / "runs").is_dir()


def test_plan_json_mode(workspace_dir: Path) -> None:
    path = _recipe_file(workspace_dir)
    result = runner.invoke(app, ["plan", str(path), "--workspace", str(workspace_dir), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["runnable"] is True
    assert payload["nodes"] > 0
    assert payload["estimate"]["samples"] == 4


def test_run_then_list_then_show_then_export(workspace_dir: Path) -> None:
    path = _recipe_file(workspace_dir)
    run = runner.invoke(app, ["run", str(path), "--workspace", str(workspace_dir)])
    assert run.exit_code == 0, run.output
    assert "accepted 4" in run.output

    listed = runner.invoke(app, ["jobs", "--workspace", str(workspace_dir)])
    assert listed.exit_code == 0
    assert "test-recipe" in listed.output

    shown = runner.invoke(app, ["job", "show", "latest", "--workspace", str(workspace_dir)])
    assert shown.exit_code == 0
    assert "counters" in shown.output

    exported = runner.invoke(
        app,
        ["export", "latest", "--workspace", str(workspace_dir), "--path", "./data/cli-export"],
    )
    assert exported.exit_code == 0, exported.output
    assert (workspace_dir / "data" / "cli-export" / "manifest.json").is_file()


def test_run_dry_run_writes_a_plan_and_generates_nothing(workspace_dir: Path) -> None:
    path = _recipe_file(workspace_dir)
    result = runner.invoke(app, ["run", str(path), "--workspace", str(workspace_dir), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "candidates 0" in result.output
    job_dirs = list((workspace_dir / "runs").iterdir())
    assert job_dirs
    assert (job_dirs[0] / "plan.json").is_file()


def test_run_json_mode(workspace_dir: Path) -> None:
    path = _recipe_file(workspace_dir)
    result = runner.invoke(app, ["run", str(path), "--workspace", str(workspace_dir), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["state"] == "succeeded"
    assert payload["export"]["accepted"] == 4


def test_run_limits_the_sample_count(workspace_dir: Path) -> None:
    path = _recipe_file(workspace_dir)
    result = runner.invoke(app, ["run", str(path), "--workspace", str(workspace_dir), "--limit", "2"])
    assert result.exit_code == 0
    assert "accepted 2" in result.output


def test_run_reports_a_missing_recipe(workspace_dir: Path) -> None:
    result = runner.invoke(
        app, ["run", str(workspace_dir / "missing.yaml"), "--workspace", str(workspace_dir)]
    )
    assert result.exit_code == 2
    assert "RECIPE_INVALID" in result.output


def test_qa_command_reports_and_rescores(workspace_dir: Path) -> None:
    path = _recipe_file(workspace_dir)
    runner.invoke(app, ["run", str(path), "--workspace", str(workspace_dir)])
    result = runner.invoke(app, ["qa", "latest", "--workspace", str(workspace_dir), "--show", "3"])
    assert result.exit_code == 0, result.output
    assert "policy" in result.output
    assert "evaluated" in result.output


def test_job_show_json_includes_stage_totals(workspace_dir: Path) -> None:
    path = _recipe_file(workspace_dir)
    runner.invoke(app, ["run", str(path), "--workspace", str(workspace_dir)])
    result = runner.invoke(app, ["job", "show", "latest", "--workspace", str(workspace_dir), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["stages"]
    assert payload["manifest"]["job_id"]


def test_job_show_evidence(workspace_dir: Path) -> None:
    path = _recipe_file(workspace_dir)
    runner.invoke(app, ["run", str(path), "--workspace", str(workspace_dir)])
    result = runner.invoke(
        app, ["job", "show", "latest", "--workspace", str(workspace_dir), "--evidence", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["evidence"]
    assert all("operator" in row for row in payload["evidence"])


def test_jobs_state_filter_rejects_an_unknown_state(workspace_dir: Path) -> None:
    result = runner.invoke(app, ["jobs", "--workspace", str(workspace_dir), "--state", "nonsense"])
    assert result.exit_code == 2
    assert "unknown job state" in result.output


def test_report_commands_write_and_read_reports(workspace_dir: Path) -> None:
    path = _recipe_file(workspace_dir)
    runner.invoke(app, ["run", str(path), "--workspace", str(workspace_dir)])

    review = runner.invoke(app, ["report", "review", "latest", "--workspace", str(workspace_dir)])
    assert review.exit_code == 0, review.output
    review_dir = workspace_dir / "runs"
    html_files = list(review_dir.rglob("review/index.html"))
    assert html_files
    assert "VidLiner review report" in html_files[0].read_text()

    summary = runner.invoke(app, ["report", "summary", "latest", "--workspace", str(workspace_dir)])
    assert summary.exit_code == 0
    assert "counters" in summary.output

    events = runner.invoke(
        app, ["report", "events", "latest", "--workspace", str(workspace_dir), "--limit", "5"]
    )
    assert events.exit_code == 0
    assert "job." in events.output


def test_job_cancel_marks_the_job(workspace_dir: Path) -> None:
    path = _recipe_file(workspace_dir)
    runner.invoke(app, ["run", str(path), "--workspace", str(workspace_dir)])
    result = runner.invoke(app, ["job", "cancel", "latest", "--workspace", str(workspace_dir)])
    assert result.exit_code == 0, result.output
    assert "cancelled" in result.output


def test_job_resume_after_cancel(workspace_dir: Path) -> None:
    path = _recipe_file(workspace_dir)
    runner.invoke(app, ["run", str(path), "--workspace", str(workspace_dir)])
    runner.invoke(app, ["job", "cancel", "latest", "--workspace", str(workspace_dir)])
    result = runner.invoke(app, ["job", "resume", "latest", "--workspace", str(workspace_dir), "--no-export"])
    assert result.exit_code == 0, result.output
    assert "resumed" in result.output


# --------------------------------------------------------------------------- #
# backend
# --------------------------------------------------------------------------- #


def test_backend_list(workspace_dir: Path) -> None:
    result = runner.invoke(app, ["backend", "list", "--workspace", str(workspace_dir)])
    assert result.exit_code == 0, result.output
    assert "builtin_detector" in result.output
    assert "profile local" in result.output


def test_backend_list_capabilities(workspace_dir: Path) -> None:
    result = runner.invoke(app, ["backend", "list", "--capabilities"])
    assert result.exit_code == 0
    assert "vision.object_detection.v1" in result.output


def test_backend_check_probes_everything(workspace_dir: Path) -> None:
    result = runner.invoke(app, ["backend", "check", "--workspace", str(workspace_dir)])
    assert result.exit_code == 0, result.output
    assert "ok" in result.output
    assert "builtin_detector" in result.output


def test_backend_check_json(workspace_dir: Path) -> None:
    result = runner.invoke(app, ["backend", "check", "--workspace", str(workspace_dir), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["probes"]
    assert "vision.object_detection.v1" in payload["coverage"]


def test_backend_check_without_probing(workspace_dir: Path) -> None:
    result = runner.invoke(app, ["backend", "check", "--workspace", str(workspace_dir), "--no-probe"])
    assert result.exit_code == 0
    assert "probe disabled" in result.output


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


def test_commands_refuse_an_uninitialised_workspace(tmp_path: Path) -> None:
    result = runner.invoke(app, ["jobs", "--workspace", str(tmp_path / "empty")])
    assert result.exit_code == 2
    assert "WORKSPACE_INVALID" in result.output


def test_export_before_any_run(workspace_dir: Path) -> None:
    result = runner.invoke(app, ["export", "latest", "--workspace", str(workspace_dir)])
    assert result.exit_code == 2
    assert "no job" in result.output


def test_help_lists_every_documented_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "init",
        "inspect",
        "recipe",
        "plan",
        "run",
        "jobs",
        "job",
        "qa",
        "export",
        "backend",
        "report",
    ):
        assert command in result.output
