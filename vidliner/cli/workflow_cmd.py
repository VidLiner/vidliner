"""Workflow exchange, palette discovery and offline canvas preview."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from vidliner.cli.common import Output, fail, resolve_workspace
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.workflow import WorkflowDocument
from vidliner.domain.workflow_edit import WorkflowEditRequest
from vidliner.pipeline.service import Session, load_recipe
from vidliner.pipeline.workflow import (
    graph_from_workflow,
    load_workflow,
    operator_catalogue,
    workflow_from_graph,
)
from vidliner.pipeline.workflow_edit import apply_workflow_edits, workflow_digest
from vidliner.pipeline.workflow_execution import WorkflowExecutionRequest
from vidliner.reports.workflow import render_workflow

app = typer.Typer(no_args_is_help=True)


@app.command("digest")
def digest_command(document_path: Path) -> None:
    """Return the reviewed draft identity used by edit and execution requests."""
    try:
        typer.echo(workflow_digest(load_workflow(document_path)))
    except ValidationFailure as exc:
        raise typer.Exit(fail(exc)) from exc


@app.command("patch")
def patch_command(
    document_path: Path,
    patch_path: Path,
    output_path: Path = typer.Option(..., "--output", "-o"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Apply an atomic typed graph edit batch and validate the complete postimage."""
    try:
        original = load_workflow(document_path)
        patch = WorkflowEditRequest.model_validate_json(patch_path.read_text(encoding="utf-8"))
        result = apply_workflow_edits(original, patch)
        _write(output_path, result.model_dump_json(indent=2) + "\n", force=force)
    except (OSError, ValueError, ValidationFailure) as exc:
        error = (
            exc
            if isinstance(exc, ValidationFailure)
            else ValidationFailure(str(exc), code=ErrorCode.GRAPH_INVALID)
        )
        raise typer.Exit(fail(error)) from exc
    typer.echo(f"workflow draft written to {output_path}")


@app.command("edit-schema")
def edit_schema_command() -> None:
    """JSON Schema for frontend structural editing transactions."""
    typer.echo(json.dumps(WorkflowEditRequest.model_json_schema(), indent=2))


@app.command("execution-schema")
def execution_schema_command() -> None:
    """JSON Schema for an authenticated execution host's reviewed request."""
    typer.echo(json.dumps(WorkflowExecutionRequest.model_json_schema(), indent=2))


def _write(path: Path, content: str, *, force: bool) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w" if force else "x", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        raise ValidationFailure(f"cannot write {path}: {exc}", code=ErrorCode.GRAPH_INVALID) from exc


@app.command("export")
def export_command(
    recipe: Path = typer.Argument(..., help="Recipe to compile without generation."),
    output_path: Path = typer.Option(..., "--output", "-o", help="Canvas JSON destination."),
    workspace: Path | None = typer.Option(None, "--workspace"),
    runtime: Path | None = typer.Option(None, "--runtime"),
    limit: int = typer.Option(1, "--limit", min=0, help="Samples to plan; 0 means all."),
    force: bool = typer.Option(False, "--force", help="Replace an existing document."),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Compile the recipe into a portable node/edge document; no backend calls."""
    output = Output(json_mode=json_mode)
    session: Session | None = None
    try:
        loaded = load_recipe(recipe)
        session = Session.open(resolve_workspace(workspace), runtime_path=runtime)
        plan, _ = session.plan(loaded, limit=limit, instantiate=False)
        document = workflow_from_graph(
            plan.graph,
            name=loaded.name,
            recipe_hash=loaded.recipe_hash(),
            capability_bindings=plan.resolution.as_mapping(),
            unmet_capabilities=plan.resolution.unmet,
        )
        graph_from_workflow(document)
        _write(output_path, document.model_dump_json(indent=2) + "\n", force=force)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    except ValueError as exc:
        error = ValidationFailure(f"cannot export workflow: {exc}", code=ErrorCode.GRAPH_INVALID)
        raise typer.Exit(code=fail(error, output=output)) from exc
    finally:
        if session is not None:
            session.close()
    output.payload(
        {
            "path": str(output_path),
            "nodes": len(document.nodes),
            "edges": len(document.edges),
            "unmet_capabilities": list(document.unmet_capabilities),
        },
        fallback=f"workflow written to {output_path} ({len(document.nodes)} nodes)",
    )


@app.command("validate")
def validate_command(
    path: Path = typer.Argument(...),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Check schema, connections, cycles, ports, configs and registered contracts."""
    output = Output(json_mode=json_mode)
    try:
        document = load_workflow(path)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    output.payload(
        {"valid": True, "name": document.name, "nodes": len(document.nodes), "edges": len(document.edges)},
        fallback=f"workflow {document.name!r} is valid",
    )


@app.command("schema")
def schema_command() -> None:
    """Print the versioned canvas JSON Schema."""
    Output(json_mode=True).payload(WorkflowDocument.model_json_schema())


@app.command("catalog")
def catalog_command() -> None:
    """Print the operator palette with typed ports and config schemas as JSON."""
    Output(json_mode=True).payload(operator_catalogue())


@app.command("preview")
def preview_command(
    path: Path = typer.Argument(...),
    output_path: Path = typer.Option(..., "--output", "-o", help="Standalone HTML destination."),
    force: bool = typer.Option(False, "--force"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Write a searchable canvas with pan, zoom and node inspection, usable offline."""
    output = Output(json_mode=json_mode)
    try:
        document = load_workflow(path)
        _write(output_path, render_workflow(document), force=force)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    output.payload({"path": str(output_path)}, fallback=f"workflow preview written to {output_path}")
