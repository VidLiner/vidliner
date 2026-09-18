"""``vidliner plan``: compile a recipe and estimate the work without running it.

Planning resolves capability *bindings* and counts nodes. It never calls a generative backend, which
is the property that makes ``--dry-run`` safe to run before spending anything.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vidliner.cli.common import Output, fail, resolve_workspace
from vidliner.core.errors import ValidationFailure
from vidliner.pipeline.service import Session, load_recipe

__all__ = ["plan_command"]


def plan_command(
    recipe: Path = typer.Argument(..., help="Recipe file to plan."),
    workspace: Path | None = typer.Option(None, "--workspace", help="Workspace directory."),
    runtime: Path | None = typer.Option(None, "--runtime", help="Runtime profile to use."),
    limit: int = typer.Option(0, "--limit", help="Plan for at most this many samples."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report estimates only (this is the default)."),
    write: bool = typer.Option(True, "--write/--no-write", help="Write plan.json into the run directory."),
    quiet: bool = typer.Option(False, "--quiet"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Compile the graph, resolve backends, and print the plan and its estimate."""
    del dry_run  # planning is always a dry run
    output = Output(json_mode=json_mode, quiet=quiet)
    try:
        loaded = load_recipe(recipe)
        session = Session.open(resolve_workspace(workspace), runtime_path=runtime)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    try:
        plan, discovery = session.plan(loaded, limit=limit, instantiate=False)
    except ValidationFailure as exc:
        session.close()
        raise typer.Exit(code=fail(exc, output=output)) from exc

    estimate = plan.job_plan.estimate
    payload = {
        "job_id": plan.job_plan.job_id,
        "recipe_hash": plan.job_plan.recipe_hash,
        "graph_hash": plan.job_plan.graph_hash,
        "runnable": plan.is_runnable,
        "nodes": plan.job_plan.node_count,
        "capability_bindings": plan.job_plan.capability_bindings,
        "unmet_capabilities": list(plan.resolution.unmet),
        "estimate": estimate.model_dump(mode="json"),
        "stages": [{"stage": stage.value, "nodes": count} for stage, count in plan.graph.describe_stages()],
        "dataset": {
            "root": str(discovery.root),
            "samples": discovery.count,
            "splits": discovery.split_counts,
            "augment_splits": list(discovery.augment_splits),
        },
    }
    if write:
        from vidliner.domain.enums import JobState
        from vidliner.domain.jobs import JobManifest
        from vidliner.pipeline.service import RunOutcome

        placeholder = JobManifest(
            job_id=plan.job_plan.job_id,
            state=JobState.PLANNED,
            recipe_name=loaded.name,
            recipe_hash=loaded.recipe_hash(),
            recipe_snapshot=loaded.snapshot(),
            runtime_snapshot={},
            seed=loaded.replacement.seed or 0,
            workspace_root=str(session.workspace.root),
            run_dir=str(session.workspace.run_dir(plan.job_plan.job_id)),
            dataset_input=loaded.dataset.input,
            output_path=loaded.export.path,
            node_count=plan.job_plan.node_count,
        )
        session.write_plan_document(RunOutcome(manifest=placeholder, plan=plan, discovery=discovery))
    session.close()

    output.payload(payload)
    if json_mode:
        return
    output.line(f"job        {payload['job_id']}")
    output.line(f"graph      {payload['nodes']} nodes, hash {payload['graph_hash'][:12]}")
    output.line(
        "stages     "
        + ", ".join(
            f"{stage}={count}" for stage, count in [(s["stage"], s["nodes"]) for s in payload["stages"]]
        )
    )
    output.line(f"samples    {estimate.samples}  candidates {estimate.candidates}")
    output.line(
        f"external   {estimate.external_calls} call(s)  accelerator ops {estimate.accelerator_operations}"
    )
    cost = (
        f"{estimate.estimated_cost:.4f} {estimate.currency or ''}".strip()
        if estimate.estimated_cost
        else "n/a"
    )
    output.line(f"estimate   {estimate.estimated_seconds:.2f}s, cost {cost}")
    for capability, backend in payload["capability_bindings"].items():
        output.line(f"  {capability:44s} -> {backend}")
    if payload["unmet_capabilities"]:
        output.line("UNBOUND capabilities:")
        for line in plan.unmet_explanation():
            output.line(f"  {line}")
    for note in estimate.notes:
        output.line(f"note       {note}")
    if payload["unmet_capabilities"]:
        raise typer.Exit(code=1)
