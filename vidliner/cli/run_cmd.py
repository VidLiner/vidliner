"""``vidliner run``: execute a job."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from vidliner.cli.common import Output, fail, resolve_workspace
from vidliner.core.errors import ValidationFailure
from vidliner.domain.enums import JobState
from vidliner.pipeline.runner import JobOptions
from vidliner.pipeline.service import JobRequest, Session, load_recipe

__all__ = ["run_command"]


def run_command(
    recipe: Path = typer.Argument(..., help="Recipe file to run."),
    workspace: Path | None = typer.Option(None, "--workspace", help="Workspace directory."),
    runtime: Path | None = typer.Option(None, "--runtime", help="Runtime profile to use."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compile and estimate only; call no generative backend."
    ),
    limit: int = typer.Option(0, "--limit", help="Process at most this many samples."),
    seed: int | None = typer.Option(None, "--seed", help="Override the recipe's seed."),
    job_id: str | None = typer.Option(None, "--job-id", help="Run under an explicit job id."),
    new_job: bool = typer.Option(
        False, "--new", help="Force a fresh job id instead of reusing the derived one."
    ),
    workers: int = typer.Option(4, "--workers", help="Maximum concurrent nodes."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore cached node results."),
    no_resume: bool = typer.Option(False, "--no-resume", help="Re-execute nodes instead of resuming them."),
    no_export: bool = typer.Option(False, "--no-export", help="Skip the dataset export."),
    include_review: bool = typer.Option(
        False, "--include-review", help="Export NEEDS_REVIEW candidates too."
    ),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Stop at the first node failure."),
    timeout: float = typer.Option(0.0, "--timeout", help="Whole-job timeout in seconds (0 = none)."),
    verbose: bool = typer.Option(False, "--verbose", help="Mirror the event log to stderr."),
    quiet: bool = typer.Option(False, "--quiet"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Run the augmentation pipeline and export the accepted samples."""
    output = Output(json_mode=json_mode, quiet=quiet)
    try:
        loaded = load_recipe(recipe)
        if seed is not None:
            loaded = loaded.model_copy(
                update={"replacement": loaded.replacement.model_copy(update={"seed": seed})}
            )
        session = Session.open(resolve_workspace(workspace), runtime_path=runtime)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc

    request = JobRequest(
        recipe=loaded,
        options=JobOptions(
            dry_run=dry_run,
            use_cache=not no_cache,
            resume=not no_resume,
            fail_fast=fail_fast,
            workers=workers,
            job_timeout_s=timeout or None,
            seed=seed,
        ),
        limit=limit,
        job_id=None if new_job else job_id,
        export=not no_export,
        include_review=include_review,
    )
    try:
        outcome = asyncio.run(session.run(request))
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        output.line("interrupted; the job can be resumed with 'vidliner job resume <job-id>'")
        session.close()
        raise typer.Exit(code=1) from None
    except ValidationFailure as exc:
        session.close()
        raise typer.Exit(code=fail(exc, output=output)) from exc
    except Exception as exc:
        session.close()
        raise typer.Exit(code=fail(exc, output=output, code=1)) from exc

    manifest = outcome.manifest
    payload = {
        "job_id": manifest.job_id,
        "state": manifest.state.value,
        "counters": manifest.counters.model_dump(mode="json"),
        "acceptance_rate": round(manifest.counters.acceptance_rate, 4),
        "run_dir": manifest.run_dir,
        "export": None
        if outcome.export is None
        else {
            "path": str(outcome.export.path),
            "format": outcome.export.format,
            "accepted": outcome.export.accepted,
            "rejected": outcome.export.rejected,
            "review": outcome.export.review,
            "duplicates": outcome.export.duplicates,
            "images": outcome.export.images_written,
        },
        "failure": None
        if manifest.failure_code is None
        else {
            "class": manifest.failure_class,
            "code": manifest.failure_code,
            "message": manifest.failure_message,
        },
    }
    session.close()
    output.payload(payload)
    if json_mode:
        if manifest.state is not JobState.SUCCEEDED:
            raise typer.Exit(code=1)
        return
    counters = manifest.counters
    output.line(f"job        {manifest.job_id}  [{manifest.state.value}]")
    output.line(
        f"candidates {counters.generated}  accepted {counters.accepted}  rejected {counters.rejected}  "
        f"review {counters.review}  acceptance {counters.acceptance_rate:.0%}"
    )
    output.line(
        f"nodes      {counters.nodes_succeeded} ok, {counters.nodes_failed} failed, "
        f"{counters.nodes_cached} cached, {counters.nodes_resumed} resumed"
    )
    if outcome.export is not None:
        output.line(f"dataset    {outcome.export.path} ({outcome.export.format})")
        output.line(
            f"  written  {outcome.export.images_written} image(s), {outcome.export.accepted} accepted sample(s), "
            f"{outcome.export.duplicates} duplicate(s) excluded"
        )
        for note in outcome.export.notes:
            output.line(f"  note     {note}")
    output.line(f"evidence   {manifest.run_dir}")
    if manifest.failure_message:
        output.line(f"failure    [{manifest.failure_code}] {manifest.failure_message}")
    _ = verbose
    if manifest.state is not JobState.SUCCEEDED:
        raise typer.Exit(code=1)
