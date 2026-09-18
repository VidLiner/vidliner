"""``vidliner export``: write a job's accepted samples as a dataset."""

from __future__ import annotations

from pathlib import Path

import typer

from vidliner.cli.common import Output, fail, resolve_workspace
from vidliner.core.errors import ValidationFailure
from vidliner.domain.annotations import ANNOTATION_FORMATS
from vidliner.pipeline.service import Session

__all__ = ["export_command"]


def export_command(
    job_id: str = typer.Argument(..., help="Job id, or 'latest'."),
    workspace: Path | None = typer.Option(None, "--workspace"),
    runtime: Path | None = typer.Option(None, "--runtime"),
    fmt: str | None = typer.Option(
        None, "--format", help=f"Annotation format: {', '.join(ANNOTATION_FORMATS)}."
    ),
    path: Path | None = typer.Option(None, "--path", help="Output directory."),
    include_review: bool = typer.Option(
        False, "--include-review", help="Also export NEEDS_REVIEW candidates."
    ),
    quiet: bool = typer.Option(False, "--quiet"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Write images, annotations, provenance, and reports for a job's accepted samples."""
    output = Output(json_mode=json_mode, quiet=quiet)
    session = Session.open(resolve_workspace(workspace), runtime_path=runtime)
    try:
        manifest = session.state.latest_job() if job_id == "latest" else session.state.require_job(job_id)
        if manifest is None:
            raise ValidationFailure("no job has been run in this workspace yet")
        outcome = session.export_job(
            manifest.job_id,
            include_review=include_review,
            format_override=fmt,
            path_override=str(path) if path is not None else None,
        )
    except ValidationFailure as exc:
        session.close()
        raise typer.Exit(code=fail(exc, output=output)) from exc
    session.close()
    payload = {
        "job_id": manifest.job_id,
        "path": str(outcome.path),
        "format": outcome.format,
        "accepted": outcome.accepted,
        "rejected": outcome.rejected,
        "review": outcome.review,
        "duplicates": outcome.duplicates,
        "images": outcome.images_written,
        "categories": [
            {"id": category.category_id, "name": category.name} for category in outcome.categories
        ],
        "manifest": str(outcome.manifest_path) if outcome.manifest_path else None,
        "report": str(outcome.report_path) if outcome.report_path else None,
        "leakage_checked": outcome.leakage_checked,
        "notes": list(outcome.notes),
    }
    output.payload(payload)
    if json_mode:
        return
    output.line(f"dataset    {outcome.path} ({outcome.format})")
    output.line(f"accepted   {outcome.accepted} sample(s), {outcome.images_written} image(s)")
    output.line(
        f"excluded   {outcome.rejected} rejected, {outcome.review} review, {outcome.duplicates} duplicate(s)"
    )
    output.line(f"categories {', '.join(category.name for category in outcome.categories) or 'none'}")
    if outcome.manifest_path:
        output.line(f"manifest   {outcome.manifest_path}")
    if outcome.report_path:
        output.line(f"report     {outcome.report_path}")
    for note in outcome.notes:
        output.line(f"note       {note}")
    if not outcome.accepted:
        output.line(
            "warning    nothing was accepted; inspect 'vidliner qa <job-id> --show 10' before retuning"
        )
        raise typer.Exit(code=1)
