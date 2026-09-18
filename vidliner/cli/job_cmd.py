"""``vidliner job``, ``jobs``, ``qa``, and ``report``: inspection and re-evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from vidliner.cli.common import Output, fail, resolve_workspace
from vidliner.core.errors import ValidationFailure
from vidliner.domain.enums import JobState
from vidliner.pipeline.service import Session, load_recipe

__all__ = ["app", "jobs_command", "qa_command", "report_app"]

app = typer.Typer(no_args_is_help=True)
report_app = typer.Typer(no_args_is_help=True)


def jobs_command(
    workspace: Path | None = typer.Option(None, "--workspace"),
    runtime: Path | None = typer.Option(None, "--runtime"),
    limit: int = typer.Option(25, "--limit"),
    state: str | None = typer.Option(None, "--state", help="Filter by job state."),
    quiet: bool = typer.Option(False, "--quiet"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """List jobs, newest first."""
    output = Output(json_mode=json_mode, quiet=quiet)
    try:
        session = Session.open(resolve_workspace(workspace), runtime_path=runtime)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    states: tuple[JobState, ...] = ()
    if state:
        try:
            states = (JobState(state),)
        except ValueError as exc:
            session.close()
            raise typer.Exit(
                code=fail(
                    ValidationFailure(
                        f"unknown job state {state!r}; expected one of "
                        f"{', '.join(item.value for item in JobState)}"
                    ),
                    output=output,
                )
            ) from exc
    manifests = session.jobs(limit=limit, states=states)
    payload: list[dict[str, Any]] = [
        {
            "job_id": manifest.job_id,
            "state": manifest.state.value,
            "recipe": manifest.recipe_name,
            "created_at": manifest.created_at.isoformat(),
            "accepted": manifest.counters.accepted,
            "rejected": manifest.counters.rejected,
            "review": manifest.counters.review,
            "acceptance_rate": round(manifest.counters.acceptance_rate, 4),
        }
        for manifest in manifests
    ]
    session.close()
    if json_mode:
        output.payload(payload)
        return
    output.table(
        ["job", "state", "recipe", "accepted", "rejected", "review", "rate", "created"],
        [
            [
                row["job_id"],
                row["state"],
                row["recipe"],
                str(row["accepted"]),
                str(row["rejected"]),
                str(row["review"]),
                f"{row['acceptance_rate']:.0%}",
                row["created_at"][:19],
            ]
            for row in payload
        ],
        fallback="no jobs yet",
    )


@app.command("show")
def show_command(
    job_id: str = typer.Argument(..., help="Job id, or 'latest'."),
    workspace: Path | None = typer.Option(None, "--workspace"),
    runtime: Path | None = typer.Option(None, "--runtime"),
    evidence: bool = typer.Option(False, "--evidence", help="Include per-node evidence rows."),
    quiet: bool = typer.Option(False, "--quiet"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Show one job's manifest, counters, and evidence."""
    output = Output(json_mode=json_mode, quiet=quiet)
    try:
        session = Session.open(resolve_workspace(workspace), runtime_path=runtime)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    try:
        manifest = _resolve_job(session, job_id)
    except ValidationFailure as exc:
        session.close()
        raise typer.Exit(code=fail(exc, output=output)) from exc
    payload: dict[str, Any] = {
        "manifest": manifest.model_dump(mode="json"),
        "stages": session.state.stage_totals(manifest.job_id),
        "candidates": session.state.candidate_counts(manifest.job_id),
        "metric_averages": session.state.metric_averages(manifest.job_id),
        "reports": session.state.reports_for(manifest.job_id),
    }
    if evidence:
        payload["evidence"] = session.state.node_evidence(manifest.job_id)
    session.close()
    output.payload(payload)
    if json_mode:
        return
    counters = manifest.counters
    output.line(f"job         {manifest.job_id}  [{manifest.state.value}]")
    output.line(f"recipe      {manifest.recipe_name}  hash {manifest.recipe_hash[:12]}")
    output.line(f"profile     {manifest.runtime_profile_name or 'default'}  seed {manifest.seed}")
    output.line(f"created     {manifest.created_at.isoformat()}   duration {manifest.duration_s or 0:.1f}s")
    output.line(f"run dir     {manifest.run_dir}")
    output.line(
        f"counters    samples {counters.samples}  targets {counters.targets}  generated {counters.generated}  "
        f"accepted {counters.accepted}  rejected {counters.rejected}  review {counters.review}"
    )
    output.line(
        f"nodes       succeeded {counters.nodes_succeeded}  failed {counters.nodes_failed}  "
        f"cached {counters.nodes_cached}  resumed {counters.nodes_resumed}"
    )
    stages = payload["stages"]
    for stage, statuses in stages.items():  # type: ignore[union-attr]
        rendered = ", ".join(f"{name}={count}" for name, count in sorted(statuses.items()))
        output.line(f"  {stage:10s} {rendered}")
    if manifest.summary and manifest.summary.reason_code_histogram:
        output.line(
            "reasons     "
            + ", ".join(
                f"{code}={count}" for code, count in list(manifest.summary.reason_code_histogram.items())[:8]
            )
        )
    if manifest.failure_message:
        output.line(f"failure     [{manifest.failure_code}] {manifest.failure_message}")


@app.command("resume")
def resume_command(
    job_id: str = typer.Argument(..., help="Job id, or 'latest'."),
    recipe: Path | None = typer.Option(None, "--recipe", help="Override the stored recipe snapshot."),
    workspace: Path | None = typer.Option(None, "--workspace"),
    runtime: Path | None = typer.Option(None, "--runtime"),
    no_export: bool = typer.Option(False, "--no-export"),
    quiet: bool = typer.Option(False, "--quiet"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Resume an interrupted job from its run directory and state rows."""
    import asyncio

    output = Output(json_mode=json_mode, quiet=quiet)
    try:
        session = Session.open(resolve_workspace(workspace), runtime_path=runtime)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    try:
        manifest = _resolve_job(session, job_id)
        loaded = load_recipe(recipe) if recipe is not None else None
        outcome = asyncio.run(session.resume(manifest.job_id, recipe=loaded, export=not no_export))
    except ValidationFailure as exc:
        session.close()
        raise typer.Exit(code=fail(exc, output=output)) from exc
    except Exception as exc:
        session.close()
        raise typer.Exit(code=fail(exc, output=output, code=1)) from exc
    payload = {
        "job_id": outcome.manifest.job_id,
        "state": outcome.manifest.state.value,
        "counters": outcome.manifest.counters.model_dump(mode="json"),
    }
    session.close()
    output.payload(payload)
    if not json_mode:
        output.line(f"job {payload['job_id']} resumed -> {payload['state']}")
    if outcome.manifest.state is not JobState.SUCCEEDED:
        raise typer.Exit(code=1)


@app.command("cancel")
def cancel_command(
    job_id: str = typer.Argument(..., help="Job id, or 'latest'."),
    workspace: Path | None = typer.Option(None, "--workspace"),
    runtime: Path | None = typer.Option(None, "--runtime"),
    quiet: bool = typer.Option(False, "--quiet"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Mark a job as cancelled.

    A job runs in the process that started it, so cancellation is recorded durably here and honoured
    by the running process through its own control surface; a job that is not running simply ends up
    cancelled, which is the state the operator asked for.
    """
    output = Output(json_mode=json_mode, quiet=quiet)
    try:
        session = Session.open(resolve_workspace(workspace), runtime_path=runtime)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    try:
        manifest = _resolve_job(session, job_id)
        session.state.transition_job(manifest.job_id, JobState.CANCELLED)
    except ValidationFailure as exc:
        session.close()
        raise typer.Exit(code=fail(exc, output=output)) from exc
    session.close()
    output.payload({"job_id": manifest.job_id, "state": JobState.CANCELLED.value})
    if not json_mode:
        output.line(f"job {manifest.job_id} marked cancelled")


def qa_command(
    job_id: str = typer.Argument(..., help="Job id, or 'latest'."),
    recipe: Path | None = typer.Option(
        None, "--recipe", help="Re-evaluate under a different recipe's policy."
    ),
    workspace: Path | None = typer.Option(None, "--workspace"),
    runtime: Path | None = typer.Option(None, "--runtime"),
    show: int = typer.Option(0, "--show", help="List this many candidates with their decisions."),
    quiet: bool = typer.Option(False, "--quiet"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Re-apply the acceptance policy to a job's stored metric evidence."""
    output = Output(json_mode=json_mode, quiet=quiet)
    try:
        session = Session.open(resolve_workspace(workspace), runtime_path=runtime)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    try:
        manifest = _resolve_job(session, job_id)
        loaded = load_recipe(recipe) if recipe is not None else None
        result = session.re_evaluate(manifest.job_id, loaded)
        candidates = session.state.candidates(manifest.job_id)[: show or 0]
    except ValidationFailure as exc:
        session.close()
        raise typer.Exit(code=fail(exc, output=output)) from exc
    payload: dict[str, Any] = {
        **result,
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "category": candidate.category,
                "state": candidate.state.value,
                "overall_score": candidate.overall_score,
                "reason_codes": list(candidate.reason_codes),
            }
            for candidate in candidates
        ],
        "counts": session.state.candidate_counts(manifest.job_id),
    }
    session.close()
    output.payload(payload)
    if json_mode:
        return
    output.line(f"job        {result['job_id']}")
    output.line(f"policy     {result['policy_hash'][:12]}")
    output.line(f"evaluated  {result['evaluated']} candidate(s); {result['changed']} changed decision")
    counts = payload["counts"]
    output.line("counts     " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
    for row in payload["candidates"]:
        raw_codes = row.get("reason_codes")
        codes = [str(code) for code in raw_codes] if isinstance(raw_codes, (list, tuple)) else []
        reasons = ", ".join(codes) or "-"
        score = f"{row['overall_score']:.3f}" if row["overall_score"] is not None else "n/a"
        category = str(row.get("category") or "")
        output.line(f"  {row['candidate_id']}  {row['state']!s:<9s} {score}  {category:<12s} {reasons}")


@report_app.command("review")
def review_command(
    job_id: str = typer.Argument(..., help="Job id, or 'latest'."),
    workspace: Path | None = typer.Option(None, "--workspace"),
    runtime: Path | None = typer.Option(None, "--runtime"),
    quiet: bool = typer.Option(False, "--quiet"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Regenerate the static HTML review report for a job."""
    output = Output(json_mode=json_mode, quiet=quiet)
    try:
        session = Session.open(resolve_workspace(workspace), runtime_path=runtime)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    try:
        manifest = _resolve_job(session, job_id)
        from vidliner.control.review import write_review_report

        path = write_review_report(
            workspace=session.workspace,
            state=session.state,
            job_id=manifest.job_id,
            manifest=manifest,
            counters=manifest.counters,
        )
    except ValidationFailure as exc:
        session.close()
        raise typer.Exit(code=fail(exc, output=output)) from exc
    session.close()
    output.payload({"job_id": manifest.job_id, "review": str(path) if path else None})
    if not json_mode:
        output.line(f"review report: {path}")


@report_app.command("summary")
def summary_command(
    job_id: str = typer.Argument(..., help="Job id, or 'latest'."),
    workspace: Path | None = typer.Option(None, "--workspace"),
    runtime: Path | None = typer.Option(None, "--runtime"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Print a job's summary document."""
    try:
        session = Session.open(resolve_workspace(workspace), runtime_path=runtime)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=Output(json_mode=json_mode))) from exc
    try:
        manifest = _resolve_job(session, job_id)
    except ValidationFailure as exc:
        session.close()
        raise typer.Exit(code=fail(exc, output=Output(json_mode=json_mode))) from exc
    path = session.workspace.run_dir(manifest.job_id) / "job-summary.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    plan_document = session.plan_document(manifest.job_id)
    session.close()
    if json_mode:
        typer.echo(json.dumps({"summary": payload, "plan": plan_document}, indent=2, ensure_ascii=False))
        return
    if not payload:
        typer.echo("no summary was written for this job")
        return
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@report_app.command("events")
def events_command(
    job_id: str = typer.Argument(..., help="Job id, or 'latest'."),
    workspace: Path | None = typer.Option(None, "--workspace"),
    runtime: Path | None = typer.Option(None, "--runtime"),
    limit: int = typer.Option(0, "--limit", help="Show the last N events (0 = all)."),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Print a job's structured event log."""
    from vidliner.runtime.logs import read_events

    try:
        session = Session.open(resolve_workspace(workspace), runtime_path=runtime)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=Output(json_mode=json_mode))) from exc
    try:
        manifest = _resolve_job(session, job_id)
    except ValidationFailure as exc:
        session.close()
        raise typer.Exit(code=fail(exc, output=Output(json_mode=json_mode))) from exc
    path = session.workspace.run_dir(manifest.job_id) / "events.jsonl"
    events = read_events(path)
    session.close()
    if limit:
        events = events[-limit:]
    if json_mode:
        typer.echo(json.dumps(events, indent=2, ensure_ascii=False))
        return
    for event in events:
        summary = " ".join(
            f"{key}={value}"
            for key, value in event.items()
            if key != "ts" and isinstance(value, (str, int, float, bool))
        )
        typer.echo(f"{event.get('ts', '')}  {event.get('event', ''):24s} {summary}")


def _resolve_job(session: Session, job_id: str):
    if job_id == "latest":
        latest = session.state.latest_job()
        if latest is None:
            raise ValidationFailure("no job has been run in this workspace yet")
        return latest
    return session.state.require_job(job_id)
