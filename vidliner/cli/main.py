"""The ``vidliner`` command line application."""

from __future__ import annotations

import typer

from vidliner.cli import (
    backend_cmd,
    dataset_cmd,
    export_cmd,
    job_cmd,
    plan_cmd,
    recipe_cmd,
    run_cmd,
    workspace_cmd,
)

__all__ = ["app"]

app = typer.Typer(
    name="vidliner",
    help=(
        "VidLiner - object-centric synthetic data augmentation with mandatory verification.\n\n"
        "Replace a target object in an image, prove the rest of the scene is unchanged, rebuild the "
        "annotation from the generated pixels, and export only what passed the acceptance policy."
    ),
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

app.command("init", help="Create a workspace and a starter runtime profile.")(workspace_cmd.init_command)
app.command("inspect", help="Analyse a dataset directory without running a job.")(dataset_cmd.inspect_command)
app.add_typer(recipe_cmd.app, name="recipe", help="Validate, document, or scaffold a recipe.")
app.command("plan", help="Compile a recipe and estimate the work without running it.")(plan_cmd.plan_command)
app.command("run", help="Execute a job.")(run_cmd.run_command)
app.command("export", help="Export a job's accepted samples as a dataset.")(export_cmd.export_command)
app.command("jobs", help="List jobs.")(job_cmd.jobs_command)
app.command("qa", help="Re-apply the acceptance policy to a finished job.")(job_cmd.qa_command)
app.add_typer(job_cmd.app, name="job", help="Show, resume, or cancel a job.")
app.add_typer(job_cmd.report_app, name="report", help="Regenerate a job's reports.")
app.add_typer(backend_cmd.app, name="backend", help="Inspect and check runtime backends.")


def main() -> None:
    """Console entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
