"""Video probing and local variant generation commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from vidliner.cli.common import Output, fail
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.video import VideoAugmentSpec, enumerate_video_variants
from vidliner.domain.video_generation import VideoGenerationRequest, VideoTask
from vidliner.reports.video import render_video_review
from vidliner.video.render import generate_variants, probe_video, render_variant

app = typer.Typer(no_args_is_help=True)


@app.command("request-schema")
def request_schema_command() -> None:
    """Provider-neutral AI generation request schema for integration hosts."""
    typer.echo(json.dumps(VideoGenerationRequest.model_json_schema(), indent=2))


@app.command("task-schema")
def task_schema_command() -> None:
    """Durable asynchronous video task handle schema."""
    typer.echo(json.dumps(VideoTask.model_json_schema(), indent=2))


@app.command("preview")
def preview_command(
    manifest_path: Path = typer.Argument(..., help="Rendered manifest.json."),
    output_path: Path = typer.Option(..., "--output", "-o", help="Standalone HTML review page."),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Compare rendered variants against the source in an offline video review page."""
    try:
        if output_path.exists() and not force:
            raise ValidationFailure(
                f"output {output_path} already exists; pass --force", code=ErrorCode.EXPORT_INVALID
            )
        if output_path.resolve() == manifest_path.resolve():
            raise ValidationFailure("preview must not overwrite its manifest", code=ErrorCode.EXPORT_INVALID)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValidationFailure("manifest must be a JSON object", code=ErrorCode.EXPORT_INVALID)
        page = render_video_review(payload, output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(page, encoding="utf-8")
    except (OSError, ValueError, ValidationFailure) as exc:
        error = (
            exc
            if isinstance(exc, ValidationFailure)
            else ValidationFailure(str(exc), code=ErrorCode.EXPORT_INVALID)
        )
        raise typer.Exit(code=fail(error)) from exc
    typer.echo(f"video review written to {output_path}")


def _load_spec(path: Path) -> VideoAugmentSpec:
    """Load a JSON or YAML video spec."""
    if not path.is_file():
        raise ValidationFailure(f"video spec {path} does not exist", code=ErrorCode.RECIPE_INVALID)
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
        return VideoAugmentSpec.model_validate(payload)
    except Exception as exc:
        raise ValidationFailure(
            f"video spec {path} is invalid: {exc}", code=ErrorCode.RECIPE_INVALID
        ) from exc


@app.command("probe")
def probe_command(
    input_path: Path = typer.Argument(..., metavar="INPUT"), json_mode: bool = typer.Option(False, "--json")
) -> None:
    """Inspect a video without decoding it into Python memory."""
    output = Output(json_mode=json_mode)
    try:
        probe = probe_video(input_path)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    payload = {
        "path": str(probe.path),
        "duration_s": probe.duration_s,
        "width": probe.width,
        "height": probe.height,
        "fps": probe.fps,
        "has_audio": probe.has_audio,
        "codec": probe.codec,
    }
    output.payload(
        payload, fallback=f"{probe.width}x{probe.height} {probe.duration_s:.2f}s @ {probe.fps:.3f} fps"
    )


@app.command("generate")
def generate_command(
    spec_path: Path = typer.Argument(..., help="VideoAugmentSpec JSON/YAML."),
    input_path: Path = typer.Option(..., "--input", "-i", metavar="INPUT"),
    output_dir: Path = typer.Option(..., "--output", "-o", metavar="DIR"),
    force: bool = typer.Option(False, "--force"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Generate every deterministic local video variant and write a provenance manifest."""
    output = Output(json_mode=json_mode)
    try:
        spec = _load_spec(spec_path)
        results = generate_variants(input_path, output_dir, spec, overwrite=force)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    payload = {
        "output_dir": str(output_dir),
        "spec": spec.id,
        "variants": [result.manifest() for result in results],
    }
    output.payload(payload, fallback=f"generated {len(results)} video variant(s) in {output_dir}")


@app.command("render")
def render_command(
    spec_path: Path = typer.Argument(..., help="VideoAugmentSpec JSON/YAML."),
    input_path: Path = typer.Option(..., "--input", "-i", metavar="INPUT"),
    output_path: Path = typer.Option(..., "--output", "-o", metavar="OUTPUT"),
    variant: str = typer.Option("0", "--variant", help="Variant index or content-derived variant id."),
    force: bool = typer.Option(False, "--force"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Render one sampled variant for fast iteration in the visual editor."""
    output = Output(json_mode=json_mode)
    try:
        spec = _load_spec(spec_path)
        variants = enumerate_video_variants(spec)
        selected = (
            variants[int(variant)]
            if variant.isdigit()
            else next(item for item in variants if item.id == variant)
        )
        result = render_variant(input_path, output_path, selected, overwrite=force)
    except (ValidationFailure, IndexError, StopIteration, ValueError) as exc:
        error = (
            exc
            if isinstance(exc, ValidationFailure)
            else ValidationFailure(
                f"cannot select/render video variant: {exc}", code=ErrorCode.VIDEO_NOT_SUPPORTED
            )
        )
        raise typer.Exit(code=fail(error, output=output)) from exc
    output.payload(result.manifest(), fallback=f"rendered {selected.id} to {output_path}")
