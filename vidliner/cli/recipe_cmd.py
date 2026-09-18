"""``vidliner recipe``: validate, document, or scaffold a recipe."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from vidliner.cli.common import Output, fail
from vidliner.core.errors import ValidationFailure
from vidliner.domain.recipe import Recipe
from vidliner.pipeline.service import load_recipe

__all__ = ["app", "init_command", "schema_command", "show_command", "validate_command"]

app = typer.Typer(no_args_is_help=True)

_TEMPLATE = """# VidLiner recipe - see docs/recipe.md for every field.
recipe: car-swap
description: replace cars with other vehicle categories

dataset:
  input: ./data/source
  format: coco-instance
  splits:
    mode: directory
    names: [train, val, test]
  augment_splits: [train]

target:
  classes: [car]
  min_score: 0.35
  min_area_px: 1024
  max_per_sample: 2
  top_k_by: score
  strategy: all

replacement:
  mode: strict
  strategy: category
  values: [sedan, suv, pickup]
  candidates_per_object: 3
  seed: 20240517

preserve:
  background: strict
  geometry: true
  lighting: true

quality:
  minimum_overall: 0.82
  hard_gates:
    semantic_match: 0.90
    background_preservation: 0.93
    annotation_consistency: 0.95
  maximum_artifact_score: 0.15
  minimum_target_presence: 0.60
  review_band: 0.03

export:
  format: coco-instance
  path: ./data/output
  copy_images: true

limits:
  workers: 4
  resume: true
  cache: true
"""


@app.command("validate")
def validate_command(
    recipe: Path = typer.Argument(..., help="Recipe file to validate."),
    quiet: bool = typer.Option(False, "--quiet"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Validate a recipe against the schema and report what it would do."""
    output = Output(json_mode=json_mode, quiet=quiet)
    try:
        loaded = load_recipe(recipe)
        capabilities = loaded.required_capabilities()
        plan_splits = _split_plan(loaded)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    payload = {
        "recipe": loaded.name,
        "hash": loaded.recipe_hash(),
        "dataset": loaded.dataset.input,
        "augment_splits": list(loaded.dataset.augment_splits),
        "target_classes": list(loaded.target.classes),
        "replacement_values": list(loaded.replacement.values),
        "candidates_per_object": loaded.replacement.candidates_per_object,
        "mode": loaded.replacement.mode,
        "export_format": loaded.export.format,
        "required_capabilities": list(capabilities),
        "split_notes": list(plan_splits.notes),
    }
    output.payload(payload, fallback=f"recipe {loaded.name} is valid ({loaded.recipe_hash()[:12]})")
    if json_mode:
        return
    output.line(f"  dataset          {loaded.dataset.input} ({loaded.dataset.format or 'no annotations'})")
    output.line(f"  augment splits   {', '.join(loaded.dataset.augment_splits)}")
    output.line(
        f"  targets          {', '.join(loaded.target.classes)} -> {', '.join(loaded.replacement.values) or loaded.replacement.description}"
    )
    output.line(
        f"  candidates       {loaded.replacement.candidates_per_object} per object, mode {loaded.replacement.mode}"
    )
    output.line(f"  export           {loaded.export.format} -> {loaded.export.path}")
    output.line(f"  capabilities     {', '.join(capabilities)}")
    for note in plan_splits.notes:
        output.line(f"  note             {note}")


@app.command("schema")
def schema_command(
    output_path: Path | None = typer.Option(
        None, "--output", help="Write the schema to a file instead of stdout."
    ),
) -> None:
    """Print the generated JSON Schema of the recipe format."""
    schema = Recipe.model_json_schema()
    text = json.dumps(schema, indent=2, ensure_ascii=False)
    if output_path is None:
        typer.echo(text)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    typer.echo(f"wrote {output_path}")


@app.command("init")
def init_command(
    target: Path = typer.Argument(Path("recipe.yaml"), help="Where to write the starter recipe."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file."),
) -> None:
    """Write a documented starter recipe."""
    if target.exists() and not force:
        typer.echo(f"error [RECIPE_INVALID]: {target} already exists; pass --force to replace it", err=True)
        raise typer.Exit(code=2)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_TEMPLATE, encoding="utf-8")
    typer.echo(f"wrote {target}")


@app.command("show")
def show_command(
    recipe: Path = typer.Argument(..., help="Recipe file to print."),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Print a recipe's normalised form after validation."""
    output = Output(json_mode=json_mode)
    try:
        loaded = load_recipe(recipe)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc
    if json_mode:
        typer.echo(json.dumps(loaded.snapshot(), indent=2, ensure_ascii=False))
        return
    import yaml

    typer.echo(yaml.safe_dump(loaded.snapshot(), sort_keys=False))


def _split_plan(loaded: Recipe):
    from vidliner.control.splits import plan_splits

    return plan_splits(loaded)
