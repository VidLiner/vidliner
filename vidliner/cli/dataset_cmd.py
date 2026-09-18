"""``vidliner inspect``: analyse a dataset directory without running a job."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from vidliner.cli.common import Output, fail
from vidliner.control.prepare import load_source_annotations
from vidliner.control.sources import DatasetSource
from vidliner.core.errors import ValidationFailure
from vidliner.domain.recipe import Recipe

if TYPE_CHECKING:
    from vidliner.domain.recipe import Recipe as RecipeType

__all__ = ["inspect_command"]


def inspect_command(
    dataset: Path = typer.Argument(..., help="Dataset directory to inspect."),
    fmt: str | None = typer.Option(None, "--format", help="Annotation format of the dataset."),
    split_mode: str = typer.Option("none", "--split-mode", help="none, directory, filename, or manifest."),
    split_manifest: Path | None = typer.Option(
        None, "--split-manifest", help="Split manifest for --split-mode manifest."
    ),
    limit: int = typer.Option(0, "--limit", help="Stop after this many samples (0 = all)."),
    quiet: bool = typer.Option(False, "--quiet"),
    json_mode: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Report media kinds, dimensions, splits, and annotation coverage for a dataset."""
    output = Output(json_mode=json_mode, quiet=quiet)
    try:
        recipe = _inspection_recipe(
            dataset,
            fmt=fmt,
            split_mode=split_mode,
            split_manifest=split_manifest,
            limit=limit,
        )
        discovery = DatasetSource(root=dataset.expanduser().resolve(), recipe=recipe).discover()
        annotations = load_source_annotations(recipe, discovery)
    except ValidationFailure as exc:
        raise typer.Exit(code=fail(exc, output=output)) from exc

    annotated = sum(1 for sample in discovery.samples if sample.relative_path in annotations)
    objects = sum(bundle.object_count for bundle in annotations.values())
    classes: dict[str, int] = {}
    for bundle in annotations.values():
        for category in bundle.categories:
            classes[category.name] = classes.get(category.name, 0) + len(
                bundle.objects_in_category(category.name)
            )
    dimensions = sorted({f"{sample.shape.width}x{sample.shape.height}" for sample in discovery.samples})

    payload = {
        "dataset": str(discovery.root),
        "samples": discovery.count,
        "splits": discovery.split_counts,
        "augment_splits": list(discovery.augment_splits),
        "media": {"images": sum(1 for s in discovery.samples if s.media_kind.value == "image"), "videos": 0},
        "dimensions": dimensions[:20],
        "annotation_format": fmt,
        "annotated_samples": annotated,
        "annotated_objects": objects,
        "classes": dict(sorted(classes.items())),
        "skipped": [{"path": path, "reason": reason} for path, reason in discovery.skipped[:20]],
        "skipped_total": len(discovery.skipped),
        "notes": list(discovery.notes),
    }
    output.payload(payload)
    if json_mode:
        return
    output.line(f"dataset   {discovery.root}")
    output.line(f"samples   {discovery.count}")
    output.line(
        "splits    "
        + (", ".join(f"{name}={count}" for name, count in sorted(discovery.split_counts.items())) or "none")
    )
    output.line(f"augment   {', '.join(discovery.augment_splits)}")
    output.line(f"images    {payload['media']['images']}   dimensions {', '.join(dimensions[:5]) or 'n/a'}")
    if fmt:
        output.line(f"annotated {annotated}/{discovery.count} samples, {objects} objects")
        if classes:
            output.line(
                "classes   " + ", ".join(f"{name}={count}" for name, count in sorted(classes.items())[:12])
            )
    if discovery.skipped:
        output.line(f"skipped   {len(discovery.skipped)} file(s); first: {discovery.skipped[0][1]}")
    for note in discovery.notes:
        output.line(f"note      {note}")


def _inspection_recipe(
    dataset: Path,
    *,
    fmt: str | None,
    split_mode: str,
    split_manifest: Path | None,
    limit: int,
) -> RecipeType:
    """Build the minimal recipe an inspection pass needs."""
    document: dict[str, Any] = {
        "recipe": "inspect",
        "dataset": {
            "input": str(dataset),
            "format": fmt,
            "splits": {
                "mode": split_mode,
                "manifest": str(split_manifest) if split_manifest else None,
            },
            "max_samples": limit,
        },
        "target": {"classes": ["object"]},
        "replacement": {"values": ["replacement"]},
    }
    try:
        return Recipe.model_validate(document)
    except Exception as exc:  # pydantic ValidationError
        raise ValidationFailure(f"could not build an inspection recipe: {exc}") from exc
