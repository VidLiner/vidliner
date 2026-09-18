"""Turning a discovered dataset into the sample contexts a job injects.

This is the seam between "what is on disk" and "what the graph runs against". It is deliberately a
separate module from the runner so that the CLI, the tests, and an embedding application all build
their contexts the same way — including the parts that are easy to get subtly wrong: the perceptual
hash used for duplicate control, the annotation carried in from the source dataset, and the split
each sample belongs to.
"""

from __future__ import annotations

import json
from pathlib import Path

from vidliner.annotations.base import read_annotations
from vidliner.control.sources import DatasetSample, DiscoveryResult
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.core.results import ArtifactRef
from vidliner.domain.annotations import ANNOTATION_FORMATS, AnnotationBundle
from vidliner.domain.media import probe_image_shape
from vidliner.domain.recipe import Recipe
from vidliner.domain.shapes import ImageShape
from vidliner.operators.base import SampleContext
from vidliner.quality.fingerprint import perceptual_hash
from vidliner.storage.workspace import ArtifactStore

__all__ = ["build_sample_context", "build_sample_contexts", "load_source_annotations"]


def load_source_annotations(
    recipe: Recipe,
    discovery: DiscoveryResult,
) -> dict[str, AnnotationBundle]:
    """Read the source dataset's annotations into the internal IR, when the recipe declares a format.

    Returns:
        Bundles keyed by ``relative_path``. Empty when the recipe declares no source format, which is
        the common case for an unlabelled image directory.

    Raises:
        ValidationFailure: when a declared format cannot be read.
    """
    format_name = recipe.dataset.format
    if not format_name:
        return {}
    if format_name not in ANNOTATION_FORMATS:
        raise ValidationFailure(
            f"recipe declares unsupported dataset format {format_name!r}; supported formats are "
            f"{ANNOTATION_FORMATS}",
            code=ErrorCode.ANNOTATION_INVALID,
            detail={"format": format_name},
        )
    shape_cache: dict[str, ImageShape] = {sample.relative_path: sample.shape for sample in discovery.samples}

    def shape_lookup(relative: str) -> ImageShape | None:
        cached = shape_cache.get(relative)
        if cached is not None:
            return cached
        path = discovery.root / relative
        try:
            return probe_image_shape(path)
        except ValidationFailure:
            return None

    bundles, _categories = read_annotations(format_name, discovery.root, shape_lookup=shape_lookup)
    return bundles


def build_sample_context(
    sample: DatasetSample,
    *,
    root: Path,
    store: ArtifactStore,
    annotation: AnnotationBundle | None = None,
    compute_perceptual_hash: bool = True,
    perceptual_hash_algorithm: str = "phash",
) -> SampleContext:
    """Build the context one sample is executed with."""
    identifier = sample.perceptual_hash
    if compute_perceptual_hash and not identifier and sample.media_kind.value == "image":
        identifier = _perceptual_hash(sample.absolute_path, algorithm=perceptual_hash_algorithm)
    return SampleContext(
        sample_id=sample.sample_id,
        source_root=str(root),
        source_path=str(sample.absolute_path),
        relative_path=sample.relative_path,
        split=sample.split,
        media_kind=sample.media_kind.value,
        root_digest=sample.root_digest,
        source_digest=sample.digest,
        image_artifact=_image_artifact(sample, store),
        source_annotation=annotation.model_dump(mode="json") if annotation is not None else None,
        source_instances=(),
        perceptual_hash=identifier,
        metadata={"size_bytes": sample.size_bytes, "shape": sample.shape.model_dump(mode="json")},
    )


def build_sample_contexts(
    discovery: DiscoveryResult,
    *,
    recipe: Recipe,
    store: ArtifactStore,
    annotations: dict[str, AnnotationBundle] | None = None,
) -> dict[str, SampleContext]:
    """Build contexts for every augmentable sample, keyed by sample id."""
    bundles = annotations if annotations is not None else load_source_annotations(recipe, discovery)
    contexts: dict[str, SampleContext] = {}
    for sample in discovery.augmentable:
        contexts[sample.sample_id] = build_sample_context(
            sample,
            root=discovery.root,
            store=store,
            annotation=bundles.get(sample.relative_path),
            perceptual_hash_algorithm=recipe.duplicates.perceptual_hash,
        )
    return contexts


def _image_artifact(sample: DatasetSample, store: ArtifactStore) -> ArtifactRef | None:
    """A registered artifact for the sample's source bytes, or ``None`` when already stored.

    The ingest node produces the authoritative artifact; this exists so that the sample context can
    already name the source bytes for backends that run before ingest has completed (which the
    pipeline does not do, but an embedding application may).
    """
    if not store.exists(sample.digest):
        return None
    from vidliner.domain.enums import ArtifactKind

    return ArtifactRef(
        kind=ArtifactKind.SOURCE_MEDIA,
        digest=sample.digest,
        media_type="application/octet-stream",
        size_bytes=sample.size_bytes,
        width=sample.shape.width,
        height=sample.shape.height,
        suffix=sample.absolute_path.suffix.lstrip(".") or "bin",
    )


def _perceptual_hash(path: Path, *, algorithm: str) -> str | None:
    if algorithm == "none":
        return None
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as handle:
            array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None
    return perceptual_hash(array, algorithm=algorithm).hex


def write_split_manifest(discovery: DiscoveryResult, target: Path) -> Path:
    """Write a ``relative_path → split`` document, so a discovered split assignment is reproducible."""
    mapping = {sample.relative_path: sample.split for sample in discovery.samples}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    return target
