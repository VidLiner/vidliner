"""Ingest: read a source sample into the pipeline.

Ingest is where a source file becomes a first-class pipeline input, and where the rules that protect
the dataset are enforced:

* the source file is opened **read-only** and never modified;
* the file is confirmed to live inside the workspace, so a recipe cannot reach outside it;
* the decoded frame is stored as a content-addressed artifact, and its SHA-256 becomes the root of
  every identity derived from it;
* video inputs are refused with an explicit phase-2 error rather than a guess;
* if the source file changed since the dataset was inspected, ingest refuses to continue, because
  augmenting bytes that no longer match the recorded digest would silently break provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.annotations import AnnotationBundle
from vidliner.domain.enums import ArtifactKind, Determinism, MediaKind, PortType, StageName
from vidliner.domain.media import MediaAsset, inspect_media
from vidliner.operators.base import (
    ExecutionContext,
    InputSpec,
    Operator,
    OperatorSpec,
    OutputSpec,
    SampleContext,
)
from vidliner.operators.shared import annotation_from_payload, store

__all__ = ["ImageIngestOperator", "IngestConfig", "register_all"]


class IngestConfig(BaseModel):
    """Configuration of the ingest operator."""

    model_config = ConfigDict(extra="forbid")

    reject_video: bool = True
    max_pixels: int = Field(default=64_000_000, gt=0)


class ImageIngestOperator(Operator):
    """Decode one source sample and register it as an artifact."""

    @property
    def spec(self) -> OperatorSpec:
        """Declared contract of the ingest operator."""
        return OperatorSpec(
            name="ingest.image",
            version="1.0.0",
            stage=StageName.INGEST,
            summary="read a source sample into the pipeline without modifying it",
            inputs={
                "source": InputSpec(PortType.ANY, "sample-level source reference"),
                "annotation": InputSpec(
                    PortType.ANNOTATION, "source annotation, when present", required=False
                ),
            },
            outputs={
                "asset": OutputSpec(PortType.MEDIA_ASSET, "immutable description of the source media"),
                "image": OutputSpec(PortType.IMAGE_REF, "decoded source frame as an artifact"),
                "annotation": OutputSpec(
                    PortType.ANNOTATION, "source annotation in internal IR form, when the dataset has one"
                ),
            },
            config_model=IngestConfig,
            capabilities=(),
            determinism=Determinism.DETERMINISTIC,
            timeout_s=120.0,
            max_parallelism=4,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Decode the sample and return its asset and decoded artifact."""
        del inputs  # the source arrives through the sample context, which the job injected
        sample = _require_sample(context)
        config = IngestConfig.model_validate(context.config)
        if sample.media_kind == MediaKind.VIDEO.value and config.reject_video:
            raise ValidationFailure(
                "video ingestion and replacement are a phase-2 capability; this build handles still images",
                code=ErrorCode.VIDEO_NOT_SUPPORTED,
                detail={"path": sample.source_path},
            )
        resolved = _resolve_source(context, sample)
        asset = inspect_media(resolved, relative_to=store(context).workspace.root)
        if sample.source_digest and asset.digest != sample.source_digest:
            raise ValidationFailure(
                "the source file changed after the dataset was inspected; refusing to augment stale bytes",
                code=ErrorCode.ARTIFACT_DIGEST_MISMATCH,
                detail={"path": sample.source_path, "expected": sample.source_digest, "actual": asset.digest},
            )
        if asset.shape.pixel_count > config.max_pixels:
            raise ValidationFailure(
                f"source frame has {asset.shape.pixel_count} pixels, above the configured cap of "
                f"{config.max_pixels}",
                code=ErrorCode.MEDIA_TOO_LARGE,
                detail={"path": sample.source_path, "pixels": asset.shape.pixel_count},
            )
        image_artifact = store(context).write(
            resolved.read_bytes(),
            kind=ArtifactKind.SOURCE_MEDIA,
            media_type=asset.media_type,
            suffix=asset.suffix.lstrip(".") or "bin",
            width=asset.shape.width,
            height=asset.shape.height,
        )
        context.publish(
            "ingest.completed",
            sample_id=sample.sample_id,
            digest=asset.digest,
            shape=[asset.shape.width, asset.shape.height],
            media_kind=asset.kind.value,
        )
        return {"asset": asset, "image": image_artifact, "annotation": source_bundle(sample, asset)}


def source_bundle(sample: SampleContext, asset: MediaAsset) -> AnnotationBundle:
    """The source annotation for a sample, or an empty bundle when the dataset had none."""
    if sample.source_annotation is not None:
        return annotation_from_payload(sample.source_annotation)
    return AnnotationBundle(
        sample_id=sample.sample_id,
        image_shape=asset.shape,
        categories=(),
        objects=(),
        source_format=None,
    )


def _require_sample(context: ExecutionContext) -> SampleContext:
    if context.sample is None:
        raise ValidationFailure(
            "ingest requires a sample context; the job did not inject one",
            code=ErrorCode.PORT_UNBOUND,
            detail={"node_id": context.node.node_id},
        )
    return context.sample


def _resolve_source(context: ExecutionContext, sample: SampleContext) -> Path:
    """Resolve the source path inside the workspace boundary and confirm it is a file."""
    path = Path(sample.source_path)
    if not path.is_absolute():
        path = store(context).workspace.root / path
    resolved = store(context).workspace.resolve_inside(path, must_exist=True)
    if not resolved.is_file():
        raise ValidationFailure(
            f"source {resolved} is not a regular file",
            code=ErrorCode.MEDIA_UNSUPPORTED,
            detail={"path": str(resolved)},
        )
    return resolved


def register_all(registry: Any) -> None:
    """Register this module's operators in ``registry``."""
    registry.register(ImageIngestOperator)
