"""Detection: find the objects a recipe might want to replace."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vidliner.capabilities.backend import DetectionRequest
from vidliner.capabilities.names import CAP_OBJECT_DETECTION
from vidliner.core.errors import ErrorCode, OperatorFailure
from vidliner.core.results import ArtifactRef
from vidliner.domain.enums import Determinism, MediaKind, PortType, StageName
from vidliner.domain.media import MediaAsset
from vidliner.operators.backend_context import backend_context
from vidliner.operators.base import ExecutionContext, InputSpec, Operator, OperatorSpec, OutputSpec
from vidliner.operators.shared import backend_handle

__all__ = ["DetectConfig", "DetectObjectsOperator", "register_all"]


class DetectConfig(BaseModel):
    """Configuration of the detection operator."""

    model_config = ConfigDict(extra="forbid")

    classes: tuple[str, ...] = ()
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    max_objects: int = Field(default=64, ge=1, le=1024)


class DetectObjectsOperator(Operator):
    """Run the bound object detector over one frame."""

    @property
    def spec(self) -> OperatorSpec:
        """Declared contract of the detection operator."""
        return OperatorSpec(
            name="detect.objects",
            version="1.0.0",
            stage=StageName.DETECT,
            summary="locate objects and report class, confidence, and bounding box",
            inputs={
                "image": InputSpec(PortType.IMAGE_REF, "decoded frame to detect in"),
                "asset": InputSpec(PortType.MEDIA_ASSET, "source media description", required=False),
                "annotation": InputSpec(
                    PortType.ANNOTATION,
                    "source annotation, which defines the label vocabulary",
                    required=False,
                ),
            },
            outputs={"instances": OutputSpec(PortType.INSTANCES, "detected objects")},
            config_model=DetectConfig,
            capabilities=(CAP_OBJECT_DETECTION,),
            determinism=Determinism.DETERMINISTIC,
            timeout_s=300.0,
            max_parallelism=2,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Detect objects and return them on the ``instances`` port."""
        image = _artifact(inputs.get("image"), port="image")
        asset = inputs.get("asset")
        config = DetectConfig.model_validate(context.config)
        media_kind = MediaKind.IMAGE
        if isinstance(asset, MediaAsset):
            media_kind = asset.kind
        elif isinstance(asset, dict) and asset.get("kind"):
            try:
                media_kind = MediaKind(str(asset["kind"]))
            except ValueError:  # pragma: no cover - defensive
                media_kind = MediaKind.IMAGE

        handle = await backend_handle(context, CAP_OBJECT_DETECTION)
        known = tuple(_known_classes(inputs.get("annotation")))
        request = DetectionRequest(
            image=image,
            classes=tuple(config.classes),
            min_score=config.min_score,
            max_objects=config.max_objects,
            media_kind=media_kind,
            known_classes=known,
        )
        result = await handle.call("detect", request, backend_context(context))
        context.publish(
            "detect.completed",
            backend=handle.backend_id,
            found=len(result.instances),
            classes=sorted({instance.class_name for instance in result.instances}),
        )
        return {"instances": tuple(result.instances)}


def _known_classes(annotation: Any) -> list[str]:
    """The class vocabulary of the source dataset, when one was supplied."""
    if annotation is None:
        return []
    bundle = annotation
    if isinstance(annotation, dict):
        categories = annotation.get("categories") or []
        return [
            str(category.get("name"))
            for category in categories
            if isinstance(category, dict) and category.get("name")
        ]
    categories = getattr(bundle, "categories", ())
    return [str(category.name) for category in categories]


def _artifact(value: Any, *, port: str) -> ArtifactRef:
    if isinstance(value, ArtifactRef):
        return value
    if isinstance(value, dict):
        return ArtifactRef.model_validate(value)
    raise OperatorFailure(
        f"detection needs an image artifact on port {port!r}, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
        detail={"port": port},
    )


def register_all(registry: Any) -> None:
    """Register this module's operators in ``registry``."""
    registry.register(DetectObjectsOperator)
