"""Segmentation: give one target object a pixel-level mask.

A target without a mask cannot be replaced, cannot be measured, and cannot be annotated, so this
stage is a hard prerequisite for everything downstream. It also enforces the first structural rule
of the pipeline: an empty mask is a *failure*, never an empty result that quietly produces a
candidate with no object in it.

The node is per-target: it receives one :class:`~vidliner.domain.instances.ObjectItem` and returns
the same item carrying its mask. That is what makes the graph fan out by object rather than
processing every object in one node.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vidliner.capabilities.backend import SegmentationRequest
from vidliner.capabilities.names import CAP_INSTANCE_SEGMENTATION
from vidliner.core.errors import ErrorCode, OperatorFailure
from vidliner.core.results import ArtifactRef
from vidliner.domain.enums import Determinism, PortType, StageName
from vidliner.domain.instances import ObjectItem
from vidliner.domain.shapes import ImageShape
from vidliner.operators.backend_context import backend_context
from vidliner.operators.base import ExecutionContext, InputSpec, Operator, OperatorSpec, OutputSpec
from vidliner.operators.shared import backend_handle

__all__ = ["SegmentConfig", "SegmentTargetOperator", "register_all"]


class SegmentConfig(BaseModel):
    """Configuration of the segmentation operator."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = "bbox"
    """``bbox``, ``point``, or ``text``. Text prompting requires a backend that supports it."""
    min_area_px: int = Field(default=16, ge=1)
    max_coverage: float = Field(default=0.98, gt=0.0, le=1.0)
    text_prompt_template: str = "the {class_name}"


class SegmentTargetOperator(Operator):
    """Segment one target object."""

    @property
    def spec(self) -> OperatorSpec:
        """Declared contract of the segmentation operator."""
        return OperatorSpec(
            name="segment.target",
            version="1.0.0",
            stage=StageName.SEGMENT,
            summary="produce a pixel-level mask for one target object",
            inputs={
                "image": InputSpec(PortType.IMAGE_REF, "decoded frame"),
                "asset": InputSpec(PortType.MEDIA_ASSET, "source media description"),
                "item": InputSpec(PortType.INSTANCES, "the target object to segment"),
            },
            outputs={
                "item": OutputSpec(PortType.INSTANCES, "the target object carrying its mask"),
                "mask": OutputSpec(PortType.MASK_REF, "the mask artifact on its own port"),
            },
            config_model=SegmentConfig,
            capabilities=(CAP_INSTANCE_SEGMENTATION,),
            determinism=Determinism.DETERMINISTIC,
            timeout_s=300.0,
            max_parallelism=2,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Segment the target object and attach the mask to its instance."""
        image = _artifact(inputs.get("image"), port="image")
        item = _item(inputs.get("item"))
        shape = _shape_of(inputs, image)
        config = SegmentConfig.model_validate(context.config)
        handle = await backend_handle(context, CAP_INSTANCE_SEGMENTATION)
        request = SegmentationRequest(
            image=image,
            image_shape=shape,
            object_id=item.object_id,
            guidance=config.prompt,
            **_prompt(item, config),
        )
        result = await handle.call("segment", request, backend_context(context))
        if result.area_px < config.min_area_px:
            raise OperatorFailure(
                f"segmentation produced a mask of {result.area_px} pixels for object {item.object_id}, "
                f"below the {config.min_area_px} pixel minimum",
                code=ErrorCode.MASK_EMPTY,
                detail={"object_id": item.object_id, "area_px": result.area_px},
            )
        coverage = result.area_px / shape.pixel_count if shape.pixel_count else 0.0
        if coverage > config.max_coverage:
            raise OperatorFailure(
                f"segmentation covered {coverage:.1%} of the frame for object {item.object_id}; "
                "refusing to replace almost the entire image",
                code=ErrorCode.MASK_INVALID,
                detail={"object_id": item.object_id, "coverage": round(coverage, 4)},
            )
        mask_ref = result.mask_ref
        instance = item.instance.with_mask(mask_ref, result.polygon)
        instance = instance.with_attributes(segmentation_score=round(result.score, 4))
        updated = item.with_instance(instance)
        context.publish(
            "segment.completed",
            backend=handle.backend_id,
            object_id=item.object_id,
            area_px=result.area_px,
        )
        return {"item": updated, "mask": mask_ref}


def _prompt(item: ObjectItem, config: SegmentConfig) -> dict[str, Any]:
    if config.prompt == "point":
        centre = item.instance.bbox.centroid
        return {"point": (centre.x, centre.y)}
    if config.prompt == "text":
        return {"text": config.text_prompt_template.format(class_name=item.class_name)}
    return {"bbox": item.instance.bbox.as_xyxy()}


def _item(value: Any) -> ObjectItem:
    if isinstance(value, ObjectItem):
        return value
    if isinstance(value, dict):
        return ObjectItem.model_validate(value)
    raise OperatorFailure(
        f"segmentation needs a target item, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
    )


def _shape_of(inputs: dict[str, Any], image: ArtifactRef) -> ImageShape:
    asset = inputs.get("asset")
    shape = getattr(asset, "shape", None)
    if isinstance(shape, ImageShape):
        return shape
    if isinstance(asset, dict) and asset.get("shape"):
        return ImageShape.model_validate(asset["shape"])
    if image.width and image.height:
        return ImageShape(width=image.width, height=image.height)
    raise OperatorFailure(
        "segmentation needs the frame dimensions; neither the asset nor the artifact reported them",
        code=ErrorCode.MEDIA_CORRUPT,
        detail={"artifact": image.digest[:12]},
    )


def _artifact(value: Any, *, port: str) -> ArtifactRef:
    if isinstance(value, ArtifactRef):
        return value
    if isinstance(value, dict):
        return ArtifactRef.model_validate(value)
    raise OperatorFailure(
        f"segmentation needs an image artifact on port {port!r}, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
        detail={"port": port},
    )


def register_all(registry: Any) -> None:
    """Register this module's operators in ``registry``."""
    registry.register(SegmentTargetOperator)
