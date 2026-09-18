"""Scene analysis: measure the target object's surroundings before planning a replacement.

The measurements produced here are what make a *strict* replacement possible: without lighting
direction, ground contact, shadow region, and occlusion relations, a planner can only ask a model to
"make it look right". With them, the plan can state exactly what must not change, and the quality
stage can check whether it did.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from vidliner.backends.imageops import load_image_array
from vidliner.capabilities.backend import SceneAnalysisRequest
from vidliner.capabilities.names import CAP_SCENE_ANALYSIS
from vidliner.core.errors import ErrorCode, OperatorFailure
from vidliner.core.results import ArtifactRef
from vidliner.domain.enums import Determinism, PortType, StageName
from vidliner.domain.instances import ObjectInstance, ObjectItem
from vidliner.domain.shapes import ImageShape
from vidliner.operators.backend_context import backend_context
from vidliner.operators.base import ExecutionContext, InputSpec, Operator, OperatorSpec, OutputSpec
from vidliner.operators.shared import backend_handle, require_instances

__all__ = ["AnalyseSceneOperator", "SceneConfig"]


class SceneConfig(BaseModel):
    """Configuration of the scene-analysis operator."""

    model_config = ConfigDict(extra="forbid")

    neighbours: int = 8
    include_luminance: bool = True


class AnalyseSceneOperator(Operator):
    """Measure the scene around one target object."""

    @property
    def spec(self) -> OperatorSpec:
        """Declared contract of the scene-analysis operator."""
        return OperatorSpec(
            name="scene.analyse",
            version="1.0.0",
            stage=StageName.SCENE,
            summary="measure lighting, orientation, scale, occlusion, and ground contact",
            inputs={
                "image": InputSpec(PortType.IMAGE_REF, "decoded frame"),
                "asset": InputSpec(PortType.MEDIA_ASSET, "source media description"),
                "item": InputSpec(PortType.INSTANCES, "the segmented target object"),
                "instances": InputSpec(
                    PortType.INSTANCES, "every detected object in the frame", required=False
                ),
            },
            outputs={"scene": OutputSpec(PortType.SCENE_CONTEXT, "measured scene context")},
            config_model=SceneConfig,
            capabilities=(CAP_SCENE_ANALYSIS,),
            determinism=Determinism.DETERMINISTIC,
            timeout_s=120.0,
            max_parallelism=2,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Analyse the scene and return the measured context."""
        image = _artifact(inputs.get("image"), port="image")
        item = _item(inputs.get("item"))
        shape = _shape_of(inputs, image)
        config = SceneConfig.model_validate(context.config)
        mask_ref = item.instance.mask_ref
        if mask_ref is None:
            raise OperatorFailure(
                f"scene analysis requires the target object {item.object_id} to be segmented first",
                code=ErrorCode.MASK_EMPTY,
                detail={"object_id": item.object_id},
            )
        mask_array = context.io.load_mask(mask_ref.artifact)
        luminance = _luminance(context, image) if config.include_luminance else None
        neighbours = _neighbours(inputs, item, limit=config.neighbours)
        handle = await backend_handle(context, CAP_SCENE_ANALYSIS)
        request = SceneAnalysisRequest(
            image=image,
            image_shape=shape,
            target=item.instance,
            neighbours=neighbours,
            mask_array=mask_array,
            luminance=luminance,
        )
        scene = await handle.call("analyse", request, backend_context(context))
        context.publish(
            "scene.completed",
            backend=handle.backend_id,
            object_id=item.object_id,
            scale_class=scene.scale_class.value,
            occluded=scene.is_occluded,
        )
        return {"scene": scene}


def _luminance(context: ExecutionContext, image: ArtifactRef) -> np.ndarray:
    array = load_image_array(context.io, image)
    return (array * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=2).astype(np.float32)


def _neighbours(inputs: dict[str, Any], item: ObjectItem, *, limit: int) -> tuple[ObjectInstance, ...]:
    """Other objects in the frame, preferring the ones that actually overlap the target."""
    candidates: list[ObjectInstance] = []
    for value in (inputs.get("instances"), item.neighbours):
        if value is None:
            continue
        try:
            candidates.extend(require_instances(value, where="scene.analyse"))
        except Exception:
            continue
    unique: dict[str, ObjectInstance] = {}
    for instance in candidates:
        if instance.object_id != item.object_id:
            unique.setdefault(instance.object_id, instance)
    ordered = sorted(unique.values(), key=lambda instance: -instance.bbox.iou(item.instance.bbox))
    return tuple(ordered[:limit])


def _item(value: Any) -> ObjectItem:
    if isinstance(value, ObjectItem):
        return value
    if isinstance(value, dict):
        return ObjectItem.model_validate(value)
    raise OperatorFailure(
        f"scene analysis needs a target item, received {type(value).__name__}",
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
        "scene analysis needs the frame dimensions",
        code=ErrorCode.MEDIA_CORRUPT,
        detail={"artifact": image.digest[:12]},
    )


def _artifact(value: Any, *, port: str) -> ArtifactRef:
    if isinstance(value, ArtifactRef):
        return value
    if isinstance(value, dict):
        return ArtifactRef.model_validate(value)
    raise OperatorFailure(
        f"scene analysis needs an image artifact on port {port!r}, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
        detail={"port": port},
    )


def register_all(registry: Any) -> None:
    """Register this module's operators in ``registry``."""
    registry.register(AnalyseSceneOperator)
