"""Post-generation re-detection and re-segmentation — the loop that makes the pipeline closed.

Without this stage the pipeline is **open**: the mask handed to the generator is the same mask used
to compare geometry, to compute the boundary metric, and to build the exported annotation. That makes
several checks tautological — a generator that ignores the mask entirely still passes
``target_presence`` and ``geometry``, because both compare the input mask with itself. On real
backends that is the difference between "we asked for an SUV" and "there is an SUV in the output".

This operator closes the loop. It runs the object detector on the *generated* image, matches a
detection to the region that was replaced, and re-segments that detection to obtain the mask of the
object that is actually there. Everything downstream then describes the object that exists:

* ``geometry`` compares the source object's mask against the **regenerated** mask;
* ``annotation_consistency`` is computed from the regenerated mask;
* ``target_presence`` requires a detection to have been found at all;
* the annotation exported to the dataset is rebuilt from the regenerated mask.

Not finding the object is an **outcome, not an error**: the operator returns an empty mask and
``found=False``, the quality stage turns that into ``OBJECT_NOT_FOUND``, and the acceptance gate
rejects the candidate with a machine-readable reason. A pipeline that silently produced a label for
an object that is not in the image would be worse than one that reports nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vidliner.backends.imageops import mask_geometry_annotation
from vidliner.capabilities.backend import DetectionRequest, SegmentationRequest
from vidliner.capabilities.names import CAP_INSTANCE_SEGMENTATION, CAP_OBJECT_DETECTION
from vidliner.core.errors import ErrorCode, OperatorFailure, VidlinerError
from vidliner.core.results import ArtifactRef
from vidliner.domain.enums import Determinism, PortType, StageName
from vidliner.domain.instances import ObjectInstance, ObjectItem
from vidliner.domain.masks import MaskRef
from vidliner.domain.shapes import BoundingBox, ImageShape
from vidliner.operators.backend_context import backend_context
from vidliner.operators.base import ExecutionContext, InputSpec, Operator, OperatorSpec, OutputSpec
from vidliner.operators.shared import backend_handle

__all__ = ["RedetectConfig", "RedetectTargetOperator", "register_all"]


class RedetectConfig(BaseModel):
    """Configuration of the re-detection stage."""

    model_config = ConfigDict(extra="forbid")

    min_iou: float = Field(default=0.15, ge=0.0, le=1.0)
    """How much a detection must overlap the replaced region to count as the replacement."""
    min_score: float = Field(default=0.10, ge=0.0, le=1.0)
    detect_class: str = "source"
    """Which class to look for: ``source`` (the class the object had) or ``replacement``.

    A detector's label set is fixed, and a replacement category such as ``sedan`` is usually a
    *sub*-category that is not in it. Asking a detector to report ``sedan`` therefore finds nothing
    even when the replacement succeeded. Detection answers "is there an object of the original kind
    at the target location"; whether the replacement belongs to the requested category is the
    semantic evaluator's job, because that is a judgement about appearance rather than a label lookup.
    """
    require_same_class: bool = True
    """Require the detection's class to match the class being looked for."""
    max_objects: int = Field(default=16, ge=1, le=256)
    prompt: str = "bbox"
    max_coverage: float = Field(default=0.98, gt=0.0, le=1.0)
    max_polygon_points: int = Field(default=96, ge=3, le=512)


class RedetectTargetOperator(Operator):
    """Re-detect and re-segment the replaced object in the generated image."""

    @property
    def spec(self) -> OperatorSpec:
        """Declared contract of the re-detection operator."""
        return OperatorSpec(
            name="verify.redetect",
            version="1.0.0",
            stage=StageName.VERIFY,
            summary="re-detect and re-segment the replaced object in the generated image",
            inputs={
                "image": InputSpec(PortType.IMAGE_REF, "the refined candidate"),
                "item": InputSpec(PortType.INSTANCES, "the target object as segmented before generation"),
                "mask": InputSpec(PortType.MASK_REF, "the mask that was handed to the generator"),
                "plan": InputSpec(PortType.PLAN, "the plan, which names the expected category"),
                "candidate": InputSpec(PortType.CANDIDATE_REF, "the generated candidate payload"),
            },
            outputs={
                "item": OutputSpec(PortType.INSTANCES, "the object found in the generated image"),
                "mask": OutputSpec(PortType.MASK_REF, "the regenerated mask (empty when not found)"),
                "found": OutputSpec(
                    PortType.FLAG, "whether the replaced object was found in the generated image"
                ),
            },
            config_model=RedetectConfig,
            capabilities=(CAP_OBJECT_DETECTION, CAP_INSTANCE_SEGMENTATION),
            determinism=Determinism.DETERMINISTIC,
            timeout_s=300.0,
            max_parallelism=2,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Find the replaced object in the generated image and mask it."""
        config = RedetectConfig.model_validate(context.config)
        image = _artifact(inputs.get("image"), port="image")
        item = _item(inputs.get("item"))
        input_mask = _mask(inputs.get("mask"))
        expected_category = _expected_category(inputs)
        wanted_class = _wanted_class(item, expected_category, config)
        target_box = _input_mask_bbox(context, input_mask)

        found = await self._match(context, image, item, wanted_class, target_box, config)
        if found is None:
            empty = _empty_mask(context, input_mask.shape)
            context.publish(
                "verify.not_found",
                object_id=item.object_id,
                source_class=item.instance.class_name,
                expected_category=expected_category,
                wanted_class=wanted_class,
                target_bbox=list(target_box.as_xyxy()),
            )
            return {"item": item, "mask": empty, "found": False}

        matched, score = found
        regenerated = await self._segment(context, image, matched, input_mask, config)
        if regenerated is None:
            empty = _empty_mask(context, input_mask.shape)
            context.publish(
                "verify.segmentation_failed",
                object_id=item.object_id,
                detected_class=matched.class_name,
                detection_score=round(matched.score, 4),
            )
            return {"item": item, "mask": empty, "found": False}

        mask_ref, polygon = regenerated
        mask_array = context.io.load_mask(mask_ref.artifact)
        bbox, traced, area = mask_geometry_annotation(
            mask_array, max_polygon_points=config.max_polygon_points
        )
        instance = ObjectInstance(
            object_id=item.object_id,
            class_name=matched.class_name,
            bbox=bbox,
            score=matched.score,
            mask_ref=mask_ref,
            polygon=traced or polygon,
            source_frame=item.instance.source_frame,
            track_id=item.instance.track_id,
            area_px=area,
            prompt=config.prompt,
            attributes={
                **item.instance.attributes,
                "regenerated": True,
                "source_class": item.instance.class_name,
                "expected_category": expected_category,
                "detection_score": round(matched.score, 4),
                "match_iou": round(score, 4),
                "source_bbox": list(item.instance.bbox.as_xyxy()),
                "source_area_px": item.instance.area_px or 0,
            },
        )
        context.publish(
            "verify.redetected",
            object_id=item.object_id,
            detected_class=matched.class_name,
            detection_score=round(matched.score, 4),
            match_iou=round(score, 4),
            area_px=area,
            bbox=list(bbox.as_xyxy()),
        )
        return {"item": item.with_instance(instance), "mask": mask_ref, "found": True}

    # -- internals --------------------------------------------------------- #

    async def _match(
        self,
        context: ExecutionContext,
        image: ArtifactRef,
        item: ObjectItem,
        wanted_class: str,
        target_box: BoundingBox,
        config: RedetectConfig,
    ) -> tuple[ObjectInstance, float] | None:
        """Detect objects in the generated image and pick the one that is the replacement."""
        handle = await backend_handle(context, CAP_OBJECT_DETECTION)
        classes = (wanted_class,) if config.require_same_class and wanted_class else ()
        result = await handle.call(
            "detect",
            DetectionRequest(
                image=image,
                classes=classes,
                min_score=config.min_score,
                max_objects=config.max_objects,
                known_classes=(wanted_class,) if wanted_class else (),
            ),
            backend_context(context),
        )
        best: tuple[ObjectInstance, float] | None = None
        for instance in result.instances:
            if config.require_same_class and wanted_class and instance.class_name != wanted_class:
                continue
            overlap = instance.bbox.iou(target_box)
            if overlap < config.min_iou:
                continue
            if best is None or overlap > best[1]:
                best = (instance, overlap)
        if best is None:
            # A detector that reports nothing inside the replaced region means the region is still
            # whatever the generator left there. Recording the whole detection set makes the audit
            # readable: the operator reports what it saw, not only what it failed to find.
            context.publish(
                "verify.detections",
                object_id=item.object_id,
                detections=[
                    {"class": instance.class_name, "score": round(instance.score, 4)}
                    for instance in result.instances
                ],
                min_iou=config.min_iou,
            )
        return best

    async def _segment(
        self,
        context: ExecutionContext,
        image: ArtifactRef,
        matched: ObjectInstance,
        input_mask: MaskRef,
        config: RedetectConfig,
    ) -> tuple[MaskRef, Any] | None:
        """Segment the matched detection to obtain the mask of the object that is actually there."""
        shape = input_mask.shape
        handle = await backend_handle(context, CAP_INSTANCE_SEGMENTATION)
        request = SegmentationRequest(
            image=image,
            image_shape=shape,
            object_id=matched.object_id,
            guidance=config.prompt,
            bbox=matched.bbox.as_xyxy(),
        )
        try:
            result = await handle.call("segment", request, backend_context(context))
        except VidlinerError as exc:
            if exc.code is ErrorCode.MASK_EMPTY:
                return None
            raise
        coverage = result.area_px / shape.pixel_count if shape.pixel_count else 0.0
        if coverage > config.max_coverage:
            raise OperatorFailure(
                f"re-segmentation covered {coverage:.1%} of the frame, which cannot describe a single "
                "replaced object",
                code=ErrorCode.MASK_INVALID,
                detail={"coverage": round(coverage, 4), "max_coverage": config.max_coverage},
            )
        mask_ref = result.mask_ref
        return mask_ref, result.polygon


def _input_mask_bbox(context: ExecutionContext, mask_ref: MaskRef) -> Any:
    """The bounding box of the mask that was handed to the generator.

    That region is what "where the replacement was asked to happen" means, so it is what a
    detection has to overlap to count as the replacement.
    """
    array = context.io.load_mask(mask_ref.artifact)
    if not array.any():
        raise OperatorFailure(
            "the input mask is empty, so there is no region to re-detect in",
            code=ErrorCode.MASK_EMPTY,
        )
    rows, columns = np.nonzero(array)
    return BoundingBox(
        x_min=float(columns.min()),
        y_min=float(rows.min()),
        x_max=float(columns.max() + 1),
        y_max=float(rows.max() + 1),
    )


def _empty_mask(context: ExecutionContext, shape: ImageShape) -> MaskRef:
    """A stored empty mask, so "not found" keeps the same port types as "found"."""
    array = np.zeros((shape.height, shape.width), dtype=bool)
    artifact = context.io.save_mask(array)
    return MaskRef(artifact=artifact, shape=shape, area_px=0)


def _wanted_class(item: ObjectItem, expected_category: str, config: RedetectConfig) -> str:
    """The class the detector should look for, given the configured detection strategy."""
    if config.detect_class == "replacement":
        return expected_category
    return item.instance.class_name or expected_category


def _expected_category(inputs: dict[str, Any]) -> str:
    """The category the generator was asked for, preferring the candidate's own value."""
    candidate = inputs.get("candidate")
    if isinstance(candidate, dict) and candidate.get("category"):
        return str(candidate["category"])
    plan = inputs.get("plan")
    if isinstance(plan, dict):
        return str(plan.get("expected_category") or "")
    return str(getattr(plan, "expected_category", "") or "")


def _artifact(value: Any, *, port: str) -> ArtifactRef:
    if isinstance(value, ArtifactRef):
        return value
    if isinstance(value, dict):
        return ArtifactRef.model_validate(value)
    raise OperatorFailure(
        f"re-detection needs an image artifact on port {port!r}, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
        detail={"port": port},
    )


def _mask(value: Any) -> MaskRef:
    if isinstance(value, MaskRef):
        return value
    if isinstance(value, dict):
        return MaskRef.model_validate(value)
    raise OperatorFailure(
        f"re-detection needs a mask, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
    )


def _item(value: Any) -> ObjectItem:
    if isinstance(value, ObjectItem):
        return value
    if isinstance(value, dict):
        return ObjectItem.model_validate(value)
    raise OperatorFailure(
        f"re-detection needs a target item, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
    )


def register_all(registry: Any) -> None:
    """Register this module's operators in ``registry``."""
    registry.register(RedetectTargetOperator)
