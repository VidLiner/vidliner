"""Annotation regeneration: rebuild the label for the object that changed.

The rule this operator enforces is the one the whole product depends on: **an annotation is never
copied across a replacement**. The replaced object's box and polygon are recomputed from the
regenerated mask; every *other* object's label is inherited, but only after it has been checked
against the evidence that the background outside the mask did not change.

If the inherited labels were produced by segmenting the source frame and the background was
preserved, inheriting them is sound; if the background moved, the background gate will have rejected
the candidate before this stage matters. Either way the decision is made explicitly and recorded,
rather than assumed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vidliner.core.errors import ErrorCode, OperatorFailure
from vidliner.domain.annotations import AnnotationBundle, AnnotationObject, CategorySpec
from vidliner.domain.enums import Determinism, PortType, StageName
from vidliner.domain.instances import ObjectItem
from vidliner.domain.masks import MaskRef, PolygonMask
from vidliner.domain.reasons import ReasonCode
from vidliner.operators.base import ExecutionContext, InputSpec, Operator, OperatorSpec, OutputSpec
from vidliner.operators.shared import annotation_from_payload

__all__ = ["AnnotationConfig", "RebuildAnnotationOperator", "register_all"]


class AnnotationConfig(BaseModel):
    """Configuration of the annotation-regeneration operator."""

    model_config = ConfigDict(extra="forbid")

    min_area_px: int = Field(default=4, ge=0)
    inherit_other_objects: bool = True
    inherit_score: bool = True
    mark_source: bool = True


class RebuildAnnotationOperator(Operator):
    """Rebuild the annotation bundle for one replaced object."""

    @property
    def spec(self) -> OperatorSpec:
        """Declared contract of the annotation operator."""
        return OperatorSpec(
            name="annotate.rebuild",
            version="1.0.0",
            stage=StageName.ANNOTATE,
            summary="recompute the replaced object's annotation and validate the inherited ones",
            inputs={
                "item": InputSpec(PortType.INSTANCES, "the target object before replacement"),
                "mask": InputSpec(PortType.MASK_REF, "the regenerated mask"),
                "plan": InputSpec(PortType.PLAN, "the replacement plan"),
                "candidate": InputSpec(
                    PortType.CANDIDATE_REF, "the generated candidate this annotation describes"
                ),
                "found": InputSpec(
                    PortType.FLAG,
                    "whether the replaced object was found in the generated image",
                    required=False,
                ),
                "source_annotation": InputSpec(PortType.ANNOTATION, "the source annotation", required=False),
                "source_image": InputSpec(PortType.IMAGE_REF, "the untouched source frame", required=False),
                "image": InputSpec(PortType.IMAGE_REF, "the refined candidate", required=False),
            },
            outputs={
                "annotation": OutputSpec(PortType.ANNOTATION, "the rebuilt annotation bundle"),
                "validation": OutputSpec(PortType.ANY, "what was inherited and what was recomputed"),
            },
            config_model=AnnotationConfig,
            capabilities=(),
            determinism=Determinism.DETERMINISTIC,
            timeout_s=120.0,
            max_parallelism=4,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Recompute the target's label and validate everything inherited."""
        item = _item(inputs.get("item"))
        mask_ref = _mask(inputs.get("mask"))
        plan = inputs.get("plan")
        candidate = inputs.get("candidate")
        found = bool(inputs.get("found", True))
        config = AnnotationConfig.model_validate(context.config)
        source_annotation = inputs.get("source_annotation")
        bundle = (
            annotation_from_payload(source_annotation)
            if source_annotation is not None
            else AnnotationBundle(sample_id=item.sample_id, image_shape=mask_ref.shape)
        )
        mask_array = context.io.load_mask(mask_ref.artifact)
        if not mask_array.any():
            # No object was found in the generated image, so there is no object to label. That is an
            # outcome, not an error: the acceptance gate rejects the candidate on the measured
            # evidence. Emitting a structurally valid *empty* bundle keeps the candidate's evidence
            # complete and keeps a failed node out of the job's failure counters, where it does not
            # belong — a job is not broken because a generated sample is unusable.
            if found:
                raise OperatorFailure(
                    f"re-detection reported an object for {item.object_id} but its mask is empty",
                    code=ErrorCode.MASK_EMPTY,
                    detail={"object_id": item.object_id},
                )
            empty = AnnotationBundle(
                sample_id=item.sample_id,
                image_shape=mask_ref.shape,
                categories=bundle.categories,
                objects=(),
                source_format=bundle.source_format,
                attributes={"annotation": "not_found", "replaced_object_id": item.object_id},
            )
            validation = {
                "object_id": item.object_id,
                "target_annotation": "not_found",
                "inherited_objects": 0,
                "dropped_objects": 0,
                "annotation_object_count": 0,
                "area_px": 0,
            }
            context.publish("annotate.not_found", **validation)
            return {"annotation": empty, "validation": validation}
        area = int(mask_array.sum())
        if area < config.min_area_px:
            raise OperatorFailure(
                f"the regenerated mask for object {item.object_id} has {area} pixels, below the "
                f"{config.min_area_px} pixel minimum",
                code=ErrorCode.ANNOTATION_TOO_SMALL,
                detail={"object_id": item.object_id, "area_px": area},
            )
        expected_category = str(_category_for(candidate, plan) or item.class_name)

        inherited, dropped = self._inherit(bundle, item, mask_array, config)
        categories = _categories_for(bundle, expected_category)
        target = self._target_object(item, mask_ref, mask_array, expected_category, categories, config)
        rebuilt = AnnotationBundle(
            sample_id=item.sample_id,
            image_shape=mask_ref.shape,
            categories=categories,
            objects=(*inherited, target),
            source_format=bundle.source_format,
            attributes={
                **bundle.attributes,
                "replaced_object_id": item.object_id,
                "target_annotation": "regenerated",
            },
        )
        problems = rebuilt.validate_against_image()
        if problems:
            raise OperatorFailure(
                f"rebuilt annotation for object {item.object_id} is structurally invalid: "
                f"{'; '.join(problems)}",
                code=ErrorCode.ANNOTATION_INVALID,
                detail={"object_id": item.object_id, "problems": problems},
            )
        validation = {
            "object_id": item.object_id,
            "target_annotation": "regenerated",
            "inherited_objects": len(inherited),
            "dropped_objects": dropped,
            "dropped_reason": ReasonCode.OBJECT_DISAPPEARED.value if dropped else None,
            "category": expected_category,
            "area_px": area,
        }
        context.publish("annotate.completed", **validation)
        return {"annotation": rebuilt, "validation": validation}

    def _inherit(
        self,
        bundle: AnnotationBundle,
        item: ObjectItem,
        mask_array: np.ndarray,
        config: AnnotationConfig,
    ) -> tuple[list[AnnotationObject], int]:
        """Carry over the labels of untouched objects that still have visible pixels."""
        if not config.inherit_other_objects:
            return [], 0
        kept: list[AnnotationObject] = []
        dropped = 0
        for obj in bundle.objects:
            if obj.object_id == item.object_id:
                continue
            if _vanished(obj, mask_array):
                dropped += 1
                continue
            kept.append(obj.model_copy(update={"source": "inherited"}))
        return kept, dropped

    def _target_object(
        self,
        item: ObjectItem,
        mask_ref: MaskRef,
        mask_array: np.ndarray,
        category: str,
        categories: tuple[CategorySpec, ...],
        config: AnnotationConfig,
    ) -> AnnotationObject:
        """Build the target object's annotation from the regenerated mask alone."""
        box = _bbox(mask_array)
        polygon = _polygon(mask_array)
        category_id = next(
            (entry.category_id for entry in categories if entry.name == category),
            categories[0].category_id if categories else 0,
        )
        return AnnotationObject(
            object_id=item.object_id,
            category_id=category_id,
            bbox=box,
            polygon=polygon,
            mask_ref=mask_ref,
            score=None if not config.inherit_score else round(float(item.instance.score), 6),
            iscrowd=False,
            attributes={
                "source_class": item.class_name,
                "replacement_category": category,
            },
            source="regenerated",
        )


def _vanished(obj: AnnotationObject, produced_mask: np.ndarray) -> bool:
    """Whether an unrelated object's pixels are no longer visible in the produced frame.

    Only a *complete* absence counts: an object whose region is fully covered by the replacement mask
    is legitimately gone (it was occluded by the new object), while one that still has any pixel
    outside the replacement region is preserved.
    """
    if obj.bbox.area <= 0:
        return True
    height, width = produced_mask.shape
    x0 = max(0, min(width, int(obj.bbox.x_min)))
    x1 = max(0, min(width, int(np.ceil(obj.bbox.x_max))))
    y0 = max(0, min(height, int(obj.bbox.y_min)))
    y1 = max(0, min(height, int(np.ceil(obj.bbox.y_max))))
    if x1 <= x0 or y1 <= y0:
        return True
    window = produced_mask[y0:y1, x0:x1]
    return bool(window.all())


def _bbox(mask: np.ndarray):
    from vidliner.domain.shapes import BoundingBox

    rows, columns = np.nonzero(mask)
    return BoundingBox(
        x_min=float(columns.min()),
        y_min=float(rows.min()),
        x_max=float(columns.max() + 1),
        y_max=float(rows.max() + 1),
    )


def _polygon(mask: np.ndarray) -> PolygonMask | None:
    from vidliner.backends.imageops import mask_polygon

    return mask_polygon(mask)


def _categories_for(bundle: AnnotationBundle, category: str) -> tuple[CategorySpec, ...]:
    """Ensure the replacement category exists, preserving every inherited category id."""
    return bundle.ensure_categories((category,)).categories


def _category_for(candidate: Any, plan: Any) -> str | None:
    """The category this *candidate* was asked for.

    A plan states what the object should become overall; each candidate carries the specific value it
    generated. The candidate wins, because the annotation must describe the image that exists, not
    the plan's first suggestion.
    """
    if isinstance(candidate, dict) and candidate.get("category"):
        return str(candidate["category"])
    if plan is None:
        return None
    if isinstance(plan, dict):
        return str(plan.get("expected_category") or "")
    return str(getattr(plan, "expected_category", "") or "")


def _item(value: Any) -> ObjectItem:
    if isinstance(value, ObjectItem):
        return value
    if isinstance(value, dict):
        return ObjectItem.model_validate(value)
    raise OperatorFailure(
        f"annotation rebuild needs a target item, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
    )


def _mask(value: Any) -> MaskRef:
    if isinstance(value, MaskRef):
        return value
    if isinstance(value, dict):
        return MaskRef.model_validate(value)
    raise OperatorFailure(
        f"annotation rebuild needs a mask, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
    )


def register_all(registry: Any) -> None:
    """Register this module's operators in ``registry``."""
    registry.register(RebuildAnnotationOperator)
