"""Refinement: mask cleanup, alpha blending, and harmonisation, as one auditable chain.

Every refinement is a separate backend call with a declared capability, so each one can be replaced,
switched off, or measured independently. The chain is deliberately *not* folded into the generator:
a better harmoniser should not require regenerating anything, and the effect of each step should be
visible in the run's evidence.

The chain enforces one invariant unconditionally: **nothing outside the (feathered) mask may
change.** The compositor restores any pixel it touched outside the mask, because a blend that leaks
into the background is exactly the failure the background gate would otherwise have to catch.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vidliner.capabilities.backend import (
    CompositingRequest,
    HarmonizationRequest,
    MaskRefinementRequest,
)
from vidliner.capabilities.names import (
    CAP_IMAGE_COMPOSITING,
    CAP_IMAGE_HARMONIZATION,
    CAP_MASK_REFINEMENT,
)
from vidliner.core.errors import ErrorCode, OperatorFailure
from vidliner.core.results import ArtifactRef
from vidliner.domain.enums import Determinism, PortType, StageName
from vidliner.domain.masks import MaskRef
from vidliner.operators.backend_context import backend_context
from vidliner.operators.base import ExecutionContext, InputSpec, Operator, OperatorSpec, OutputSpec
from vidliner.operators.shared import backend_handle

__all__ = ["RefineCandidateOperator", "RefineConfig", "register_all"]


class RefineConfig(BaseModel):
    """Configuration of the refinement chain."""

    model_config = ConfigDict(extra="forbid")

    feather_px: int = Field(default=3, ge=0, le=64)
    close_px: int = Field(default=5, ge=0, le=64)
    dilate_px: int = Field(default=0, ge=0, le=64)
    harmonize: bool = True
    harmonize_strength: float = Field(default=0.45, ge=0.0, le=1.0)
    composite: bool = True


class RefineCandidateOperator(Operator):
    """Clean the mask, blend the generated region, and harmonise it."""

    @property
    def spec(self) -> OperatorSpec:
        """Declared contract of the refinement operator."""
        return OperatorSpec(
            name="refine.candidate",
            version="1.0.0",
            stage=StageName.REFINE,
            summary="clean the mask, composite the generated region, and match colour and lighting",
            inputs={
                "item": InputSpec(PortType.INSTANCES, "the target object with its mask"),
                "candidate": InputSpec(PortType.CANDIDATE_REF, "the generated candidate"),
                "mask": InputSpec(PortType.MASK_REF, "the mask produced by segmentation", required=False),
            },
            outputs={
                "image": OutputSpec(PortType.IMAGE_REF, "the refined image"),
                "mask": OutputSpec(PortType.MASK_REF, "the refined mask"),
                "report": OutputSpec(PortType.ANY, "what refinement did, for the audit trail"),
            },
            config_model=RefineConfig,
            capabilities=(CAP_MASK_REFINEMENT, CAP_IMAGE_COMPOSITING, CAP_IMAGE_HARMONIZATION),
            determinism=Determinism.DETERMINISTIC,
            timeout_s=300.0,
            max_parallelism=2,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Run the refinement chain and return the refined image, mask, and a report."""
        item = _item(inputs.get("item"))
        candidate = _candidate_payload(inputs.get("candidate"))
        config = RefineConfig.model_validate(context.config)
        source_image = _artifact(inputs.get("source_image"), port="source_image")
        generated_image = _generated_artifact(candidate, inputs)
        mask_ref = _mask(inputs.get("mask")) or item.instance.mask_ref
        if mask_ref is None:
            raise OperatorFailure(
                f"refinement requires a mask for object {item.object_id}",
                code=ErrorCode.MASK_EMPTY,
                detail={"object_id": item.object_id},
            )
        shape = mask_ref.shape
        report: dict[str, Any] = {"mask_area_before_px": mask_ref.area_px, "steps": []}

        refined_mask = mask_ref
        if config.feather_px or config.close_px or config.dilate_px:
            handle = await backend_handle(context, CAP_MASK_REFINEMENT)
            result = await handle.call(
                "refine_mask",
                MaskRefinementRequest(
                    mask=refined_mask.artifact,
                    shape=shape,
                    feather_px=config.feather_px,
                    close_px=config.close_px,
                    dilate_px=config.dilate_px,
                ),
                backend_context(context),
            )
            refined_mask = MaskRef(artifact=result.mask, shape=shape, area_px=result.area_px)
            report["steps"].append(
                {
                    "operator": "refine.mask_edges",
                    "backend": handle.backend_id,
                    "area_px": result.area_px,
                    "area_delta_px": result.area_delta_px,
                }
            )

        produced = generated_image
        if config.composite:
            handle = await backend_handle(context, CAP_IMAGE_COMPOSITING)
            result = await handle.call(
                "composite",
                CompositingRequest(
                    source_image=source_image,
                    candidate_image=generated_image,
                    mask=refined_mask.artifact,
                    shape=shape,
                    feather_px=config.feather_px,
                    preserve_outside_mask=True,
                ),
                backend_context(context),
            )
            produced = result.image
            report["steps"].append(
                {
                    "operator": "refine.alpha_blend",
                    "backend": handle.backend_id,
                    "changed_ratio_outside_mask": result.changed_ratio_outside_mask,
                }
            )
        if config.harmonize:
            handle = await backend_handle(context, CAP_IMAGE_HARMONIZATION)
            produced = await handle.call(
                "harmonize",
                HarmonizationRequest(
                    image=produced,
                    mask=refined_mask.artifact,
                    shape=shape,
                    strength=config.harmonize_strength,
                ),
                backend_context(context),
            )
            report["steps"].append(
                {
                    "operator": "refine.harmonize",
                    "backend": handle.backend_id,
                    "strength": config.harmonize_strength,
                }
            )
        report["mask_area_after_px"] = refined_mask.area_px
        report["output_digest"] = produced.digest
        context.publish(
            "refine.completed",
            object_id=item.object_id,
            steps=len(report["steps"]),
            output=produced.digest[:12],
        )
        return {"image": produced, "mask": refined_mask, "report": report}


def _generated_artifact(candidate: dict[str, Any], inputs: dict[str, Any]) -> ArtifactRef:
    """The generated image artifact, from the candidate payload or a bound port."""
    generation = candidate.get("generation")
    if isinstance(generation, dict) and generation.get("image"):
        return ArtifactRef.model_validate(generation["image"])
    for port in ("candidate_image", "generated"):
        value = inputs.get(port)
        if value is not None:
            return _artifact(value, port=port)
    raise OperatorFailure(
        "refinement could not find the generated image artifact",
        code=ErrorCode.PORT_UNBOUND,
        detail={"candidate_key": candidate.get("candidate_key")},
    )


def _candidate_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and "generation" in value:
        return value
    raise OperatorFailure(
        f"refinement needs a generation payload, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
    )


def _mask(value: Any) -> MaskRef | None:
    if value is None:
        return None
    if isinstance(value, MaskRef):
        return value
    if isinstance(value, dict):
        return MaskRef.model_validate(value)
    return None


def _item(value: Any) -> Any:
    from vidliner.domain.instances import ObjectItem

    if isinstance(value, ObjectItem):
        return value
    if isinstance(value, dict):
        return ObjectItem.model_validate(value)
    raise OperatorFailure(
        f"refinement needs a target item, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
    )


def _artifact(value: Any, *, port: str) -> ArtifactRef:
    if isinstance(value, ArtifactRef):
        return value
    if isinstance(value, dict):
        return ArtifactRef.model_validate(value)
    raise OperatorFailure(
        f"refinement needs an artifact on port {port!r}, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
        detail={"port": port},
    )


def register_all(registry: Any) -> None:
    """Register this module's operators in ``registry``."""
    registry.register(RefineCandidateOperator)
