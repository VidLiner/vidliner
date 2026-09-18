"""Candidate generation: one node per candidate.

Splitting generation into one node per candidate is what gives the pipeline its most useful
properties: candidates are evaluated independently, a failure in one does not destroy the others,
each carries its own seed and its own cost, and the node cache can reuse exactly the candidates that
were already produced.

The node's *seed* comes from the graph identity (``sample → object → candidate``), and the plan's
candidate spec carries the same derived seed, so the two always agree and both are recorded in the
manifest.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from vidliner.capabilities.backend import GenerationRequest
from vidliner.capabilities.names import CAP_OBJECT_REPLACEMENT
from vidliner.core.errors import ErrorCode, OperatorFailure
from vidliner.core.results import ArtifactRef
from vidliner.domain.enums import Determinism, PortType, StageName
from vidliner.domain.instances import ObjectItem
from vidliner.domain.replacement import (
    CandidateSpec,
    ReplacementPlan,
)
from vidliner.operators.backend_context import backend_context
from vidliner.operators.base import ExecutionContext, InputSpec, Operator, OperatorSpec, OutputSpec
from vidliner.operators.shared import backend_handle

__all__ = ["GenerateConfig", "GenerateReplacementOperator", "register_all"]


class GenerateConfig(BaseModel):
    """Configuration of the generation operator."""

    model_config = ConfigDict(extra="forbid")

    candidate_key: str
    category: str
    description: str = ""
    ordinal: int = 0
    intensity: float = 1.0
    parameters: dict[str, Any] = {}
    context_padding_px: int = 32


class GenerateReplacementOperator(Operator):
    """Generate one replacement candidate for one target object."""

    @property
    def spec(self) -> OperatorSpec:
        """Declared contract of the generation operator."""
        return OperatorSpec(
            name="generate.replacement",
            version="1.0.0",
            stage=StageName.GENERATE,
            summary="produce one replacement candidate image for one target object",
            inputs={
                "item": InputSpec(PortType.INSTANCES, "the segmented target object"),
                "plan": InputSpec(PortType.PLAN, "the replacement plan for this object"),
            },
            outputs={"candidate": OutputSpec(PortType.CANDIDATE_REF, "one generated candidate")},
            config_model=GenerateConfig,
            capabilities=(CAP_OBJECT_REPLACEMENT,),
            determinism=Determinism.SEEDED,
            # Generative operations are not retried unless the backend declares itself idempotent:
            # a retry may produce a different image and consume another paid call.
            retry_attempts=1,
            timeout_s=600.0,
            max_parallelism=1,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Generate the candidate described by the node's configuration."""
        item = _item(inputs.get("item"))
        plan = _plan(inputs.get("plan"))
        config = GenerateConfig.model_validate(context.config)
        spec = _spec_for(plan, config)
        image = _image(inputs, context)
        mask_ref = item.instance.mask_ref
        if mask_ref is None:
            raise OperatorFailure(
                f"generation requires a mask for object {item.object_id}",
                code=ErrorCode.MASK_EMPTY,
                detail={"object_id": item.object_id},
            )
        request = GenerationRequest(
            sample_id=item.sample_id,
            target_object=item.object_id,
            candidate_key=spec.candidate_key,
            source_image=image,
            target_mask=mask_ref.artifact,
            mask_shape=mask_ref.shape,
            target_bbox=item.instance.bbox,
            context_crop=None,
            positive_prompt=_prompt_for(plan, spec),
            negative_constraints=tuple(plan.negative_constraints),
            expected_category=spec.category,
            replacement_description=spec.description,
            seed=spec.seed,
            intensity=config.intensity,
            reference_image=spec.reference,
            parameters={**spec.parameters, **config.parameters},
        )
        handle = await backend_handle(context, CAP_OBJECT_REPLACEMENT)
        outcome = await handle.call("replace", request, backend_context(context))
        context.publish(
            "generate.completed",
            backend=handle.backend_id,
            object_id=item.object_id,
            candidate_key=spec.candidate_key,
            seed=spec.seed,
            artifact=outcome.image.digest[:12],
            external=outcome.external,
        )
        payload = {
            "candidate_key": spec.candidate_key,
            "category": spec.category,
            "description": spec.description,
            "seed": spec.seed,
            "object_id": item.object_id,
            "sample_id": item.sample_id,
            "intent": plan.intent.model_dump(mode="json"),
            "plan_id": plan.plan_id,
            "generation": outcome.model_dump(mode="json"),
        }
        return {"candidate": payload}


def _spec_for(plan: ReplacementPlan, config: GenerateConfig) -> CandidateSpec:
    """Find the plan's candidate spec that matches this node, or synthesise a consistent one.

    Matching on the candidate key (not on position) keeps node identity stable when a recipe adds a
    candidate value: existing candidates keep their key, their seed, and their cache entry.
    """
    for spec in plan.candidate_specs:
        if spec.candidate_key == config.candidate_key:
            return spec
    return CandidateSpec(
        candidate_key=config.candidate_key,
        category=config.category,
        description=config.description,
        seed=plan.plan_seed,
        ordinal=config.ordinal,
    )


def _prompt_for(plan: ReplacementPlan, spec: CandidateSpec) -> str:
    """Render the plan's prompt for one candidate's description."""
    if spec.description and spec.description != plan.expected_category:
        return f"{plan.positive_prompt} Target appearance: {spec.description}."
    return plan.positive_prompt


def _image(inputs: dict[str, Any], context: ExecutionContext) -> ArtifactRef:
    """The source frame the generator must edit."""
    for value in inputs.values():
        if isinstance(value, ArtifactRef) and value.media_type.startswith("image/"):
            return value
        if isinstance(value, dict) and value.get("media_type", "").startswith("image/"):
            return ArtifactRef.model_validate(value)
    sample = context.sample
    artifact = getattr(sample, "image_artifact", None) if sample is not None else None
    if isinstance(artifact, ArtifactRef):
        return artifact
    if isinstance(artifact, dict):
        return ArtifactRef.model_validate(artifact)
    raise OperatorFailure(
        "generation could not determine the source image artifact",
        code=ErrorCode.PORT_UNBOUND,
        detail={"node_id": context.node.node_id},
    )


def _item(value: Any) -> ObjectItem:
    if isinstance(value, ObjectItem):
        return value
    if isinstance(value, dict):
        return ObjectItem.model_validate(value)
    raise OperatorFailure(
        f"generation needs a target item, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
    )


def _plan(value: Any) -> ReplacementPlan:
    if isinstance(value, ReplacementPlan):
        return value
    if isinstance(value, dict):
        return ReplacementPlan.model_validate(value)
    raise OperatorFailure(
        f"generation needs a replacement plan, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
    )


def register_all(registry: Any) -> None:
    """Register this module's operators in ``registry``."""
    registry.register(GenerateReplacementOperator)
