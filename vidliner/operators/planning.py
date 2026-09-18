"""Replacement planning: turn an intent and a measured scene into an executable plan.

Planning is free by construction. The operator calls a *planner* backend, and a planner backend is
required to make no generative call — the default one is entirely rule-based. That is what makes
``vidliner plan`` a cost-free operation and what makes ``--dry-run`` trustworthy: it resolves
capabilities and counts nodes without ever asking a generative service for anything.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from vidliner.capabilities.backend import PlanningRequest
from vidliner.capabilities.names import CAP_PLANNING
from vidliner.core.errors import ErrorCode, OperatorFailure
from vidliner.domain.enums import Determinism, PortType, ReplacementStrategy, StageName
from vidliner.domain.instances import ObjectItem
from vidliner.domain.replacement import ReplacementIntent
from vidliner.domain.scene import SceneContext
from vidliner.operators.backend_context import backend_context
from vidliner.operators.base import ExecutionContext, InputSpec, Operator, OperatorSpec, OutputSpec
from vidliner.operators.shared import backend_handle

__all__ = ["PlanConfig", "PlanReplacementOperator", "register_all"]


class PlanConfig(BaseModel):
    """Configuration of the planning operator."""

    model_config = ConfigDict(extra="forbid")

    values: tuple[str, ...] = ()
    description: str = ""
    references: tuple[str, ...] = ()
    candidates_per_object: int = 3
    mode: str = "strict"
    strategy: str = "category"
    preserve: dict[str, bool] = {}
    allow_geometry_change: bool = False
    mask_expansion_px: int = 0
    prompt_template: str
    negative_constraints: tuple[str, ...] = ()


class PlanReplacementOperator(Operator):
    """Build the replacement plan for one target object."""

    @property
    def spec(self) -> OperatorSpec:
        """Declared contract of the planning operator."""
        return OperatorSpec(
            name="plan.replacement",
            version="1.0.0",
            stage=StageName.PLAN,
            summary="decide what changes and what must remain for one target object",
            inputs={
                "item": InputSpec(PortType.INSTANCES, "the segmented target object"),
                "scene": InputSpec(PortType.SCENE_CONTEXT, "measured scene context"),
            },
            outputs={"plan": OutputSpec(PortType.PLAN, "the replacement plan")},
            config_model=PlanConfig,
            capabilities=(CAP_PLANNING,),
            determinism=Determinism.DETERMINISTIC,
            timeout_s=60.0,
            max_parallelism=4,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Plan the replacement and return it."""
        item = _item(inputs.get("item"))
        scene = _scene(inputs.get("scene"))
        config = PlanConfig.model_validate(context.config)
        intent = _intent(item, config, seed=context.seed)
        handle = await backend_handle(context, CAP_PLANNING)
        request = PlanningRequest(
            sample_id=item.sample_id,
            intent=intent,
            scene=scene,
            candidates_per_object=config.candidates_per_object,
            prompt_template=config.prompt_template,
            negative_constraints=tuple(config.negative_constraints),
            mask_expansion_px=config.mask_expansion_px,
            candidate_values=tuple(config.values),
        )
        plan = await handle.call("plan", request, backend_context(context))
        context.publish(
            "plan.completed",
            backend=handle.backend_id,
            object_id=item.object_id,
            candidates=plan.candidate_count,
            expected_category=plan.expected_category,
        )
        return {"plan": plan}


def _intent(item: ObjectItem, config: PlanConfig, *, seed: int) -> ReplacementIntent:
    """Build the intent the planner sees, honouring the recipe's strategy."""
    strategy = ReplacementStrategy(config.strategy)
    preserve = dict(config.preserve)
    replacement_category = config.values[0] if config.values else item.class_name
    if strategy is ReplacementStrategy.REFERENCE and not config.references:
        raise OperatorFailure(
            "replacement strategy 'reference' requires at least one reference image",
            code=ErrorCode.OPERATOR_CONFIG_INVALID,
        )
    return ReplacementIntent(
        target_object=item.object_id,
        replacement_category=replacement_category,
        replacement_description=config.description,
        strategy=strategy,
        mode=config.mode,
        preserve_pose=preserve.get("pose", True),
        preserve_scale=preserve.get("scale", True),
        preserve_position=preserve.get("position", True),
        preserve_lighting=preserve.get("lighting", True),
        preserve_occlusion=preserve.get("occlusion", True),
        allowed_geometry_change=config.allow_geometry_change,
        seed=seed,
        negatives=tuple(config.negative_constraints),
    )


def _item(value: Any) -> ObjectItem:
    if isinstance(value, ObjectItem):
        return value
    if isinstance(value, dict):
        return ObjectItem.model_validate(value)
    raise OperatorFailure(
        f"planning needs a target item, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
    )


def _scene(value: Any) -> SceneContext:
    if isinstance(value, SceneContext):
        return value
    if isinstance(value, dict):
        return SceneContext.model_validate(value)
    raise OperatorFailure(
        f"planning needs a scene context, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
    )


def register_all(registry: Any) -> None:
    """Register this module's operators in ``registry``."""
    registry.register(PlanReplacementOperator)
