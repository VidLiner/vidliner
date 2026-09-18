"""Target selection: decide which detected objects the recipe will replace.

Selection is a pure, deterministic filter. It never invents an object and never re-orders by
anything unstable: ties are broken by object id, so the same inputs always select the same targets
and therefore produce the same node identities on a re-run.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vidliner.domain.enums import Determinism, PortType, StageName
from vidliner.domain.instances import ObjectInstance, ObjectItem, ObjectSelection
from vidliner.operators.base import ExecutionContext, InputSpec, Operator, OperatorSpec, OutputSpec
from vidliner.operators.shared import require_instances

__all__ = ["SelectConfig", "SelectTargetsOperator", "register_all"]


class SelectConfig(BaseModel):
    """Configuration of the target-selection operator."""

    model_config = ConfigDict(extra="forbid")

    classes: tuple[str, ...] = Field(min_length=1)
    min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    min_area_px: int = Field(default=1024, ge=0)
    max_per_sample: int = Field(default=2, ge=1, le=64)
    top_k_by: str = "score"
    strategy: str = "all"
    object_ids: tuple[str, ...] = ()

    def ordered(self, candidates: list[ObjectInstance]) -> list[ObjectInstance]:
        """Order candidates according to ``top_k_by``, breaking ties by object id."""
        if self.top_k_by == "area":
            return sorted(candidates, key=lambda item: (-(item.area_px or 0), item.object_id))
        return sorted(candidates, key=lambda item: (-item.score, item.object_id))


class SelectTargetsOperator(Operator):
    """Filter and rank detected objects into the set the recipe will act on."""

    @property
    def spec(self) -> OperatorSpec:
        """Declared contract of the selection operator."""
        return OperatorSpec(
            name="select.targets",
            version="1.0.0",
            stage=StageName.SELECT,
            summary="choose the target objects the recipe describes",
            inputs={"instances": InputSpec(PortType.INSTANCES, "detected objects")},
            outputs={
                "selection": OutputSpec(PortType.INSTANCES, "selection record with drop counts"),
                "items": OutputSpec(PortType.INSTANCES, "one item per selected target object"),
            },
            config_model=SelectConfig,
            capabilities=(),
            determinism=Determinism.DETERMINISTIC,
            max_parallelism=8,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Select targets and record why everything else was dropped."""
        config = SelectConfig.model_validate(dict(context.config))
        instances = require_instances(inputs.get("instances"), where="select.targets")
        wanted = set(config.classes)
        dropped_class = dropped_score = dropped_area = 0
        eligible: list[ObjectInstance] = []
        for instance in instances:
            if instance.class_name not in wanted:
                dropped_class += 1
                continue
            if instance.score < config.min_score:
                dropped_score += 1
                continue
            area = instance.area_px or int(instance.bbox.area)
            if area < config.min_area_px:
                dropped_area += 1
                continue
            eligible.append(instance)

        if config.strategy == "explicit":
            allowed = set(config.object_ids)
            eligible = [instance for instance in eligible if instance.object_id in allowed]

        ordered = config.ordered(eligible)
        if config.strategy == "largest":
            ordered = sorted(eligible, key=lambda item: (-(item.area_px or 0), item.object_id))
        elif config.strategy == "first":
            ordered = sorted(eligible, key=lambda item: item.object_id)
        dropped_cap = max(0, len(ordered) - config.max_per_sample)
        selected = tuple(ordered[: config.max_per_sample])
        sample = context.sample
        selection = ObjectSelection(
            sample_id=sample.sample_id if sample else "",
            selected=selected,
            considered=len(instances),
            dropped_below_score=dropped_score,
            dropped_below_area=dropped_area,
            dropped_by_class=dropped_class,
            dropped_by_cap=dropped_cap,
            classes_requested=config.classes,
        )
        items = tuple(
            ObjectItem(
                instance=instance,
                sample_id=selection.sample_id,
                selection_key=f"object:{instance.object_id}",
                classes_requested=config.classes,
                neighbours=tuple(item for item in instances if item.object_id != instance.object_id),
                ordinal=index,
            )
            for index, instance in enumerate(selected)
        )
        return {"selection": selection, "items": items}


def register_all(registry: Any) -> None:
    """Register this module's operators in ``registry``."""
    registry.register(SelectTargetsOperator)
