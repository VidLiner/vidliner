"""Planning-only video operators.

The first video node is deliberately renderer-neutral. It turns the declarative video spec into
stable variant and contrast-pair records. Local ffmpeg and Hypit operators can consume those
records later without each inventing its own sampling or provenance rules.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from vidliner.domain.enums import Determinism, PortType, StageName
from vidliner.domain.video import VideoAugmentSpec, build_contrast_pairs, enumerate_video_variants
from vidliner.operators.base import ExecutionContext, Operator, OperatorSpec, OutputSpec

__all__ = ["PlanVideoVariantsOperator", "VideoPlanConfig", "register_all"]


class VideoPlanConfig(BaseModel):
    """Configuration for deterministic video variant enumeration."""

    model_config = ConfigDict(extra="forbid")

    spec: VideoAugmentSpec


class PlanVideoVariantsOperator(Operator):
    """Enumerate video variants and their one-axis contrast pairs."""

    @property
    def spec(self) -> OperatorSpec:
        """Declared contract of the video planning operator."""
        return OperatorSpec(
            name="video.plan_variants",
            version="1.0.0",
            stage=StageName.PLAN,
            summary="enumerate deterministic video variants and contrast pairs",
            inputs={},
            outputs={
                "variants": OutputSpec(PortType.VIDEO_VARIANTS, "planned video variants"),
                "pairs": OutputSpec(PortType.CONTRAST_PAIRS, "one-axis contrast relationships"),
            },
            config_model=VideoPlanConfig,
            capabilities=(),
            determinism=Determinism.DETERMINISTIC,
            timeout_s=30.0,
            max_parallelism=8,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Enumerate variants without reading media or calling a backend."""
        del inputs
        config = VideoPlanConfig.model_validate(context.config)
        variants = enumerate_video_variants(config.spec)
        pairs = build_contrast_pairs(variants)
        context.publish(
            "video.plan.completed",
            spec_id=config.spec.id,
            variants=len(variants),
            pairs=len(pairs),
        )
        return {"variants": variants, "pairs": pairs}


def register_all(registry: Any) -> None:
    """Register the video planning operator."""
    registry.register(PlanVideoVariantsOperator)
