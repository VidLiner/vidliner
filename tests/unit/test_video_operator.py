"""The video planning node is deterministic and backend-free."""

from __future__ import annotations

import asyncio

from vidliner.core.graph import OperationNode
from vidliner.domain.enums import Determinism, PortType, StageName
from vidliner.domain.video import VideoAugmentSpec, VideoOperatorSpec
from vidliner.operators.base import ExecutionContext
from vidliner.operators.video import PlanVideoVariantsOperator


def _context(spec: VideoAugmentSpec) -> ExecutionContext:
    node = OperationNode(
        node_id="video-plan",
        operator="video.plan_variants",
        operator_version="1.0.0",
        stage=StageName.PLAN,
        outputs={"variants": PortType.VIDEO_VARIANTS, "pairs": PortType.CONTRAST_PAIRS},
        config={"spec": spec.model_dump(mode="json")},
        determinism=Determinism.DETERMINISTIC,
    )
    return ExecutionContext(job_id="job", node=node, seed=spec.seed, io=None)  # type: ignore[arg-type]


def test_video_plan_operator_returns_stable_variants_and_pairs() -> None:
    spec = VideoAugmentSpec(
        id="street",
        seed=11,
        strategy="paired",
        operators=(VideoOperatorSpec(name="format.reframe", axis="format", domains={"aspect": ("1:1",)}),),
    )
    result = asyncio.run(PlanVideoVariantsOperator().run({}, _context(spec)))
    assert len(result["variants"]) == 2
    assert len(result["pairs"]) == 1
    assert result["variants"][0].ops == ()
