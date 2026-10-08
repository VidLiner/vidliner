"""Video task lifecycle nodes using the same capability broker as image generation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from vidliner.capabilities.names import CAP_VIDEO_GENERATION
from vidliner.domain.enums import Determinism, PortType, StageName
from vidliner.domain.video_generation import VideoGenerationRequest, VideoTask
from vidliner.operators.backend_context import backend_context
from vidliner.operators.base import ExecutionContext, InputSpec, Operator, OperatorSpec, OutputSpec
from vidliner.operators.shared import backend_handle


class VideoSubmitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: VideoGenerationRequest


class VideoTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SubmitVideoOperator(Operator):
    """Submit one generation request through the runtime's video capability."""

    @property
    def spec(self) -> OperatorSpec:
        """Declare a non-cacheable, single-attempt asynchronous task output."""
        return OperatorSpec(
            name="video.submit",
            version="1.0.0",
            stage=StageName.GENERATE,
            summary="submit one video task without waiting or resubmitting",
            inputs={},
            outputs={"task": OutputSpec(PortType.VIDEO_TASK, "durable provider task handle")},
            config_model=VideoSubmitConfig,
            capabilities=(CAP_VIDEO_GENERATION,),
            determinism=Determinism.NONDETERMINISTIC,
            cacheable=False,
            retry_attempts=1,
            timeout_s=90,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Persist the returned handle as a typed graph value."""
        del inputs
        context.check_cancelled()
        request = VideoSubmitConfig.model_validate(context.config).request
        handle = await backend_handle(context, CAP_VIDEO_GENERATION)
        task = await handle.call("submit", request, backend_context(context))
        task = VideoTask.model_validate(task)
        context.publish(
            "video.submitted", task_id=task.task_id, backend_id=task.backend_id, status=task.status
        )
        return {"task": task}


class VideoStatusOperator(Operator):
    """Query an existing handle once without submitting a new generation."""

    method = "status"

    @property
    def spec(self) -> OperatorSpec:
        """Declare the shared typed task input and output lifecycle contract."""
        return OperatorSpec(
            name=f"video.{self.method}",
            version="1.0.0",
            stage=StageName.GENERATE,
            summary=f"{self.method} an existing video task without creating a new generation",
            inputs={"task": InputSpec(PortType.VIDEO_TASK, "existing provider task handle")},
            outputs={"task": OutputSpec(PortType.VIDEO_TASK, "current provider task state")},
            config_model=VideoTaskConfig,
            capabilities=(CAP_VIDEO_GENERATION,),
            determinism=Determinism.NONDETERMINISTIC,
            cacheable=False,
            retry_attempts=1,
            timeout_s=150,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Invoke the configured lifecycle method through the capacity limiter."""
        context.check_cancelled()
        task = VideoTask.model_validate(inputs["task"])
        handle = await backend_handle(context, CAP_VIDEO_GENERATION)
        result = await handle.call(self.method, task, backend_context(context))
        return {"task": VideoTask.model_validate(result)}


class CancelVideoOperator(VideoStatusOperator):
    """Explicitly request provider cancellation through the shared task ports."""

    method = "cancel"


def register_all(registry: Any) -> None:
    """Add every video lifecycle node to the existing operator catalogue."""
    registry.register(SubmitVideoOperator)
    registry.register(VideoStatusOperator)
    registry.register(CancelVideoOperator)
