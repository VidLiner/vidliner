"""Provider-neutral asynchronous video generation contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, JsonValue, model_validator

from vidliner.core.results import ArtifactRef
from vidliner.domain.video import LabelTransform


class VideoGenerationRequest(BaseModel):
    """One generation intent; credentials and routing belong to the runtime profile."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, hide_input_in_errors=True)

    prompt: str = Field(min_length=1)
    mode: Literal["text-to-video", "image-to-video", "video-to-video"] = "text-to-video"
    image: ArtifactRef | None = None
    image_url: HttpUrl | None = None
    video_url: HttpUrl | None = None
    seed: int | None = Field(default=None, ge=0, le=4294967295)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _media(self) -> Self:
        if self.image is not None and self.image_url is not None:
            raise ValueError("supply an image artifact or URL, not both")
        if self.mode == "image-to-video" and self.image is None and self.image_url is None:
            raise ValueError("image-to-video requires an image")
        if self.mode == "video-to-video" and self.video_url is None:
            raise ValueError("video-to-video requires a video URL")
        if self.mode == "text-to-video" and any(
            x is not None for x in (self.image, self.image_url, self.video_url)
        ):
            raise ValueError("text-to-video does not accept source media")
        if self.mode == "image-to-video" and self.video_url is not None:
            raise ValueError("image-to-video does not accept a video URL")
        if self.mode == "video-to-video" and (self.image is not None or self.image_url is not None):
            raise ValueError("video-to-video does not accept a source image")
        return self


class VideoTask(BaseModel):
    """Durable task handle. Submission success is not generation or dataset acceptance."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    schema_version: Literal["vidliner.video-task/v1"] = "vidliner.video-task/v1"
    backend_id: str
    model_id: str
    task_id: str = Field(min_length=1)
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "unknown"] = "queued"
    status_url: HttpUrl | None = None
    result_url: HttpUrl | None = None
    cancel_url: HttpUrl | None = None
    output_urls: tuple[HttpUrl, ...] = ()
    failure_code: str | None = None
    label_transform: LabelTransform = Field(default_factory=lambda: LabelTransform(semantics="synthesized"))
    verified: Literal[False] = False

    @property
    def terminal(self) -> bool:
        return self.status in {"succeeded", "failed", "cancelled"}
