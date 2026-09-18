"""Cross-frame object association.

Tracking is a Phase 2 capability, but the model exists now because it is what makes the video
extension an *extension* rather than a rewrite: the segmentation, quality, and annotation stages
already operate on ``ObjectInstance``, and a track is simply an ordered set of instances that share
an identity.

The requirements this model has to be able to express:

* partial occlusion — an observation whose mask is incomplete but whose visibility is above zero;
* temporary disappearance — a gap in frame indices with no observations;
* re-entry — more than one gap, counted in ``reentry_count``.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidliner.domain.masks import MaskRef
from vidliner.domain.shapes import BoundingBox

__all__ = ["ObjectTrack", "TrackObservation"]


class TrackObservation(BaseModel):
    """One frame's evidence for a tracked object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_index: int = Field(ge=0)
    bbox: BoundingBox
    mask_ref: MaskRef | None = None
    visibility: float = Field(default=1.0, ge=0.0, le=1.0)
    """1.0 means fully visible; lower values express partial occlusion."""
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    interpolated: bool = False
    """True when the observation was filled in from neighbouring frames rather than measured."""

    @property
    def is_occluded(self) -> bool:
        """True when the object is partially or fully hidden in this frame."""
        return self.visibility < 0.999


class ObjectTrack(BaseModel):
    """An object followed across frames of one media asset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    track_id: str
    asset_digest: str
    class_name: str
    observations: tuple[TrackObservation, ...] = ()
    visibility: float = Field(default=1.0, ge=0.0, le=1.0)
    """Mean visibility over the track's observations."""
    reentry_count: int = Field(default=0, ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    backend_id: str | None = None

    @model_validator(mode="after")
    def _check_observations(self) -> Self:
        indices = [observation.frame_index for observation in self.observations]
        if indices != sorted(indices):
            raise ValueError(f"track {self.track_id} observations are not ordered by frame index")
        if len(set(indices)) != len(indices):
            raise ValueError(f"track {self.track_id} has two observations for the same frame")
        return self

    @property
    def first_frame(self) -> int:
        """First frame the track was observed in, or ``-1`` for an empty track."""
        return self.observations[0].frame_index if self.observations else -1

    @property
    def last_frame(self) -> int:
        """Last frame the track was observed in, or ``-1`` for an empty track."""
        return self.observations[-1].frame_index if self.observations else -1

    @property
    def length(self) -> int:
        """Number of frames the track was observed in."""
        return len(self.observations)

    @property
    def span(self) -> int:
        """Number of frames between first and last observation, inclusive."""
        if not self.observations:
            return 0
        return self.last_frame - self.first_frame + 1

    @property
    def gaps(self) -> tuple[tuple[int, int], ...]:
        """Frame ranges with no observation, excluding the leading and trailing edges."""
        gaps: list[tuple[int, int]] = []
        for previous, current in zip(self.observations, self.observations[1:], strict=True):
            if current.frame_index - previous.frame_index > 1:
                gaps.append((previous.frame_index + 1, current.frame_index - 1))
        return tuple(gaps)

    @property
    def continuity(self) -> float:
        """Observed frames divided by span: ``1.0`` for a gapless track."""
        return self.length / self.span if self.span else 0.0

    def observation_at(self, frame_index: int) -> TrackObservation | None:
        """Return the observation for one frame, or ``None`` when the track has a gap there."""
        for observation in self.observations:
            if observation.frame_index == frame_index:
                return observation
        return None
