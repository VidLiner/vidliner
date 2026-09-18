"""Measured facts about a target object and its surroundings.

``SceneContext`` is the input to replacement planning. Every field is *measured or explicitly
unknown* — never invented. Fields that a given backend cannot supply stay ``None``, and the planner
degrades to a more conservative plan rather than fabricating an orientation it did not measure.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidliner.domain.enums import SceneScaleClass
from vidliner.domain.masks import MaskRef
from vidliner.domain.shapes import BoundingBox, ImageShape, Point

__all__ = ["LightingEstimate", "OcclusionRelation", "SceneContext"]


class LightingEstimate(BaseModel):
    """Direction and strength of the dominant illumination on the target object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    direction_deg: float | None = Field(default=None, ge=0.0, lt=360.0)
    """Azimuth of the dominant light in image space; ``0`` is to the right, increasing clockwise."""
    intensity: float | None = Field(default=None, ge=0.0, le=1.0)
    """Normalised mean luminance of the object region."""
    contrast: float | None = Field(default=None, ge=0.0, le=1.0)
    shadow_region: BoundingBox | None = None
    color_temperature: float | None = None
    """Estimated correlated colour temperature in kelvin, when the backend can measure it."""
    measured_by: str | None = None

    @property
    def is_known(self) -> bool:
        """True when at least a direction was measured."""
        return self.direction_deg is not None


class OcclusionRelation(BaseModel):
    """Which object is in front of which, as far as the scene analysis can tell."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    other_object_id: str
    relation: str
    """One of ``in_front_of``, ``behind``, ``overlapping``, ``adjacent``."""
    overlap_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    depth_delta: float | None = None

    @model_validator(mode="after")
    def _check_relation(self) -> Self:
        allowed = {"in_front_of", "behind", "overlapping", "adjacent"}
        if self.relation not in allowed:
            raise ValueError(f"unsupported occlusion relation {self.relation!r}")
        return self


class SceneContext(BaseModel):
    """Everything the planner is allowed to reason about.

    Attributes:
        object_category: category of the *source* object, as measured.
        orientation_deg: in-plane rotation of the object's principal axis, when measured.
        pose: free-form pose label (``front``, ``three_quarter``, ...) when a backend supplies one.
        scale_class: coarse size bucket derived from coverage.
        lighting: illumination estimate.
        shadow: measured shadow region, which a strict replacement must preserve or reproduce.
        depth_order: relative depth of the target within the scene, ``0`` nearest.
        occlusions: relations to neighbouring objects.
        background_summary: short, non-generative description used in the generation prompt.
        ground_contact: pixel row where the object meets the ground plane, when detected.
        mask: the target mask this context was computed for.
        neighbours: other objects in the same frame that must survive the edit.
        measured_by: backend id that produced the analysis.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    object_category: str
    object_id: str
    image_shape: ImageShape
    bbox: BoundingBox
    mask: MaskRef
    orientation_deg: float | None = Field(default=None, ge=-180.0, le=180.0)
    pose: str | None = None
    scale_class: SceneScaleClass = SceneScaleClass.MEDIUM
    lighting: LightingEstimate = Field(default_factory=LightingEstimate)
    shadow: BoundingBox | None = None
    depth_order: int | None = Field(default=None, ge=0)
    occlusions: tuple[OcclusionRelation, ...] = ()
    background_summary: str = ""
    ground_contact: int | None = None
    neighbours: tuple[str, ...] = ()
    measured_by: str | None = None

    @model_validator(mode="after")
    def _check_geometry(self) -> Self:
        if self.ground_contact is not None and not 0 <= self.ground_contact <= self.image_shape.height:
            raise ValueError("ground contact row is outside the image")
        if self.mask.shape != self.image_shape:
            raise ValueError("scene context mask shape does not match the image shape")
        return self

    @property
    def coverage(self) -> float:
        """Fraction of the image occupied by the target object."""
        return self.mask.coverage

    @property
    def is_occluded(self) -> bool:
        """True when any measured relation places another object in front of the target."""
        return any(relation.relation in {"in_front_of", "overlapping"} for relation in self.occlusions)

    @property
    def centroid(self) -> Point:
        """Centroid of the target bounding box."""
        return self.bbox.centroid

    @property
    def is_near_frame_edge(self) -> bool:
        """True when the object touches the image border, which constrains allowed geometry change."""
        margin = 2.0
        return (
            self.bbox.x_min <= margin
            or self.bbox.y_min <= margin
            or self.bbox.x_max >= self.image_shape.width - margin
            or self.bbox.y_max >= self.image_shape.height - margin
        )
