"""Mask representations.

A mask appears in three shapes during the pipeline and each one has a different job:

* a **binary mask** is what segmentation produces and what compositing consumes;
* a **polygon** is what COCO and YOLO segmentation annotations need;
* a **mask reference** is how a mask is stored, content-addressed, and passed between nodes without
  ever loading pixel data into a model object.

The mask itself is never embedded in a domain object. That keeps every model small, cacheable, and
safe to serialize into a manifest.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidliner.core.identity import artifact_digest_key
from vidliner.core.results import ArtifactRef
from vidliner.domain.enums import ArtifactKind
from vidliner.domain.shapes import BoundingBox, ImageShape, Point

__all__ = ["BinaryMask", "DepthMap", "MaskEncoding", "MaskRef", "PolygonMask"]


class MaskEncoding(str):
    """Storage encoding of a mask artifact."""

    PNG_L8 = "png-l8"
    NUMPY_BOOL = "npy-bool"
    RUN_LENGTH = "rle"


class PolygonMask(BaseModel):
    """One or more closed rings in pixel space.

    Rings are stored as the outer boundary first; holes, when they exist, are subsequent rings. No
    winding rule is applied by the domain layer — exporters use the convention their format
    requires, and the COCO/YOLO converters document theirs explicitly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rings: list[list[Point]] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_rings(self) -> Self:
        for index, ring in enumerate(self.rings):
            if len(ring) < 3:
                raise ValueError(f"polygon ring {index} needs at least three points")
        return self

    @property
    def point_count(self) -> int:
        """Total number of vertices across every ring."""
        return sum(len(ring) for ring in self.rings)

    @property
    def bbox(self) -> BoundingBox:
        """Tight bounding box around every ring."""
        return BoundingBox.from_points([point for ring in self.rings for point in ring])

    def as_coco_flat(self, *, decimals: int = 2) -> list[float]:
        """Flatten all rings into the ``[x1, y1, x2, y2, ...]`` layout COCO uses."""
        flat: list[float] = []
        for ring in self.rings:
            for point in ring:
                flat.append(round(point.x, decimals))
                flat.append(round(point.y, decimals))
        return flat

    @classmethod
    def from_coco_flat(cls, flat: list[float]) -> PolygonMask:
        """Rebuild a single-ring polygon from a flat coordinate list."""
        if len(flat) < 6 or len(flat) % 2 != 0:
            raise ValueError("a COCO polygon needs an even number of coordinates and at least three points")
        ring = [Point(x=float(flat[index]), y=float(flat[index + 1])) for index in range(0, len(flat), 2)]
        return cls(rings=[ring])

    def normalized_rings(self, shape: ImageShape) -> list[list[tuple[float, float]]]:
        """Rings expressed as fractions of the image size, for YOLO export."""
        return [[(point.x / shape.width, point.y / shape.height) for point in ring] for ring in self.rings]

    def clipped(self, shape: ImageShape) -> PolygonMask:
        """Return a polygon whose points are clamped into the image rectangle."""
        return PolygonMask(rings=[[point.clamped(shape) for point in ring] for ring in self.rings])


class MaskRef(BaseModel):
    """A digest-addressed reference to a stored mask, plus the facts a consumer needs.

    ``area_px`` and ``coverage`` are recorded at creation time so that downstream operators (and the
    acceptance gate) can reject an empty or implausibly large mask without loading the pixels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact: ArtifactRef
    shape: ImageShape
    area_px: int = Field(ge=0)
    encoding: str = MaskEncoding.PNG_L8

    @model_validator(mode="after")
    def _check_artifact(self) -> Self:
        if self.artifact.kind is not ArtifactKind.MASK:
            raise ValueError(f"mask reference must point at a mask artifact, got {self.artifact.kind.value}")
        return self

    @property
    def digest(self) -> str:
        """The mask artifact digest."""
        return self.artifact.digest

    @property
    def catalogue_key(self) -> str:
        """Kind-namespaced catalogue key for the underlying artifact."""
        return artifact_digest_key(ArtifactKind.MASK, self.artifact.digest)

    @property
    def coverage(self) -> float:
        """Fraction of the image covered by the mask."""
        return self.area_px / self.shape.pixel_count if self.shape.pixel_count else 0.0

    @property
    def is_empty(self) -> bool:
        """True when the mask selects no pixels."""
        return self.area_px == 0


class BinaryMask(BaseModel):
    """A mask produced by segmentation, before it is stored.

    This is the only domain object that describes mask *content* rather than a reference to it, and
    it is deliberately short-lived: operators convert it to a :class:`MaskRef` immediately.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    shape: ImageShape
    area_px: int = Field(ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    polygon: PolygonMask | None = None
    source_prompt: str | None = None
    """How the mask was requested: ``bbox``, ``point``, ``text``, or ``auto``."""

    @model_validator(mode="after")
    def _check_area(self) -> Self:
        if self.area_px > self.shape.pixel_count:
            raise ValueError("mask area cannot exceed the image area")
        return self

    @property
    def coverage(self) -> float:
        """Fraction of the image covered by the mask."""
        return self.area_px / self.shape.pixel_count if self.shape.pixel_count else 0.0

    @property
    def is_empty(self) -> bool:
        """True when the mask selects no pixels."""
        return self.area_px == 0


class DepthMap(BaseModel):
    """A digest-addressed depth estimate for one frame. Reserved for the video/depth phase."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact: ArtifactRef
    shape: ImageShape
    min_depth: float
    max_depth: float
    metric: str = "relative"

    @model_validator(mode="after")
    def _check_artifact(self) -> Self:
        if self.artifact.kind is not ArtifactKind.DEPTH:
            raise ValueError("depth map must point at a depth artifact")
        if self.max_depth < self.min_depth:
            raise ValueError("depth range is inverted")
        return self

    def normalized(self, value: float) -> float:
        """Map a raw depth value into ``[0, 1]`` over this map's range."""
        span = self.max_depth - self.min_depth
        if span <= 0:
            return 0.0
        return min(max((value - self.min_depth) / span, 0.0), 1.0)
