"""Pixel-space geometry value objects.

Coordinates are pixel indices in the source image, with ``(0, 0)`` at the top-left corner.
Boxes are half-open in the sense that ``x_max``/``y_max`` may equal the image width/height, which
matches how detection and segmentation frameworks report boxes.
"""

from __future__ import annotations

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["BoundingBox", "ImageShape", "Point"]


class Point(BaseModel):
    """A single point in pixel space."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        """Euclidean distance between two points."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def clamped(self, shape: ImageShape) -> Point:
        """Return this point clamped into the bounds of ``shape``."""
        return Point(
            x=min(max(self.x, 0.0), float(shape.width)), y=min(max(self.y, 0.0), float(shape.height))
        )


class ImageShape(BaseModel):
    """Width/height (and optionally channel count) of an image."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    channels: int | None = Field(default=None, gt=0)

    @property
    def pixel_count(self) -> int:
        """Total number of pixels in the image."""
        return self.width * self.height

    @property
    def diagonal(self) -> float:
        """Length of the image diagonal, used to normalise distances."""
        return math.hypot(self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height."""
        return self.width / self.height


class BoundingBox(BaseModel):
    """An axis-aligned box in pixel space."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @model_validator(mode="after")
    def _check_ordering(self) -> Self:
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("bounding box maximum must not be smaller than its minimum")
        return self

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float) -> BoundingBox:
        """Build a box from a top-left corner plus size."""
        return cls(x_min=x, y_min=y, x_max=x + width, y_max=y + height)

    @classmethod
    def from_points(cls, points: list[Point]) -> BoundingBox:
        """Build the tightest box containing ``points``."""
        if not points:
            raise ValueError("cannot build a bounding box from an empty point list")
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        return cls(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return max(self.width, 0.0) * max(self.height, 0.0)

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height, or ``0.0`` for a degenerate box."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def centroid(self) -> Point:
        """Centre of the box."""
        return Point(x=(self.x_min + self.x_max) / 2.0, y=(self.y_min + self.y_max) / 2.0)

    @property
    def is_degenerate(self) -> bool:
        """True when the box has zero area."""
        return self.width <= 0.0 or self.height <= 0.0

    def as_xywh(self) -> tuple[float, float, float, float]:
        """Return ``(x, y, width, height)``."""
        return (self.x_min, self.y_min, self.width, self.height)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        """Return ``(x_min, y_min, x_max, y_max)``."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def normalized(self, shape: ImageShape) -> tuple[float, float, float, float]:
        """Return ``(x, y, w, h)`` as fractions of the image size."""
        return (
            self.x_min / shape.width,
            self.y_min / shape.height,
            self.width / shape.width,
            self.height / shape.height,
        )

    def clipped(self, shape: ImageShape) -> BoundingBox:
        """Return this box intersected with the image rectangle."""
        return BoundingBox(
            x_min=min(max(self.x_min, 0.0), float(shape.width)),
            y_min=min(max(self.y_min, 0.0), float(shape.height)),
            x_max=min(max(self.x_max, 0.0), float(shape.width)),
            y_max=min(max(self.y_max, 0.0), float(shape.height)),
        )

    def expanded(self, pixels: float, shape: ImageShape | None = None) -> BoundingBox:
        """Grow the box by ``pixels`` on every side, optionally clipping to ``shape``."""
        grown = BoundingBox(
            x_min=self.x_min - pixels,
            y_min=self.y_min - pixels,
            x_max=self.x_max + pixels,
            y_max=self.y_max + pixels,
        )
        return grown.clipped(shape) if shape is not None else grown

    def scale_about_centre(self, factor: float, shape: ImageShape | None = None) -> BoundingBox:
        """Scale width and height by ``factor`` around the box centre."""
        centre = self.centroid
        half_w = self.width * factor / 2.0
        half_h = self.height * factor / 2.0
        scaled = BoundingBox(
            x_min=centre.x - half_w,
            y_min=centre.y - half_h,
            x_max=centre.x + half_w,
            y_max=centre.y + half_h,
        )
        return scaled.clipped(shape) if shape is not None else scaled

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union with another box."""
        inter_w = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        inter_h = max(0.0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        intersection = inter_w * inter_h
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def contains_point(self, point: Point) -> bool:
        """True when ``point`` lies inside the box (inclusive of the maximum edge)."""
        return self.x_min <= point.x <= self.x_max and self.y_min <= point.y <= self.y_max
