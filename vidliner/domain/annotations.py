"""The internal annotation intermediate representation.

Every annotation format is converted into these objects on the way in and out of the pipeline.
Operators never see COCO JSON or YOLO text; they see an :class:`AnnotationBundle`. That is what
makes "add another export format" a bounded change and what makes annotation roundtrip tests
possible at all.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidliner.domain.masks import MaskRef, PolygonMask
from vidliner.domain.shapes import BoundingBox, ImageShape

__all__ = [
    "ANNOTATION_FORMATS",
    "AnnotationBundle",
    "AnnotationFormat",
    "AnnotationObject",
    "CategorySpec",
    "ExportProfile",
]


class AnnotationFormat(str):
    """Supported annotation formats."""

    COCO_DETECTION = "coco-detection"
    COCO_INSTANCE = "coco-instance"
    YOLO_DETECTION = "yolo-detection"
    YOLO_SEGMENTATION = "yolo-segmentation"


ANNOTATION_FORMATS: tuple[str, ...] = (
    AnnotationFormat.COCO_DETECTION,
    AnnotationFormat.COCO_INSTANCE,
    AnnotationFormat.YOLO_DETECTION,
    AnnotationFormat.YOLO_SEGMENTATION,
)


class CategorySpec(BaseModel):
    """One label in the annotation vocabulary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category_id: int = Field(ge=0)
    name: str
    supercategory: str | None = None

    @property
    def slug(self) -> str:
        """Filesystem-safe form of the category name."""
        cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in self.name.strip())
        return cleaned.lower() or f"category_{self.category_id}"


class AnnotationObject(BaseModel):
    """One labelled object in one image.

    ``source`` records where the annotation came from, which is what lets the exporter validate that
    an inherited annotation was preserved rather than assumed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    object_id: str
    category_id: int = Field(ge=0)
    bbox: BoundingBox
    polygon: PolygonMask | None = None
    mask_ref: MaskRef | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    iscrowd: bool = False
    track_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    source: str = "inherited"
    """``inherited`` (carried over unchanged), ``regenerated`` (recomputed from pixels), or
    ``manual``."""

    @property
    def area(self) -> float:
        """Pixel area: mask area when segmented, otherwise box area."""
        if self.mask_ref is not None:
            return float(self.mask_ref.area_px)
        if self.polygon is not None and not self.iscrowd:
            return _polygon_area(self.polygon)
        return self.bbox.area

    @property
    def is_segmented(self) -> bool:
        """True when this object carries pixel-level geometry."""
        return self.mask_ref is not None or self.polygon is not None


class AnnotationBundle(BaseModel):
    """The annotation set for one image."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str
    image_shape: ImageShape
    categories: tuple[CategorySpec, ...] = ()
    objects: tuple[AnnotationObject, ...] = ()
    source_format: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_categories(self) -> Self:
        declared = {category.category_id for category in self.categories}
        for obj in self.objects:
            if declared and obj.category_id not in declared:
                raise ValueError(
                    f"annotation object {obj.object_id} references undeclared category {obj.category_id}"
                )
        return self

    @property
    def object_count(self) -> int:
        """Number of annotated objects."""
        return len(self.objects)

    @property
    def is_empty(self) -> bool:
        """True when the image has no annotated objects."""
        return not self.objects

    def category_name(self, category_id: int) -> str | None:
        """Look up a category name by id."""
        for category in self.categories:
            if category.category_id == category_id:
                return category.name
        return None

    def category_id(self, name: str) -> int | None:
        """Look up a category id by name."""
        for category in self.categories:
            if category.name == name:
                return category.category_id
        return None

    def objects_in_category(self, name: str) -> tuple[AnnotationObject, ...]:
        """Every object whose category name matches ``name``."""
        category_id = self.category_id(name)
        if category_id is None:
            return ()
        return tuple(obj for obj in self.objects if obj.category_id == category_id)

    def with_objects(self, objects: tuple[AnnotationObject, ...]) -> AnnotationBundle:
        """Return a copy with a different object list."""
        return self.model_copy(update={"objects": objects})

    def replace_object(self, object_id: str, replacement: AnnotationObject) -> AnnotationBundle:
        """Return a copy where one object has been replaced (or appended when absent)."""
        found = False
        updated: list[AnnotationObject] = []
        for obj in self.objects:
            if obj.object_id == object_id:
                updated.append(replacement)
                found = True
            else:
                updated.append(obj)
        if not found:
            updated.append(replacement)
        return self.model_copy(update={"objects": tuple(updated)})

    def ensure_categories(self, names: tuple[str, ...]) -> AnnotationBundle:
        """Return a copy whose category vocabulary contains at least ``names``.

        New categories are appended with the next free id, preserving existing ids so that an
        inherited annotation keeps its labels.
        """
        existing = {category.name for category in self.categories}
        missing = [name for name in names if name not in existing]
        if not missing:
            return self
        next_id = max((category.category_id for category in self.categories), default=-1) + 1
        added = tuple(
            CategorySpec(category_id=next_id + offset, name=name) for offset, name in enumerate(missing)
        )
        return self.model_copy(update={"categories": (*self.categories, *added)})

    def validate_against_image(self) -> list[str]:
        """Return a list of structural problems (empty when the bundle is sound).

        This deliberately returns problems instead of raising: the annotation-consistency metric
        needs to *count* problems, and the exporter needs to refuse them.
        """
        problems: list[str] = []
        for obj in self.objects:
            box = obj.bbox
            if (
                box.x_min < 0
                or box.y_min < 0
                or box.x_max > self.image_shape.width
                or box.y_max > self.image_shape.height
            ):
                problems.append(f"{obj.object_id}: bounding box outside the image")
            if box.area <= 0:
                problems.append(f"{obj.object_id}: bounding box has zero area")
            if obj.mask_ref is not None and obj.mask_ref.shape != self.image_shape:
                problems.append(f"{obj.object_id}: mask shape does not match the image")
        return problems


class ExportProfile(BaseModel):
    """How an accepted sample set should be written out."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: str = AnnotationFormat.COCO_INSTANCE
    path: str = "dataset"
    copy_images: bool = True
    image_format: str = "same"
    """``same`` keeps source extensions, otherwise ``png`` or ``jpg``."""
    min_area_px: int = Field(default=1, ge=0)
    include_score: bool = True
    include_rejected: bool = False
    include_provenance: bool = True
    include_review_report: bool = True

    @model_validator(mode="after")
    def _check_format(self) -> Self:
        if self.format not in ANNOTATION_FORMATS:
            raise ValueError(
                f"unsupported annotation format {self.format!r}; expected one of {ANNOTATION_FORMATS}"
            )
        if self.image_format not in {"same", "png", "jpg"}:
            raise ValueError("image_format must be 'same', 'png', or 'jpg'")
        return self

    @property
    def is_segmentation(self) -> bool:
        """True when the format carries pixel-level geometry."""
        return self.format in {AnnotationFormat.COCO_INSTANCE, AnnotationFormat.YOLO_SEGMENTATION}

    @property
    def is_coco(self) -> bool:
        """True when the format is a COCO variant."""
        return self.format.startswith("coco")

    @property
    def is_yolo(self) -> bool:
        """True when the format is a YOLO variant."""
        return self.format.startswith("yolo")


def _polygon_area(polygon: PolygonMask) -> float:
    """Shoelace area of the outer ring, in pixels."""
    ring = polygon.rings[0]
    total = 0.0
    for index in range(len(ring)):
        current = ring[index]
        following = ring[(index + 1) % len(ring)]
        total += current.x * following.y - following.x * current.y
    return abs(total) / 2.0
