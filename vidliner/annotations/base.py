"""Annotation format conversion around the internal IR.

Every format is a pair of functions: read into :class:`~vidliner.domain.annotations.AnnotationBundle`
and write back out. Operators only ever see the bundle, so supporting a new format is additive and
never touches the pipeline.

The COCO reader is deliberately forgiving about the parts of the specification real datasets get
wrong (missing ``info``, string ids, absent ``iscrowd``) and strict about the parts that matter
(segmentation must match the declared format, boxes must be numeric). The YOLO reader follows the
Ultralytics layout: one ``.txt`` per image with ``class cx cy w h`` or ``class x1 y1 x2 y2 ...``
normalised coordinates, plus a ``classes.txt`` (or ``data.yaml``) that names the classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.annotations import (
    ANNOTATION_FORMATS,
    AnnotationBundle,
    AnnotationFormat,
    AnnotationObject,
    CategorySpec,
)
from vidliner.domain.masks import PolygonMask
from vidliner.domain.shapes import BoundingBox, ImageShape, Point

__all__ = [
    "AnnotationReader",
    "AnnotationWriter",
    "ObjectRecord",
    "converter",
    "read_annotations",
    "supported_formats",
    "write_annotations",
]


@dataclass(frozen=True, slots=True)
class ObjectRecord:
    """Format-neutral object row produced by a reader for one image."""

    category_name: str
    bbox: BoundingBox
    polygon: PolygonMask | None = None
    score: float | None = None
    iscrowd: bool = False
    attributes: dict[str, Any] | None = None
    source: str = "inherited"


class AnnotationReader(Protocol):
    """Reads one dataset's annotations into the internal IR."""

    format_name: str

    def categories(self) -> tuple[CategorySpec, ...]:
        """The category vocabulary of the dataset."""
        ...

    def read_image(self, relative_path: str, shape: ImageShape) -> tuple[ObjectRecord, ...]:
        """Read the objects annotated for one image."""
        ...


class AnnotationWriter(Protocol):
    """Writes bundles out in one format."""

    format_name: str

    def build(
        self,
        *,
        bundles: list[tuple[str, AnnotationBundle]],
        categories: tuple[CategorySpec, ...],
    ) -> dict[str, Any]:
        """Build the in-memory representation of the exported annotation file(s)."""
        ...


def supported_formats() -> tuple[str, ...]:
    """Every annotation format VidLiner can read and write."""
    return ANNOTATION_FORMATS


def converter(
    format_name: str,
    *,
    root: Path,
    image_paths: dict[str, str] | None = None,
) -> tuple[AnnotationReader | None, AnnotationWriter | None]:
    """Return the reader and writer for a format name.

    Args:
        format_name: one of :data:`~vidliner.domain.annotations.ANNOTATION_FORMATS`.
        root: dataset directory, used by readers that need sibling files.
        image_paths: mapping of sample path to image file name, used by writers.

    Raises:
        ValidationFailure: for an unknown format.
    """
    if format_name not in ANNOTATION_FORMATS:
        raise ValidationFailure(
            f"unsupported annotation format {format_name!r}; supported formats are {ANNOTATION_FORMATS}",
            code=ErrorCode.ANNOTATION_INVALID,
            detail={"format": format_name},
        )
    from vidliner.annotations import coco, yolo

    if format_name in {AnnotationFormat.COCO_DETECTION, AnnotationFormat.COCO_INSTANCE}:
        reader = coco.CocoReader(root, instance=format_name == AnnotationFormat.COCO_INSTANCE)
        writer = coco.CocoWriter(instance=format_name == AnnotationFormat.COCO_INSTANCE)
        return reader, writer
    reader = yolo.YoloReader(root, segmentation=format_name == AnnotationFormat.YOLO_SEGMENTATION)
    writer = yolo.YoloWriter(
        segmentation=format_name == AnnotationFormat.YOLO_SEGMENTATION,
        image_paths=image_paths or {},
    )
    return reader, writer


def read_annotations(
    format_name: str,
    root: Path,
    *,
    shape_lookup: Any,
) -> tuple[dict[str, AnnotationBundle], tuple[CategorySpec, ...]]:
    """Read a whole dataset's annotations into the internal IR.

    Args:
        format_name: source format.
        root: dataset directory.
        shape_lookup: callable mapping a relative image path to its :class:`ImageShape`.

    Returns:
        ``(bundles by relative path, categories)``.
    """
    reader, _ = converter(format_name, root=root)
    if reader is None:  # pragma: no cover - every known format has a reader
        raise ValidationFailure(f"format {format_name!r} has no reader", code=ErrorCode.ANNOTATION_INVALID)
    categories = reader.categories()
    bundles: dict[str, AnnotationBundle] = {}
    for relative_path in _image_files(root):
        shape = shape_lookup(relative_path)
        if shape is None:
            continue
        records = reader.read_image(relative_path, shape)
        bundles[relative_path] = build_bundle(
            sample_id=relative_path,
            shape=shape,
            categories=categories,
            records=records,
            source_format=format_name,
        )
    return bundles, categories


def build_bundle(
    *,
    sample_id: str,
    shape: ImageShape,
    categories: tuple[CategorySpec, ...],
    records: tuple[ObjectRecord, ...],
    source_format: str | None = None,
) -> AnnotationBundle:
    """Build an annotation bundle from format-neutral records."""
    by_name = {category.name: category for category in categories}
    objects: list[AnnotationObject] = []
    used_categories = list(categories)
    next_id = max((category.category_id for category in categories), default=-1) + 1
    for index, record in enumerate(records):
        category = by_name.get(record.category_name)
        if category is None:
            category = CategorySpec(category_id=next_id, name=record.category_name)
            by_name[record.category_name] = category
            used_categories.append(category)
            next_id += 1
        objects.append(
            AnnotationObject(
                object_id=_object_key(record.category_name, index),
                category_id=category.category_id,
                bbox=record.bbox.clipped(shape),
                polygon=record.polygon,
                score=record.score,
                iscrowd=record.iscrowd,
                attributes=dict(record.attributes or {}),
                source=record.source,
            )
        )
    return AnnotationBundle(
        sample_id=sample_id,
        image_shape=shape,
        categories=tuple(used_categories),
        objects=tuple(objects),
        source_format=source_format,
    )


def write_annotations(
    format_name: str,
    bundles: list[tuple[str, AnnotationBundle]],
    *,
    categories: tuple[CategorySpec, ...],
    image_paths: dict[str, str],
) -> dict[str, Any]:
    """Build the exported document(s) for a set of bundles."""
    _, writer = converter(format_name, root=Path(), image_paths=image_paths)
    if writer is None:  # pragma: no cover - every known format has a writer
        raise ValidationFailure(f"format {format_name!r} has no writer", code=ErrorCode.ANNOTATION_INVALID)
    return writer.build(bundles=bundles, categories=categories)


def _object_key(category_name: str, index: int) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in category_name.lower()) or "object"
    return f"{slug}_{index:04d}"


def _image_files(root: Path) -> list[str]:
    """Relative paths of every image under ``root``, sorted for determinism."""
    from vidliner.domain.media import IMAGE_SUFFIXES

    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path.relative_to(root).as_posix())
    return files


def category_ids_from(names: tuple[str, ...]) -> tuple[CategorySpec, ...]:
    """Stable 1-based category ids for a list of names, as COCO conventionally uses."""
    return tuple(CategorySpec(category_id=index + 1, name=name) for index, name in enumerate(names))


def polygon_from_normalized(
    coordinates: list[float],
    shape: ImageShape,
) -> PolygonMask | None:
    """Convert normalised ``x1 y1 x2 y2 ...`` coordinates into a pixel-space polygon."""
    if len(coordinates) < 6 or len(coordinates) % 2 != 0:
        return None
    points = [
        Point(
            x=float(np_clip(coordinates[index] * shape.width, 0.0, float(shape.width))),
            y=float(np_clip(coordinates[index + 1] * shape.height, 0.0, float(shape.height))),
        )
        for index in range(0, len(coordinates), 2)
    ]
    if len(points) < 3:
        return None
    return PolygonMask(rings=[points])


def box_from_normalized(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    shape: ImageShape,
) -> BoundingBox:
    """Convert a normalised YOLO box into pixel coordinates, clipped to the image."""
    box_width = width * shape.width
    box_height = height * shape.height
    x_min = center_x * shape.width - box_width / 2.0
    y_min = center_y * shape.height - box_height / 2.0
    return BoundingBox.from_xywh(x_min, y_min, box_width, box_height).clipped(shape)


def np_clip(value: float, low: float, high: float) -> float:
    """Clamp a float, avoiding a numpy import in the annotation layer."""
    return low if value < low else high if value > high else value
