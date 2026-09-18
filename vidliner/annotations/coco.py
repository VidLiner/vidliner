"""COCO annotation reader and writer.

Supports both ``coco-detection`` (boxes only) and ``coco-instance`` (boxes plus polygons). The
writer produces one ``instances_<split>.json`` document per split with the four standard top-level
keys, and includes the fields VidLiner's provenance needs without inventing extra keys that a strict
consumer would reject.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vidliner.annotations.base import ObjectRecord
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.annotations import (
    AnnotationBundle,
    AnnotationFormat,
    AnnotationObject,
    CategorySpec,
)
from vidliner.domain.masks import PolygonMask
from vidliner.domain.shapes import BoundingBox, ImageShape

__all__ = ["CocoReader", "CocoWriter"]


@dataclass
class CocoReader:
    """Read a COCO annotations file from a dataset root."""

    root: Path
    instance: bool = True
    _document: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _categories: tuple[CategorySpec, ...] = field(default=(), init=False, repr=False)
    _by_file: dict[str, list[dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)

    format_name: str = AnnotationFormat.COCO_INSTANCE

    def __post_init__(self) -> None:
        self.format_name = (
            AnnotationFormat.COCO_INSTANCE if self.instance else AnnotationFormat.COCO_DETECTION
        )

    def _load(self) -> dict[str, Any]:
        if self._document is not None:
            return self._document
        candidates = [
            self.root / "annotations" / "instances_train.json",
            self.root / "annotations" / "instances.json",
            self.root / "instances_train.json",
            self.root / "instances.json",
            self.root / "annotations.json",
        ]
        existing = [path for path in candidates if path.is_file()]
        if not existing:
            # Fall back to the single JSON document under annotations/, if there is exactly one.
            directory = self.root / "annotations"
            if directory.is_dir():
                json_files = sorted(directory.glob("*.json"))
                if len(json_files) == 1:
                    existing = json_files
        if not existing:
            raise ValidationFailure(
                f"no COCO annotation file was found under {self.root}",
                code=ErrorCode.ANNOTATION_INVALID,
                detail={"root": str(self.root), "looked_for": [str(path) for path in candidates]},
            )
        try:
            document = json.loads(existing[0].read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationFailure(
                f"COCO annotation file {existing[0]} is not valid JSON: {exc}",
                code=ErrorCode.ANNOTATION_INVALID,
            ) from exc
        if not isinstance(document, dict) or "annotations" not in document:
            raise ValidationFailure(
                f"COCO annotation file {existing[0]} has no 'annotations' array",
                code=ErrorCode.ANNOTATION_INVALID,
            )
        self._document = document
        categories = []
        for entry in document.get("categories", []):
            if not isinstance(entry, dict) or "id" not in entry or "name" not in entry:
                continue
            categories.append(
                CategorySpec(
                    category_id=int(entry["id"]),
                    name=str(entry["name"]),
                    supercategory=_optional_str(entry.get("supercategory")),
                )
            )
        self._categories = tuple(categories)
        images = {
            int(entry["id"]): str(entry["file_name"])
            for entry in document.get("images", [])
            if "id" in entry and "file_name" in entry
        }
        for entry in document.get("annotations", []):
            if not isinstance(entry, dict) or "image_id" not in entry:
                continue
            file_name = images.get(int(entry["image_id"]))
            if file_name is None:
                continue
            self._by_file.setdefault(_normalise(file_name), []).append(entry)
        return document

    def categories(self) -> tuple[CategorySpec, ...]:
        """Category vocabulary of the dataset."""
        self._load()
        return self._categories

    def read_image(self, relative_path: str, shape: ImageShape) -> tuple[ObjectRecord, ...]:
        """Read the annotations of one image."""
        self._load()
        entries = self._by_file.get(_normalise(relative_path), [])
        names = {category.category_id: category.name for category in self._categories}
        records: list[ObjectRecord] = []
        for entry in entries:
            category_id = int(entry.get("category_id", -1))
            category_name = names.get(category_id, f"category_{category_id}")
            bbox = _bbox_from(entry.get("bbox"))
            if bbox is None:
                continue
            polygon = None
            if self.instance:
                polygon = _polygon_from(entry.get("segmentation"))
            records.append(
                ObjectRecord(
                    category_name=category_name,
                    bbox=bbox.clipped(shape),
                    polygon=polygon.clipped(shape) if polygon is not None else None,
                    score=_optional_float(entry.get("score")),
                    iscrowd=bool(entry.get("iscrowd", 0)),
                    attributes={"coco_id": entry.get("id")},
                    source="inherited",
                )
            )
        return tuple(records)


@dataclass
class CocoWriter:
    """Write bundles as a COCO annotation document."""

    instance: bool = True
    format_name: str = AnnotationFormat.COCO_INSTANCE

    def __post_init__(self) -> None:
        self.format_name = (
            AnnotationFormat.COCO_INSTANCE if self.instance else AnnotationFormat.COCO_DETECTION
        )

    def build(
        self,
        *,
        bundles: list[tuple[str, AnnotationBundle]],
        categories: tuple[CategorySpec, ...],
    ) -> dict[str, Any]:
        """Build the COCO document for a set of ``(image file name, bundle)`` pairs."""
        document: dict[str, Any] = {
            "info": {
                "description": "VidLiner exported dataset",
                "producer": "vidliner",
                "format": self.format_name,
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": category.category_id,
                    "name": category.name,
                    **({"supercategory": category.supercategory} if category.supercategory else {}),
                }
                for category in categories
            ],
        }
        annotation_id = 1
        for image_id, (file_name, bundle) in enumerate(bundles, start=1):
            document["images"].append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": bundle.image_shape.width,
                    "height": bundle.image_shape.height,
                }
            )
            for obj in bundle.objects:
                if obj.bbox.area <= 0:
                    continue
                entry: dict[str, Any] = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": obj.category_id,
                    "bbox": [
                        round(obj.bbox.x_min, 2),
                        round(obj.bbox.y_min, 2),
                        round(obj.bbox.width, 2),
                        round(obj.bbox.height, 2),
                    ],
                    "area": round(obj.area, 2),
                    "iscrowd": 1 if obj.iscrowd else 0,
                }
                if self.instance:
                    entry["segmentation"] = _segmentation_for(obj)
                if obj.score is not None:
                    entry["score"] = round(obj.score, 6)
                if obj.attributes:
                    entry["attributes"] = _safe_attributes(obj.attributes)
                document["annotations"].append(entry)
                annotation_id += 1
        return document

    def filename(self, split: str) -> str:
        """Relative path of the annotation file for a split."""
        suffix = "instance" if self.instance else "detection"
        return f"annotations/instances_{split}_{suffix}.json"


def _segmentation_for(obj: AnnotationObject) -> list[list[float]]:
    """COCO segmentation for one object: polygon when present, otherwise the antialiased box."""
    if obj.polygon is not None:
        return [obj.polygon.as_coco_flat()]
    box = obj.bbox
    return [
        [
            round(box.x_min, 2),
            round(box.y_min, 2),
            round(box.x_max, 2),
            round(box.y_min, 2),
            round(box.x_max, 2),
            round(box.y_max, 2),
            round(box.x_min, 2),
            round(box.y_max, 2),
        ]
    ]


def _safe_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Keep JSON-safe attribute values, dropping anything exotic rather than failing the export."""
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
    return safe


def _bbox_from(raw: object) -> BoundingBox | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return BoundingBox.from_xywh(x, y, width, height)


def _polygon_from(raw: object) -> PolygonMask | None:
    if not isinstance(raw, list) or not raw:
        return None
    first = raw[0]
    if not isinstance(first, list) or len(first) < 6:
        return None
    try:
        return PolygonMask.from_coco_flat([float(value) for value in first])
    except (TypeError, ValueError, ValidationFailure):
        return None


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _normalise(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")
