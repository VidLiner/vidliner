"""YOLO annotation reader and writer.

Follows the common Ultralytics layout:

```
<root>/
  classes.txt                 one class name per line, 0-based index
  data.yaml                   optional; `names:` list is also understood
  images/<name>.jpg
  labels/<name>.txt           `class cx cy w h`, or `class x1 y1 x2 y2 ...` normalised
```

Both detection and segmentation variants are supported. The writer emits ``images/``, ``labels/``,
``classes.txt``, and ``data.yaml`` so the exported tree is directly usable for training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vidliner.annotations.base import (
    ObjectRecord,
    box_from_normalized,
    polygon_from_normalized,
)
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.annotations import (
    AnnotationBundle,
    AnnotationFormat,
    AnnotationObject,
    CategorySpec,
)
from vidliner.domain.shapes import ImageShape

__all__ = ["YoloReader", "YoloWriter"]


@dataclass
class YoloReader:
    """Read YOLO label files from a dataset root."""

    root: Path
    segmentation: bool = False
    _categories: tuple[CategorySpec, ...] | None = field(default=None, init=False, repr=False)

    format_name: str = AnnotationFormat.YOLO_DETECTION

    def __post_init__(self) -> None:
        self.format_name = (
            AnnotationFormat.YOLO_SEGMENTATION if self.segmentation else AnnotationFormat.YOLO_DETECTION
        )

    def categories(self) -> tuple[CategorySpec, ...]:
        """Class vocabulary, read from ``classes.txt`` or ``data.yaml``.

        Raises:
            ValidationFailure: when neither file exists, because silently inventing class names
                would produce plausible-looking but wrong labels.
        """
        if self._categories is not None:
            return self._categories
        names: list[str] = []
        classes_file = self.root / "classes.txt"
        if classes_file.is_file():
            names = [
                line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
        if not names:
            names = _names_from_data_yaml(self.root / "data.yaml")
        if not names:
            raise ValidationFailure(
                f"no class names were found under {self.root}: expected classes.txt or data.yaml",
                code=ErrorCode.ANNOTATION_INVALID,
                detail={"root": str(self.root)},
            )
        self._categories = tuple(
            CategorySpec(category_id=index, name=name) for index, name in enumerate(names)
        )
        return self._categories

    def read_image(self, relative_path: str, shape: ImageShape) -> tuple[ObjectRecord, ...]:
        """Read the label file belonging to one image."""
        categories = self.categories()
        label_path = self._label_path(relative_path)
        if label_path is None or not label_path.is_file():
            return ()
        records: list[ObjectRecord] = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                numbers = [float(part) for part in parts]
            except ValueError as exc:
                raise ValidationFailure(
                    f"{label_path} line {line_number} contains a non-numeric field",
                    code=ErrorCode.ANNOTATION_INVALID,
                    detail={"file": str(label_path), "line": line_number},
                ) from exc
            if len(numbers) < 5:
                raise ValidationFailure(
                    f"{label_path} line {line_number} has fewer than five fields",
                    code=ErrorCode.ANNOTATION_INVALID,
                    detail={"file": str(label_path), "line": line_number},
                )
            class_index = int(numbers[0])
            category_name = (
                categories[class_index].name if 0 <= class_index < len(categories) else f"class_{class_index}"
            )
            payload = numbers[1:]
            if len(payload) == 4 and not self.segmentation:
                center_x, center_y, width, height = payload
                box = box_from_normalized(center_x, center_y, width, height, shape)
                records.append(ObjectRecord(category_name=category_name, bbox=box))
                continue
            if len(payload) == 4:
                # A detection file read in segmentation mode: keep the box rather than failing.
                center_x, center_y, width, height = payload
                box = box_from_normalized(center_x, center_y, width, height, shape)
                records.append(ObjectRecord(category_name=category_name, bbox=box))
                continue
            polygon = polygon_from_normalized(payload, shape)
            if polygon is None:
                continue
            records.append(
                ObjectRecord(category_name=category_name, bbox=polygon.bbox.clipped(shape), polygon=polygon)
            )
        return tuple(records)

    def _label_path(self, relative_path: str) -> Path | None:
        """Locate the label file for an image path, accepting ``images/`` and flat layouts."""
        path = Path(relative_path)
        candidates: list[Path] = []
        if path.parts and path.parts[0] == "images":
            candidates.append(self.root / "labels" / Path(*path.parts[1:]).with_suffix(".txt"))
        candidates.append(self.root / "labels" / path.with_suffix(".txt"))
        candidates.append(self.root / path.with_suffix(".txt"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0] if candidates else None


@dataclass
class YoloWriter:
    """Write bundles as YOLO image/label pairs."""

    segmentation: bool = False
    image_paths: dict[str, str] = field(default_factory=dict)
    format_name: str = AnnotationFormat.YOLO_DETECTION

    def __post_init__(self) -> None:
        self.format_name = (
            AnnotationFormat.YOLO_SEGMENTATION if self.segmentation else AnnotationFormat.YOLO_DETECTION
        )

    def build(
        self,
        *,
        bundles: list[tuple[str, AnnotationBundle]],
        categories: tuple[CategorySpec, ...],
    ) -> dict[str, Any]:
        """Build the label files and metadata for an exported dataset.

        Returns:
            ``{"labels": {relative label path: text}, "classes": [...], "data_yaml": {...},
            "images": {relative image path: source relative path}}``.
        """
        index_by_name = {category.name: category.category_id for category in categories}
        fallback = {category.category_id: category.name for category in categories}
        labels: dict[str, str] = {}
        images: dict[str, str] = {}
        for file_name, bundle in bundles:
            label_path = str(Path("labels") / Path(file_name).with_suffix(".txt"))
            lines: list[str] = []
            for obj in bundle.objects:
                category_name = bundle.category_name(obj.category_id)
                if category_name is None:
                    category_name = fallback.get(obj.category_id, f"category_{obj.category_id}")
                class_index = index_by_name.get(category_name)
                if class_index is None:
                    continue
                line = self._line(obj, class_index, bundle.image_shape)
                if line:
                    lines.append(line)
            labels[label_path] = "\n".join(lines) + ("\n" if lines else "")
            source = self.image_paths.get(file_name, file_name)
            images[str(Path("images") / Path(file_name).name)] = source
        return {
            "labels": labels,
            "images": images,
            "classes": [category.name for category in categories],
            "data_yaml": {
                "path": ".",
                "train": "images",
                "val": "images",
                "nc": len(categories),
                "names": [category.name for category in categories],
            },
        }

    def _line(self, obj: AnnotationObject, class_index: int, shape: ImageShape) -> str:
        if self.segmentation and obj.polygon is not None:
            coordinates: list[str] = []
            for x, y in obj.polygon.normalized_rings(shape)[0]:
                coordinates.append(f"{_clamp(x):.6f}")
                coordinates.append(f"{_clamp(y):.6f}")
            return f"{class_index} " + " ".join(coordinates)
        box = obj.bbox.clipped(shape)
        center_x = (box.x_min + box.x_max) / 2.0 / shape.width
        center_y = (box.y_min + box.y_max) / 2.0 / shape.height
        width = box.width / shape.width
        height = box.height / shape.height
        if width <= 0 or height <= 0:
            return ""
        return f"{class_index} {_clamp(center_x):.6f} {_clamp(center_y):.6f} {_clamp(width):.6f} {_clamp(height):.6f}"


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _names_from_data_yaml(path: Path) -> list[str]:
    """Extract a class-name list from a YOLO ``data.yaml``."""
    if not path.is_file():
        return []
    try:
        import yaml
    except ImportError:  # pragma: no cover - PyYAML is a hard dependency
        return []
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(document, dict):
        return []
    names = document.get("names")
    if isinstance(names, list):
        return [str(item) for item in names]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    return []
