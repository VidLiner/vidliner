"""Synthetic dataset fixtures.

Every test in this repository runs against datasets generated here. They are tiny, deterministic, and
built from plain numpy arrays so a test never needs a downloaded image, a network call, or a paid API
— which is the property that makes the whole suite runnable in CI, and the property the product's
"works with fake backends" requirement depends on.

The generator makes scenes with a *known* structure: a flat background, one or more objects with
corners, and enough texture that a detector, a segmenter, and a quality metric all have something
real to measure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "SceneObject",
    "SyntheticDataset",
    "build_dataset",
    "render_scene",
    "write_coco",
    "write_yolo",
]


@dataclass(frozen=True, slots=True)
class SceneObject:
    """One object drawn into a synthetic scene."""

    class_name: str
    bbox: tuple[int, int, int, int]
    """``(x_min, y_min, x_max, y_max)`` in pixels."""
    colour: tuple[int, int, int]
    label: int = 0

    @property
    def width(self) -> int:
        """Object width in pixels."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Object height in pixels."""
        return self.bbox[3] - self.bbox[1]


@dataclass
class SyntheticDataset:
    """A generated dataset on disk."""

    root: Path
    images: list[Path] = field(default_factory=list)
    objects: dict[str, list[SceneObject]] = field(default_factory=dict)
    splits: dict[str, str] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Number of images written."""
        return len(self.images)


def render_scene(
    *,
    width: int = 160,
    height: int = 120,
    background: tuple[int, int, int] = (110, 120, 130),
    objects: tuple[SceneObject, ...] = (),
    seed: int = 0,
    texture: float = 0.06,
) -> np.ndarray:
    """Render a synthetic scene as an RGB ``uint8`` array.

    The background gets low-frequency texture so SSIM and gradient-based metrics are meaningful;
    a perfectly flat image would make several quality checks degenerate.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:] = np.array(background, dtype=np.float32) / 255.0
    base += rng.normal(0.0, texture, size=base.shape).astype(np.float32)
    # A soft vertical gradient gives the scene a light direction to measure.
    ramp = np.linspace(-0.04, 0.04, height, dtype=np.float32)[:, None]
    base += ramp[..., None]
    for obj in objects:
        x0, y0, x1, y1 = obj.bbox
        colour = np.array(obj.colour, dtype=np.float32) / 255.0
        patch = np.zeros((max(0, y1 - y0), max(0, x1 - x0), 3), dtype=np.float32)
        if patch.size == 0:
            continue
        patch[:] = colour
        patch += rng.normal(0.0, 0.02, size=patch.shape).astype(np.float32)
        # A darker strip along the bottom edge reads as shading and gives the object structure.
        patch[-max(1, patch.shape[0] // 8) :, :, :] *= 0.75
        base[y0:y1, x0:x1] = patch
    return np.clip(base * 255.0, 0, 255).astype(np.uint8)


def build_dataset(
    root: Path,
    *,
    count: int = 10,
    width: int = 160,
    height: int = 120,
    classes: tuple[str, ...] = ("car",),
    split_mode: str = "none",
    seed: int = 7,
    annotations: bool = True,
) -> SyntheticDataset:
    """Write a synthetic image dataset to ``root``.

    Args:
        root: dataset directory (created if missing).
        count: number of images.
        width: image width in pixels.
        height: image height in pixels.
        classes: object classes to draw, cycled across images.
        split_mode: ``none`` (flat), ``directory`` (``train/``, ``val/``), or ``filename``.
        seed: base seed; each image derives its own.
        annotations: whether to write a COCO annotation file.

    Returns:
        A :class:`SyntheticDataset` describing what was written.
    """
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    dataset = SyntheticDataset(root=root, categories=list(classes))

    for index in range(count):
        rng = np.random.default_rng(seed + index)
        class_name = classes[index % len(classes)]
        object_width = int(rng.integers(width // 6, width // 3))
        object_height = int(rng.integers(height // 6, height // 3))
        x0 = int(rng.integers(4, max(5, width - object_width - 4)))
        y0 = int(rng.integers(4, max(5, height - object_height - 4)))
        colour = (
            int(rng.integers(40, 230)),
            int(rng.integers(40, 230)),
            int(rng.integers(40, 230)),
        )
        obj = SceneObject(
            class_name=class_name,
            bbox=(x0, y0, x0 + object_width, y0 + object_height),
            colour=colour,
            label=classes.index(class_name),
        )
        image = render_scene(width=width, height=height, objects=(obj,), seed=seed + index)
        relative = _relative_path(index, split_mode)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(target)
        dataset.images.append(target)
        dataset.objects[relative] = [obj]
        dataset.splits[relative] = _split_for(relative, split_mode)

    if annotations:
        if split_mode == "directory":
            for split in sorted(set(dataset.splits.values())):
                members = [path for path, name in dataset.splits.items() if name == split]
                if members:
                    write_coco(root, dataset, members, filename=f"annotations/instances_{split}.json")
        else:
            write_coco(root, dataset, list(dataset.splits), filename="annotations/instances.json")
    return dataset


def write_coco(
    root: Path,
    dataset: SyntheticDataset,
    members: list[str],
    *,
    filename: str = "annotations/instances.json",
) -> Path:
    """Write a COCO annotation document for the selected members."""
    from PIL import Image

    categories = [{"id": index + 1, "name": name} for index, name in enumerate(dataset.categories)]
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1
    for image_id, relative in enumerate(sorted(members), start=1):
        path = root / relative
        with Image.open(path) as handle:
            width, height = handle.size
        images.append({"id": image_id, "file_name": relative, "width": width, "height": height})
        for obj in dataset.objects.get(relative, []):
            x0, y0, x1, y1 = obj.bbox
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": obj.label + 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": float((x1 - x0) * (y1 - y0)),
                    "iscrowd": 0,
                    "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
                }
            )
            annotation_id += 1
    document = {
        "info": {"description": "vidliner synthetic fixture"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    target = root / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return target


def write_yolo(
    root: Path,
    dataset: SyntheticDataset,
    members: list[str],
    *,
    segmentation: bool = False,
) -> Path:
    """Write a YOLO dataset layout: ``classes.txt``, ``labels/`` and a ``data.yaml``."""
    from PIL import Image

    (root / "classes.txt").write_text("\n".join(dataset.categories) + "\n", encoding="utf-8")
    for relative in sorted(members):
        path = root / relative
        with Image.open(path) as handle:
            width, height = handle.size
        lines: list[str] = []
        for obj in dataset.objects.get(relative, []):
            x0, y0, x1, y1 = obj.bbox
            if segmentation:
                coordinates = [
                    x0 / width,
                    y0 / height,
                    x1 / width,
                    y0 / height,
                    x1 / width,
                    y1 / height,
                    x0 / width,
                    y1 / height,
                ]
                lines.append(f"{obj.label} " + " ".join(f"{value:.6f}" for value in coordinates))
            else:
                centre_x = ((x0 + x1) / 2) / width
                centre_y = ((y0 + y1) / 2) / height
                box_width = (x1 - x0) / width
                box_height = (y1 - y0) / height
                lines.append(f"{obj.label} {centre_x:.6f} {centre_y:.6f} {box_width:.6f} {box_height:.6f}")
        label_path = root / "labels" / Path(relative).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    names = ", ".join(dataset.categories)
    (root / "data.yaml").write_text(
        f"path: .\ntrain: images\nval: images\nnc: {len(dataset.categories)}\nnames: [{names}]\n",
        encoding="utf-8",
    )
    return root / "classes.txt"


def _relative_path(index: int, split_mode: str) -> str:
    name = f"img_{index:03d}.png"
    if split_mode == "directory":
        split = "train" if index < 8 else "val"
        return f"{split}/{name}"
    if split_mode == "filename":
        prefix = "train" if index < 8 else "val"
        return f"{prefix}_{name}"
    return name


def _split_for(relative: str, split_mode: str) -> str:
    if split_mode == "directory":
        return relative.split("/", 1)[0]
    if split_mode == "filename":
        return relative.split("_", 1)[0]
    return "train"
