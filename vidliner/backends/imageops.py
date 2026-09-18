"""Backend-agnostic image utilities.

Masks are the currency of this pipeline, so the mask vocabulary lives in one place: encode/decode
via the artifact store, geometry helpers (centroid, box, polygon conversion, soft edges), and the
visualisation helpers used for review artifacts.

Nothing here is vendor-specific. OpenCV, torch, or a remote service can all be plugged in above this
layer without changing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vidliner.capabilities.backend import ArtifactIO
from vidliner.core.results import ArtifactRef
from vidliner.domain.enums import ArtifactKind
from vidliner.domain.masks import PolygonMask
from vidliner.domain.shapes import BoundingBox, ImageShape, Point

__all__ = [
    "MaskGeometry",
    "boundary_points",
    "box_to_mask",
    "compose_over",
    "difference_visual",
    "dilate",
    "erode",
    "feather",
    "mask_geometry",
    "mask_geometry_annotation",
    "mask_polygon",
    "mask_ring",
    "open_mask",
    "outline_visual",
    "save_image_array",
    "to_grey",
    "touching_border",
]


@dataclass(frozen=True, slots=True)
class MaskGeometry:
    """Geometric summary of a boolean mask."""

    area_px: int
    bbox: BoundingBox
    centroid: Point
    width: int
    height: int

    @property
    def is_empty(self) -> bool:
        """True when the mask selects no pixels."""
        return self.area_px == 0

    @property
    def fill_ratio(self) -> float:
        """Mask area divided by its bounding box area."""
        box_area = self.bbox.area
        return self.area_px / box_area if box_area > 0 else 0.0


def mask_geometry(mask: np.ndarray) -> MaskGeometry:
    """Compute the geometric summary of a boolean mask."""
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return MaskGeometry(
            area_px=0,
            bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0),
            centroid=Point(x=0.0, y=0.0),
            width=0,
            height=0,
        )
    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(columns.min()), int(columns.max())
    return MaskGeometry(
        area_px=int(rows.size),
        bbox=BoundingBox(
            x_min=float(x_min), y_min=float(y_min), x_max=float(x_max + 1), y_max=float(y_max + 1)
        ),
        centroid=Point(x=float(columns.mean()), y=float(rows.mean())),
        width=x_max - x_min + 1,
        height=y_max - y_min + 1,
    )


def box_to_mask(box: BoundingBox, shape: ImageShape, *, rounding: str = "outer") -> np.ndarray:
    """Rasterise a bounding box into a boolean mask."""
    mask = np.zeros((shape.height, shape.width), dtype=bool)
    if rounding == "outer":
        x0 = int(np.floor(box.x_min))
        y0 = int(np.floor(box.y_min))
        x1 = int(np.ceil(box.x_max))
        y1 = int(np.ceil(box.y_max))
    else:
        x0 = round(box.x_min)
        y0 = round(box.y_min)
        x1 = round(box.x_max)
        y1 = round(box.y_max)
    x0 = max(0, min(x0, shape.width))
    y0 = max(0, min(y0, shape.height))
    x1 = max(0, min(x1, shape.width))
    y1 = max(0, min(y1, shape.height))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Boolean erosion with a square element."""
    result = mask.copy()
    for _ in range(max(0, radius)):
        shifted = result.copy()
        shifted[1:, :] &= result[:-1, :]
        shifted[:-1, :] &= result[1:, :]
        shifted[:, 1:] &= result[:, :-1]
        shifted[:, :-1] &= result[:, 1:]
        result = shifted
    return result


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Boolean dilation with a square element."""
    result = mask.copy()
    for _ in range(max(0, radius)):
        shifted = result.copy()
        shifted[1:, :] |= result[:-1, :]
        shifted[:-1, :] |= result[1:, :]
        shifted[:, 1:] |= result[:, :-1]
        shifted[:, :-1] |= result[:, 1:]
        result = shifted
    return result


def close(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Morphological closing: dilate then erode, to bridge small gaps in a mask."""
    return erode(dilate(mask, radius), radius)


def open_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Morphological opening: erode then dilate, to remove speckle."""
    return dilate(erode(mask, radius), radius)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 4-connected component of a mask."""
    if not mask.any():
        return mask.copy()
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    best_label = 0
    best_area = 0
    current = 0
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist(), strict=True):
        if labels[start_y, start_x] != 0:
            continue
        current += 1
        stack = [(start_y, start_x)]
        labels[start_y, start_x] = current
        area = 0
        while stack:
            y, x = stack.pop()
            area += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = current
                    stack.append((ny, nx))
        if area > best_area:
            best_area = area
            best_label = current
    return labels == best_label if best_label else mask.copy()


def feather(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Return a float alpha matte in ``[0, 1]`` with a soft edge of ``radius`` pixels."""
    if radius <= 0:
        return mask.astype(np.float32)
    alpha = mask.astype(np.float32)
    for step in range(radius):
        weight = 1.0 - (step + 1) / (radius + 1.0)
        grown = dilate(mask, step + 1).astype(np.float32)
        alpha = np.maximum(alpha, grown * weight)
    eroded = erode(mask, radius).astype(np.float32)
    return np.clip(alpha * 0.75 + eroded * 0.25, 0.0, 1.0)


def boundary_points(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of the boundary pixels of a mask."""
    return mask & ~erode(mask, 1)


def touching_border(mask: np.ndarray, margin: int = 1) -> bool:
    """True when the mask touches the image border within ``margin`` pixels."""
    if not mask.any():
        return False
    return bool(
        mask[:margin, :].any() or mask[-margin:, :].any() or mask[:, :margin].any() or mask[:, -margin:].any()
    )


def mask_ring(mask: np.ndarray, *, max_points: int = 128, epsilon: float = 1.5) -> list[Point]:
    """Trace the outer boundary of a mask as a closed polygon in pixel space.

    The trace walks the boundary pixels in order around the region and then simplifies the ring with
    a Ramer-Douglas-Peucker pass, so a raster mask becomes a compact vector outline that exporters
    and human reviewers can both read.
    """
    if not mask.any():
        return []
    boundary = boundary_points(mask)
    rows, columns = np.nonzero(boundary)
    start = int(np.argmin(rows * mask.shape[1] + columns))
    start_point = (int(rows[start]), int(columns[start]))
    visited: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    current = start_point
    height, width = mask.shape
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while True:
        visited.add(current)
        ordered.append(current)
        next_point: tuple[int, int] | None = None
        for dy, dx in neighbours:
            candidate = (current[0] + dy, current[1] + dx)
            if not (0 <= candidate[0] < height and 0 <= candidate[1] < width):
                continue
            if candidate in visited or not boundary[candidate]:
                continue
            next_point = candidate
            break
        if next_point is None or len(ordered) > boundary.sum() + 4:
            break
        current = next_point
    if len(ordered) < 3:
        geometry = mask_geometry(mask)
        box = geometry.bbox
        return [
            Point(x=box.x_min, y=box.y_min),
            Point(x=box.x_max, y=box.y_min),
            Point(x=box.x_max, y=box.y_max),
            Point(x=box.x_min, y=box.y_max),
        ]
    points = [Point(x=float(column), y=float(row)) for row, column in ordered]
    simplified = _simplify_closed(points, epsilon)
    if len(simplified) > max_points:
        stride = max(1, len(simplified) // max_points)
        simplified = simplified[::stride]
    return simplified


def mask_polygon(mask: np.ndarray, *, max_points: int = 128) -> PolygonMask | None:
    """Return a closed polygon for a mask, or ``None`` when the mask is empty."""
    ring = mask_ring(mask, max_points=max_points)
    if len(ring) < 3:
        return None
    return PolygonMask(rings=[ring])


def _simplify_closed(points: list[Point], epsilon: float) -> list[Point]:
    if len(points) < 4:
        return points
    simplified = _rdp([*points, points[0]], epsilon)
    if len(simplified) >= 2 and simplified[0] == simplified[-1]:
        simplified = simplified[:-1]
    return simplified


def _rdp(points: list[Point], epsilon: float) -> list[Point]:
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    index, distance = 0, -1.0
    for position in range(1, len(points) - 1):
        current = _perpendicular_distance(points[position], start, end)
        if current > distance:
            index, distance = position, current
    if distance > epsilon:
        left = _rdp(points[: index + 1], epsilon)
        right = _rdp(points[index:], epsilon)
        return left[:-1] + right
    return [start, end]


def _perpendicular_distance(point: Point, start: Point, end: Point) -> float:
    if start == end:
        return point.distance_to(start)
    numerator = abs(
        (end.y - start.y) * point.x - (end.x - start.x) * point.y + end.x * start.y - end.y * start.x
    )
    denominator = max(1e-9, ((end.y - start.y) ** 2 + (end.x - start.x) ** 2) ** 0.5)
    return numerator / denominator


def to_grey(image: np.ndarray) -> np.ndarray:
    """Convert an RGB/RGBA array to float grayscale in ``[0, 1]``."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        grey = array
    else:
        grey = array[..., :3].mean(axis=2)
    if float(np.max(grey)) > 1.5:
        grey = grey / 255.0
    return grey


def compose_over(base: np.ndarray, overlay: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Alpha-composite ``overlay`` onto ``base`` using a float matte."""
    matte = np.clip(alpha, 0.0, 1.0)
    if matte.ndim == 2:
        matte = matte[..., None]
    return matte * overlay.astype(np.float32) + (1.0 - matte) * base.astype(np.float32)


def outline_visual(
    image: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int] = (255, 64, 64), width: int = 2
) -> np.ndarray:
    """Draw a mask outline on a copy of an image, for the review report."""
    canvas = np.asarray(image, dtype=np.uint8).copy()
    if canvas.ndim == 2:
        canvas = np.stack([canvas] * 3, axis=2)
    ring = dilate(boundary_points(mask), max(0, width - 1))
    canvas[ring] = np.array(color, dtype=np.uint8)
    return canvas


def difference_visual(source: np.ndarray, produced: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render an amplified absolute difference with the changed region highlighted."""
    left = to_grey(source)
    right = to_grey(produced)
    delta = np.abs(left - right)
    amplified = np.clip(delta * 4.0 * 255.0, 0, 255).astype(np.uint8)
    canvas = np.stack([amplified] * 3, axis=2)
    canvas[mask] = (canvas[mask] * 0.35 + np.array([180, 40, 40]) * 0.65).astype(np.uint8)
    return canvas


def load_image_array(io: ArtifactIO, artifact: ArtifactRef) -> np.ndarray:
    """Load an artifact as a float32 RGB array in ``[0, 1]`` through a backend I/O object."""
    image = io.image(artifact)
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_image_array(io: ArtifactIO, array: np.ndarray, *, kind: ArtifactKind | None = None) -> ArtifactRef:
    """Store a float32 RGB array as a PNG artifact through a backend I/O object."""
    from PIL import Image

    clipped = np.clip(array, 0.0, 1.0)
    data = (clipped * 255.0 + 0.5).astype(np.uint8)
    image = Image.fromarray(data, mode="RGB")
    return io.save_image(image, kind=kind or ArtifactKind.IMAGE)


def mask_geometry_annotation(
    mask: np.ndarray, *, max_polygon_points: int = 128
) -> tuple[BoundingBox, PolygonMask | None, int]:
    """Derive the annotation geometry of a mask: its bounding box, polygon, and pixel area.

    One function, used by every stage that has to describe an object from pixels — the annotation
    rebuild and the post-generation re-detection. Duplicating this derivation is how a bounding box
    and a polygon end up disagreeing about the same object, so it lives in exactly one place.

    Args:
        mask: boolean mask in image coordinates.
        max_polygon_points: cap on the traced outline's vertex count.

    Returns:
        ``(bbox, polygon, area_px)``. The polygon is ``None`` when the mask has no traceable ring.

    Raises:
        ValueError: when the mask is empty, because an empty mask has no geometry to describe.
    """
    if not mask.any():
        raise ValueError("cannot derive annotation geometry from an empty mask")
    return mask_geometry(mask).bbox, mask_polygon(mask, max_points=max_polygon_points), int(mask.sum())
