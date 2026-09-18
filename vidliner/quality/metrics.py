"""Image measurements used by the quality gates.

Everything here is a pure function of pixels. There is no model, no vendor, and no randomness, which
is what makes the quality stage reproducible and testable without a paid API.

Conventions:

* images are ``float32`` arrays in ``[0, 1]``;
* masks are boolean arrays;
* every score is normalised so that **higher is better**, including the artifact scores, which
  express *freedom from* an artifact. Gate arithmetic then never has to reason about direction.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ArtifactMeasurements",
    "BackgroundMeasurements",
    "BoundaryMeasurements",
    "GeometryMeasurements",
    "annotation_consistency_score",
    "background_measurements",
    "boundary_quality",
    "difference_map",
    "geometry_measurements",
    "image_similarity",
    "mask_iou",
    "similarity_score",
]


def difference_map(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Per-pixel absolute difference between two images, averaged over channels."""
    if left.shape != right.shape:
        raise ValueError(f"cannot difference images of shapes {left.shape} and {right.shape}")
    if left.ndim == 2:
        return np.abs(left - right)
    return np.abs(left - right).mean(axis=2)


def image_similarity(left: np.ndarray, right: np.ndarray, *, region: np.ndarray | None = None) -> float:
    """Windowed structural similarity (SSIM) over an optional boolean region.

    A uniform 8-pixel window is used with the standard constants. The implementation is
    deliberately dependency-free so the metric cannot silently change meaning between environments.
    """
    if left.shape != right.shape:
        raise ValueError("SSIM requires images of identical shape")
    a = left.astype(np.float64, copy=False)
    b = right.astype(np.float64, copy=False)
    if a.ndim == 3:
        a = a.mean(axis=2)
        b = b.mean(axis=2)

    window = 8
    kernel = np.ones((window, window), dtype=np.float64) / float(window * window)

    mu_a = _convolve(a, kernel)
    mu_b = _convolve(b, kernel)
    mu_aa = mu_a * mu_a
    mu_bb = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_aa = _convolve(a * a, kernel) - mu_aa
    sigma_bb = _convolve(b * b, kernel) - mu_bb
    sigma_ab = _convolve(a * b, kernel) - mu_ab

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_aa + mu_bb + c1) * (sigma_aa + sigma_bb + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-12)

    if region is None or not region.any():
        return float(np.clip(ssim_map.mean(), 0.0, 1.0))
    valid = region & _border_valid(region.shape, window)
    if not valid.any():
        return float(np.clip(ssim_map.mean(), 0.0, 1.0))
    return float(np.clip(ssim_map[valid].mean(), 0.0, 1.0))


def _convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Same-size 2-D correlation via an integral image, valid only for a uniform kernel."""
    window = kernel.shape[0]
    if not np.allclose(kernel, kernel[0, 0]):
        raise ValueError("the integral-image path only supports uniform kernels")
    padded = np.pad(image, window // 2, mode="edge")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    height, width = image.shape
    total = (
        integral[window : window + height, window : window + width]
        - integral[0:height, window : window + width]
        - integral[window : window + height, 0:width]
        + integral[0:height, 0:width]
    )
    return total / float(window * window)


def _border_valid(shape: tuple[int, int], window: int) -> np.ndarray:
    valid = np.zeros(shape, dtype=bool)
    half = window // 2
    valid[half : shape[0] - half, half : shape[1] - half] = True
    return valid


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Sobel-free gradient magnitude using central differences, averaged over channels."""
    grey = image.mean(axis=2) if image.ndim == 3 else image
    dy = np.zeros_like(grey)
    dx = np.zeros_like(grey)
    dy[1:-1, :] = (grey[2:, :] - grey[:-2, :]) / 2.0
    dx[:, 1:-1] = (grey[:, 2:] - grey[:, :-2]) / 2.0
    return np.hypot(dx, dy)


def erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Boolean erosion with a square structuring element, implemented with shifted shifts."""
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
    """Boolean dilation with a square structuring element."""
    result = mask.copy()
    for _ in range(max(0, radius)):
        shifted = result.copy()
        shifted[1:, :] |= result[:-1, :]
        shifted[:-1, :] |= result[1:, :]
        shifted[:, 1:] |= result[:, :-1]
        shifted[:, :-1] |= result[:, 1:]
        result = shifted
    return result


def boundary_band(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """The ring of pixels within ``radius`` of the mask boundary."""
    return dilate(mask, radius) & ~erode(mask, radius)


def connected_components(mask: np.ndarray) -> list[int]:
    """Areas of the 4-connected components of a boolean mask, largest first."""
    if not mask.any():
        return []
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    areas: list[int] = []
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
        areas.append(area)
    areas.sort(reverse=True)
    return areas


class BackgroundMeasurements:
    """How much of the non-target region changed."""

    __slots__ = ("changed_ratio", "mean_delta", "perceptual_delta", "samples", "ssim")

    def __init__(
        self,
        *,
        changed_ratio: float,
        mean_delta: float,
        perceptual_delta: float,
        samples: int,
        ssim: float,
    ) -> None:
        self.changed_ratio = changed_ratio
        self.mean_delta = mean_delta
        self.perceptual_delta = perceptual_delta
        self.samples = samples
        self.ssim = ssim

    @property
    def score(self) -> float:
        """Weighted background-preservation score in ``[0, 1]``."""
        return float(
            np.clip(
                0.5 * self.ssim + 0.3 * (1.0 - self.changed_ratio) + 0.2 * self.perceptual_delta,
                0.0,
                1.0,
            )
        )

    def as_detail(self) -> dict[str, float]:
        """Measurements recorded inside the metric outcome for audit."""
        return {
            "ssim_outside": round(self.ssim, 6),
            "changed_ratio": round(self.changed_ratio, 6),
            "perceptual_delta": round(self.perceptual_delta, 6),
            "mean_delta": round(self.mean_delta, 6),
            "non_target_pixels": self.samples,
        }


def background_measurements(
    source: np.ndarray,
    produced: np.ndarray,
    mask: np.ndarray,
    *,
    feather_px: int = 2,
    change_threshold: float = 0.10,
) -> BackgroundMeasurements:
    """Measure how the pixels outside the target mask changed.

    Args:
        source: the untouched source image.
        produced: the composited image.
        mask: the target object mask.
        feather_px: pixels around the mask to exclude from the comparison, so a legitimate blend
            seam is not counted as a background change.
        change_threshold: per-pixel normalised delta above which a pixel counts as changed.

    Returns:
        The measurements; ``score`` combines them into one number.
    """
    if source.shape != produced.shape:
        raise ValueError("background comparison requires images of identical shape")
    exclude = dilate(mask, feather_px) if feather_px > 0 else mask
    outside = ~exclude
    if not outside.any():
        return BackgroundMeasurements(
            changed_ratio=0.0, mean_delta=0.0, perceptual_delta=1.0, samples=0, ssim=1.0
        )

    delta = difference_map(source, produced)
    changed = delta > change_threshold
    changed_ratio = float(changed[outside].mean()) if outside.any() else 0.0
    mean_delta = float(delta[outside].mean())
    # Perceptual delta compares luminance and chroma after light smoothing, so single-pixel sensor
    # noise does not read as a scene change while a recoloured background does.
    smoothed_source = _smooth(source, 2)
    smoothed_produced = _smooth(produced, 2)
    perceptual = difference_map(smoothed_source, smoothed_produced)
    perceptual_mean = float(perceptual[outside].mean())
    perceptual_delta = float(np.clip(1.0 - perceptual_mean / 0.25, 0.0, 1.0))
    ssim_outside = image_similarity(source, produced, region=outside)
    return BackgroundMeasurements(
        changed_ratio=changed_ratio,
        mean_delta=mean_delta,
        perceptual_delta=perceptual_delta,
        samples=int(outside.sum()),
        ssim=ssim_outside,
    )


def _smooth(image: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones((window, window), dtype=np.float64) / float(window * window)
    if image.ndim == 2:
        return _convolve(image.astype(np.float64), kernel)
    channels = [_convolve(image[:, :, index].astype(np.float64), kernel) for index in range(image.shape[2])]
    return np.stack(channels, axis=2)


class BoundaryMeasurements:
    """Mask boundary quality."""

    __slots__ = ("eroded_area", "gradient_gap", "interior_area", "is_empty", "seam")

    def __init__(
        self,
        *,
        seam: float,
        gradient_gap: float,
        eroded_area: int,
        interior_area: int,
        is_empty: bool,
    ) -> None:
        self.seam = seam
        self.gradient_gap = gradient_gap
        self.eroded_area = eroded_area
        self.interior_area = interior_area
        self.is_empty = is_empty

    @property
    def score(self) -> float:
        """Boundary score in ``[0, 1]``: seamless blending without a clipped or hollow mask."""
        if self.is_empty:
            return 0.0
        hollow = min(1.0, self.interior_area / max(1, self.eroded_area * 4))
        return float(np.clip(0.5 * self.seam + 0.3 * (1.0 - self.gradient_gap) + 0.2 * hollow, 0.0, 1.0))

    def as_detail(self) -> dict[str, float]:
        """Boundary measurements recorded inside the metric outcome."""
        return {
            "seam": round(self.seam, 6),
            "gradient_gap": round(self.gradient_gap, 6),
            "interior_area": float(self.interior_area),
            "eroded_area": float(self.eroded_area),
        }


def boundary_quality(image: np.ndarray, mask: np.ndarray, *, radius: int = 3) -> BoundaryMeasurements:
    """Measure how well the mask boundary sits on real image structure.

    A blend that follows the object's own contour produces a boundary whose gradient is comparable
    to the gradient just outside it. A visible seam produces a much stronger gradient exactly on the
    boundary; a cut-through (the object clipped by a wrong mask) produces a weak one inside.
    """
    if image.shape[:2] != mask.shape:
        raise ValueError("mask and image shapes must agree")
    if not mask.any():
        return BoundaryMeasurements(seam=0.0, gradient_gap=1.0, eroded_area=0, interior_area=0, is_empty=True)
    gradient = gradient_magnitude(image)
    band = boundary_band(mask, radius)
    inner = erode(mask, radius)
    outer = dilate(mask, radius) & ~mask
    band_mean = float(gradient[band].mean()) if band.any() else 0.0
    outer_mean = float(gradient[outer].mean()) if outer.any() else band_mean
    reference = max(outer_mean, 1e-6)
    ratio = band_mean / reference
    # A ratio near 1 means the boundary looks like ordinary image structure; far above it means a
    # visible seam, far below it means the mask cut through a flat region.
    seam = float(np.clip(1.0 - abs(ratio - 1.0) / 1.5, 0.0, 1.0))
    gradient_gap = float(np.clip(abs(band_mean - outer_mean) / max(band_mean, outer_mean, 1e-6), 0.0, 1.0))
    return BoundaryMeasurements(
        seam=seam,
        gradient_gap=gradient_gap,
        eroded_area=int(inner.sum()),
        interior_area=int(inner.sum()),
        is_empty=False,
    )


class ArtifactMeasurements:
    """Signs of trouble in the generated image."""

    __slots__ = ("component_ratio", "edge_density", "floating", "text_like", "total_area")

    def __init__(
        self,
        *,
        component_ratio: float,
        floating: float,
        edge_density: float,
        text_like: float,
        total_area: int,
    ) -> None:
        self.component_ratio = component_ratio
        self.floating = floating
        self.edge_density = edge_density
        self.text_like = text_like
        self.total_area = total_area

    @property
    def artifact_score(self) -> float:
        """How much artifact evidence was found: ``0`` clean, ``1`` severe."""
        return float(
            np.clip(
                0.35 * self.floating
                + 0.25 * self.edge_density
                + 0.20 * self.text_like
                + 0.20 * max(0.0, 1.0 - self.component_ratio),
                0.0,
                1.0,
            )
        )

    @property
    def freedom_score(self) -> float:
        """``1 - artifact_score``, the value the ``artifact_free`` gate consumes."""
        return 1.0 - self.artifact_score

    def as_detail(self) -> dict[str, float]:
        """Artifact measurements recorded inside the metric outcome."""
        return {
            "component_ratio": round(self.component_ratio, 6),
            "floating": round(self.floating, 6),
            "edge_density": round(self.edge_density, 6),
            "text_like": round(self.text_like, 6),
            "mask_area": float(self.total_area),
        }


def artifact_measurements(image: np.ndarray, mask: np.ndarray) -> ArtifactMeasurements:
    """Look for duplicated components, floating objects, seam clutter, and text-like texture.

    These are heuristics, not a vision-language judgement. They are cheap, deterministic, and
    deliberately conservative; a deployment that binds a VLM evaluator replaces them with a learned
    judgement without changing the gate.
    """
    if not mask.any():
        return ArtifactMeasurements(
            component_ratio=1.0, floating=0.0, edge_density=0.0, text_like=0.0, total_area=0
        )
    areas = connected_components(mask)
    total = sum(areas)
    largest = areas[0] if areas else 0
    component_ratio = largest / total if total else 1.0
    floating = _floating_evidence(image, mask)
    band = boundary_band(mask, 2)
    gradient = gradient_magnitude(image)
    outer = dilate(mask, 4) & ~dilate(mask, 1)
    band_mean = float(gradient[band].mean()) if band.any() else 0.0
    outer_mean = float(gradient[outer].mean()) if outer.any() else 0.0
    edge_density = float(np.clip(max(0.0, band_mean - outer_mean) / max(outer_mean, 1e-6) / 3.0, 0.0, 1.0))
    text_like = _text_like_evidence(image, mask)
    return ArtifactMeasurements(
        component_ratio=float(component_ratio),
        floating=floating,
        edge_density=edge_density,
        text_like=text_like,
        total_area=total,
    )


def _floating_evidence(image: np.ndarray, mask: np.ndarray) -> float:
    """Evidence that the object does not rest on any surface.

    The check looks at the rows just below the lowest mask pixel: a grounded object usually has a
    contact shadow or a surface edge there, so a completely flat region below a large object is
    suspicious.
    """
    rows = np.nonzero(mask.any(axis=1))[0]
    if rows.size == 0:
        return 0.0
    bottom = int(rows[-1])
    height = mask.shape[0]
    if bottom + 3 >= height:
        return 0.0  # the object leaves the frame; contact cannot be judged
    gradient = gradient_magnitude(image)
    below = gradient[bottom + 1 : min(height, bottom + 4), :]
    mask_columns = mask[bottom, :]
    if not mask_columns.any() or below.size == 0:
        return 0.0
    below_energy = float(below[:, mask_columns].mean())
    object_energy = float(gradient[mask].mean()) if mask.any() else 0.0
    if object_energy <= 1e-6:
        return 0.0
    ratio = below_energy / object_energy
    return float(np.clip(1.0 - ratio / 0.35, 0.0, 1.0))


def _text_like_evidence(image: np.ndarray, mask: np.ndarray) -> float:
    """Evidence of dense high-contrast texture inside the mask, a cheap watermark/text proxy."""
    if not mask.any():
        return 0.0
    gradient = gradient_magnitude(_smooth(image, 2))
    values = gradient[mask]
    if values.size == 0:
        return 0.0
    strong = float((values > 0.25).mean())
    return float(np.clip((strong - 0.08) / 0.25, 0.0, 1.0))


class GeometryMeasurements:
    """How the replaced object's geometry compares with the source object's."""

    __slots__ = (
        "area_ratio",
        "aspect_delta",
        "centroid_shift",
        "contact_shift",
        "source_area",
        "target_area",
    )

    def __init__(
        self,
        *,
        centroid_shift: float,
        area_ratio: float,
        aspect_delta: float,
        contact_shift: float,
        source_area: int,
        target_area: int,
    ) -> None:
        self.centroid_shift = centroid_shift
        self.area_ratio = area_ratio
        self.aspect_delta = aspect_delta
        self.contact_shift = contact_shift
        self.source_area = source_area
        self.target_area = target_area

    def as_detail(self) -> dict[str, float]:
        """Geometry measurements recorded inside the metric outcome."""
        return {
            "centroid_shift": round(self.centroid_shift, 6),
            "area_ratio": round(self.area_ratio, 6),
            "aspect_delta": round(self.aspect_delta, 6),
            "contact_shift": round(self.contact_shift, 6),
            "source_area": float(self.source_area),
            "target_area": float(self.target_area),
        }


def geometry_measurements(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
) -> GeometryMeasurements:
    """Compare the source object's geometry with the regenerated object's geometry.

    ``centroid_shift`` is expressed as a fraction of the image diagonal so a single recipe works
    across resolutions.
    """
    if source_mask.shape != target_mask.shape:
        raise ValueError("geometry comparison requires masks of identical shape")
    diagonal = float(np.hypot(*source_mask.shape))
    source_area = int(source_mask.sum())
    target_area = int(target_mask.sum())
    source_centroid = _centroid(source_mask)
    target_centroid = _centroid(target_mask)
    if source_centroid is None or target_centroid is None:
        return GeometryMeasurements(
            centroid_shift=1.0,
            area_ratio=0.0,
            aspect_delta=1.0,
            contact_shift=1.0,
            source_area=source_area,
            target_area=target_area,
        )
    shift = float(
        np.hypot(target_centroid[0] - source_centroid[0], target_centroid[1] - source_centroid[1]) / diagonal
    )
    area_ratio = target_area / source_area if source_area else 0.0
    aspect_delta = abs(_aspect(target_mask) - _aspect(source_mask))
    contact_shift = abs(_contact(source_mask) - _contact(target_mask)) / max(1.0, float(source_mask.shape[0]))
    return GeometryMeasurements(
        centroid_shift=shift,
        area_ratio=area_ratio,
        aspect_delta=float(aspect_delta),
        contact_shift=float(contact_shift),
        source_area=source_area,
        target_area=target_area,
    )


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def _aspect(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0.0
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    return width / height if height else 0.0


def _contact(mask: np.ndarray) -> float:
    rows = np.nonzero(mask.any(axis=1))[0]
    return float(rows[-1]) if rows.size else 0.0


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Intersection over union of two boolean masks."""
    if left.shape != right.shape:
        raise ValueError("IoU requires masks of identical shape")
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return intersection / union if union else 0.0


def annotation_consistency_score(
    *,
    shape: tuple[int, int],
    bbox: tuple[float, float, float, float] | None,
    mask_area: int,
    polygon_points: int,
    min_area_px: int,
    expected_category_matched: bool,
) -> tuple[float, list[str]]:
    """Score how usable an annotation is, and list the problems found.

    Returns:
        ``(score, problems)`` where ``problems`` are reason-code strings.
    """
    from vidliner.domain.reasons import ReasonCode

    problems: list[str] = []
    height, width = shape
    if bbox is None:
        return 0.0, [ReasonCode.ANNOTATION_EMPTY.value]
    x_min, y_min, x_max, y_max = bbox
    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        problems.append(ReasonCode.ANNOTATION_OUT_OF_BOUNDS.value)
    if mask_area <= 0:
        problems.append(ReasonCode.ANNOTATION_EMPTY.value)
    elif mask_area < min_area_px:
        problems.append(ReasonCode.ANNOTATION_TOO_SMALL.value)
    if polygon_points < 3:
        problems.append(ReasonCode.ANNOTATION_MISMATCH.value)
    if not expected_category_matched:
        problems.append(ReasonCode.ANNOTATION_MISMATCH.value)

    score = 1.0
    if ReasonCode.ANNOTATION_OUT_OF_BOUNDS.value in problems:
        score -= 0.5
    if ReasonCode.ANNOTATION_EMPTY.value in problems:
        score -= 1.0
    if ReasonCode.ANNOTATION_TOO_SMALL.value in problems:
        score -= 0.4
    if ReasonCode.ANNOTATION_MISMATCH.value in problems:
        score -= 0.3
    return float(np.clip(score, 0.0, 1.0)), problems


def similarity_score(value: float, *, tolerance: float = 0.25) -> float:
    """Map a raw similarity in ``[0, 1]`` to a score that treats small deviations as acceptable."""
    return float(np.clip(1.0 - max(0.0, 1.0 - value) / tolerance, 0.0, 1.0))
