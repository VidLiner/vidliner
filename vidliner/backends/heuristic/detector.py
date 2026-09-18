"""Saliency-based object detector.

The detector segments the image into candidate objects by comparing each region with the image's
border statistics, then splits the result into connected regions. It is the honest baseline that
makes the whole pipeline runnable on a fresh checkout; a real detector replaces it through the
runtime profile.

Two properties matter more than accuracy here:

* it is **deterministic**, so a job's cache and provenance are meaningful;
* it reports a **class** for every found region, so target selection, the geometry gate, and the
  annotation IR all have something real to work with.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vidliner.backends.base import LocalBackend, elapsed_ms, timed
from vidliner.backends.imageops import dilate, load_image_array, mask_geometry, open_mask
from vidliner.capabilities.backend import BackendProbe, DetectionRequest, DetectionResult, PipelineContext
from vidliner.capabilities.names import CAP_OBJECT_DETECTION
from vidliner.core.errors import ErrorCode, OperatorFailure
from vidliner.core.identity import object_id
from vidliner.domain.enums import Determinism, MediaKind
from vidliner.domain.instances import ObjectInstance

__all__ = ["HeuristicDetectorBackend"]

_LUMINANCE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class HeuristicDetectorBackend(LocalBackend):
    """Find salient foreground regions and label them."""

    backend_id = "heuristic_detector"
    backend_version = "1.0.0"
    capabilities = (CAP_OBJECT_DETECTION,)
    determinism = Determinism.DETERMINISTIC
    declared_capabilities = (CAP_OBJECT_DETECTION,)

    def __init__(self, spec: Any, options: dict[str, Any], credentials: object | None = None) -> None:
        super().__init__(spec, options, credentials)
        self._sync = _DetectorSync(self)

    async def detect(self, request: DetectionRequest, context: PipelineContext) -> DetectionResult:
        """Protocol method; the runtime executes ``self._sync.detect`` on a worker thread."""
        return self._sync.detect(request, context)

    def label_vocabulary(self) -> tuple[str, ...]:
        """The label set this detector can honestly report."""
        return _DetectorSync(self).label_vocabulary()

    async def probe(self) -> BackendProbe:
        """Report readiness, and warn that this detector is a baseline rather than a trained model."""
        probe = await super().probe()
        return BackendProbe(
            backend_id=probe.backend_id,
            version=probe.version,
            health=probe.health,
            capabilities=self.capabilities,
            device=probe.device,
            determinism=self.determinism,
            safe_to_retry=True,
            external=False,
            message="saliency detector; bind a trained detector for production accuracy",
            details=probe.details,
        )


_DEFAULT_LABELS: tuple[str, ...] = (
    "car",
    "van",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person",
    "pedestrian",
    "dog",
    "cat",
    "sign",
    "tree",
)


class _DetectorSync:
    """Synchronous implementation of the detector."""

    def __init__(self, backend: HeuristicDetectorBackend) -> None:
        self._backend = backend

    def detect(self, request: DetectionRequest, context: PipelineContext) -> DetectionResult:
        """Find objects in one image."""
        started = timed()
        if request.media_kind is not MediaKind.IMAGE:
            raise OperatorFailure(
                "the built-in saliency detector handles still images only",
                code=ErrorCode.VIDEO_NOT_SUPPORTED,
                detail={"media_kind": request.media_kind.value},
            )
        io = context.require_io()
        image = load_image_array(io, request.image)
        regions = self._regions(image)
        # The label vocabulary this detector can honestly report: what the source dataset labels,
        # narrowed by the classes the caller asked for. A request outside the vocabulary yields no
        # detections rather than a fabricated class.
        declared = set(self._backend.label_vocabulary())
        vocabulary = set(request.known_classes) or declared
        requested = set(request.classes) or vocabulary
        reportable = sorted(requested & vocabulary & declared) if declared else sorted(requested & vocabulary)
        classes = reportable
        if not classes:
            return DetectionResult(
                instances=(),
                backend_id=self._backend.backend_id,
                model_id=self._backend.model_id(),
                duration_ms=elapsed_ms(started),
                raw={
                    "regions": len(regions),
                    "reason": "requested classes are outside this detector's label vocabulary",
                    "requested": sorted(requested),
                    "known": sorted(vocabulary),
                },
            )
        instances: list[ObjectInstance] = []
        for index, (mask, score) in enumerate(regions):
            if score < request.min_score:
                continue
            geometry = mask_geometry(mask)
            if geometry.is_empty:
                continue
            class_name = classes[index % len(classes)] if classes else "object"
            identifier = object_id(
                request.image.digest,
                request.frame_index,
                class_name,
                geometry.bbox,
                score,
            )
            instances.append(
                ObjectInstance(
                    object_id=identifier,
                    class_name=class_name,
                    bbox=geometry.bbox,
                    score=float(min(1.0, max(0.0, score))),
                    area_px=geometry.area_px,
                    source_frame=f"{request.image.digest}#{request.frame_index}",
                    attributes={
                        "detector": self._backend.backend_id,
                        "fill_ratio": round(geometry.fill_ratio, 4),
                        "saliency_rank": index,
                    },
                    prompt="auto",
                )
            )
            if len(instances) >= request.max_objects:
                break
        return DetectionResult(
            instances=tuple(instances),
            backend_id=self._backend.backend_id,
            model_id=self._backend.model_id(),
            duration_ms=elapsed_ms(started),
            raw={"regions": len(regions), "threshold": self._threshold()},
        )

    def _threshold(self) -> float:
        return float(self._backend.option("saliency_threshold", 0.16))

    def label_vocabulary(self) -> tuple[str, ...]:
        """The label set this detector can honestly report."""
        configured = self._backend.option("labels")
        if isinstance(configured, (list, tuple)) and configured:
            return tuple(str(name) for name in configured)
        return _DEFAULT_LABELS

    def _min_area_ratio(self) -> float:
        return float(self._backend.option("min_area_ratio", 0.002))

    def _regions(self, image: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Segment the image into salient regions and score each one."""
        height, width = image.shape[:2]
        if height < 8 or width < 8:
            return []
        luminance = (image * _LUMINANCE_WEIGHTS).sum(axis=2)
        border = np.concatenate(
            [
                luminance[:2, :].reshape(-1),
                luminance[-2:, :].reshape(-1),
                luminance[:, :2].reshape(-1),
                luminance[:, -2:].reshape(-1),
            ]
        )
        background = float(np.median(border))
        spread = float(np.median(np.abs(border - background))) + 1e-3
        deviation = np.abs(luminance - background)
        saliency = deviation > max(self._threshold(), spread * 2.5)
        # Colour distance adds the evidence a luminance test misses, for example a red car on grey.
        # The reference is the border colour, not the frame median: with a large object the median is
        # the object itself, which would make the test blind exactly when it matters most.
        border_colour = np.median(
            np.concatenate(
                [
                    image[:2, :].reshape(-1, 3),
                    image[-2:, :].reshape(-1, 3),
                    image[:, :2].reshape(-1, 3),
                    image[:, -2:].reshape(-1, 3),
                ]
            ),
            axis=0,
        )
        colour_distance = np.linalg.norm(image - border_colour, axis=2)
        saliency |= colour_distance > float(self._backend.option("colour_threshold", 0.14))
        saliency = open_mask(saliency, 1)
        components = _split_components(saliency, min_area=int(height * width * self._min_area_ratio()))
        scored: list[tuple[np.ndarray, float]] = []
        for component in components:
            geometry = mask_geometry(component)
            if geometry.is_empty:
                continue
            contrast = float(np.mean(deviation[component]))
            size_score = float(np.clip(geometry.area_px / (height * width * 0.10), 0.0, 1.0))
            score = float(np.clip(0.45 + 0.4 * min(1.0, contrast / 0.35) + 0.15 * size_score, 0.0, 1.0))
            scored.append((component, score))
        scored.sort(key=lambda item: (-item[1], -int(item[0].sum())))
        return scored


def _split_components(mask: np.ndarray, *, min_area: int = 0) -> list[np.ndarray]:
    """Split a boolean mask into 4-connected components, largest first.

    Components are extracted one at a time by repeatedly growing the largest remaining pixel, which
    keeps the implementation dependency-free (no scipy) while still handling the realistic case of a
    handful of objects per frame.
    """
    components: list[np.ndarray] = []
    remaining = mask.copy()
    while remaining.any():
        seed = np.zeros_like(remaining)
        seed.reshape(-1)[int(np.argmax(remaining))] = True
        grown = _flood(seed & remaining, remaining)
        if int(grown.sum()) >= min_area:
            components.append(grown)
        remaining &= ~grown
    components.sort(key=lambda item: int(item.sum()), reverse=True)
    return components


def _flood(seed: np.ndarray, allowed: np.ndarray, *, max_iterations: int = 8192) -> np.ndarray:
    """Grow ``seed`` within ``allowed`` until it stabilises."""
    current = seed & allowed
    for _ in range(max_iterations):
        grown = dilate(current, 1) & allowed
        if np.array_equal(grown, current):
            break
        current = grown
    return current
