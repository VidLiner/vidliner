"""Box-prompted segmentation.

Given a bounding box (or a point, or a text hint) the segmenter grows a pixel-level mask that
respects image edges: it scores every pixel by how far it is from the prompt region and how similar
it is to the prompt's colour distribution, then keeps the dominant connected region.

The mask is stored as a PNG artifact and returned as a domain :class:`~vidliner.domain.masks.MaskRef`
with its area already measured, so downstream operators can reject an empty or implausible mask
without loading pixels again.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vidliner.backends.base import LocalBackend, elapsed_ms, timed
from vidliner.backends.imageops import (
    close,
    dilate,
    erode,
    largest_component,
    load_image_array,
    mask_polygon,
    open_mask,
)
from vidliner.capabilities.backend import (
    BackendProbe,
    PipelineContext,
    SegmentationRequest,
    SegmentationResult,
)
from vidliner.capabilities.names import CAP_INSTANCE_SEGMENTATION
from vidliner.core.errors import ErrorCode, OperatorFailure
from vidliner.domain.enums import ArtifactKind, Determinism
from vidliner.domain.shapes import BoundingBox, ImageShape

__all__ = ["HeuristicSegmentationBackend"]


class HeuristicSegmentationBackend(LocalBackend):
    """Grow a mask from a prompt using colour similarity and edge awareness."""

    backend_id = "heuristic_segmenter"
    backend_version = "1.0.0"
    capabilities = (CAP_INSTANCE_SEGMENTATION,)
    determinism = Determinism.DETERMINISTIC
    declared_capabilities = (CAP_INSTANCE_SEGMENTATION,)

    def __init__(self, spec: Any, options: dict[str, Any], credentials: object | None = None) -> None:
        super().__init__(spec, options, credentials)
        self._sync = _SegmentationSync(self)

    async def segment(self, request: SegmentationRequest, context: PipelineContext) -> SegmentationResult:
        """Protocol method; the runtime executes ``self._sync.segment`` on a worker thread."""
        return self._sync.segment(request, context)

    async def probe(self) -> BackendProbe:
        """Report readiness, noting that text prompts are unsupported by this backend."""
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
            message="box/point prompt segmenter; text prompts require a different backend",
            details=probe.details,
        )


class _SegmentationSync:
    """Synchronous implementation of the segmenter."""

    def __init__(self, backend: HeuristicSegmentationBackend) -> None:
        self._backend = backend

    def segment(self, request: SegmentationRequest, context: PipelineContext) -> SegmentationResult:
        """Produce a mask for the prompted object."""
        started = timed()
        if request.text is not None and request.bbox is None and request.point is None:
            raise OperatorFailure(
                f"backend {self._backend.backend_id} cannot segment from a text prompt alone",
                code=ErrorCode.OPERATOR_NOT_IMPLEMENTED,
                detail={"backend": self._backend.backend_id, "text": request.text},
            )
        if request.bbox is None and request.point is None:
            raise OperatorFailure(
                "segmentation requires a bounding box or point prompt",
                code=ErrorCode.OPERATOR_CONFIG_INVALID,
            )
        io = context.require_io()
        image = load_image_array(io, request.image)
        height, width = image.shape[:2]
        shape = ImageShape(width=width, height=height)

        seed = _seed_mask(request, shape)
        if seed.sum() == 0:
            raise OperatorFailure(
                "the segmentation prompt selects no pixels inside the image",
                code=ErrorCode.MASK_EMPTY,
                detail={"bbox": request.bbox, "point": request.point},
            )
        mask = self._grow(image, seed)
        mask = close(mask, int(self._backend.option("close_px", 1)))
        mask = open_mask(mask, 1)
        mask = largest_component(mask)
        dilate_px = int(self._backend.option("dilate_px", 0))
        if dilate_px:
            mask = dilate(mask, dilate_px)
        erode_px = int(self._backend.option("erode_px", 0))
        if erode_px:
            mask = erode(mask, erode_px)
        area = int(mask.sum())
        if area == 0:
            raise OperatorFailure("segmentation produced an empty mask", code=ErrorCode.MASK_EMPTY)

        artifact = io.save_mask(mask)
        polygon = mask_polygon(mask, max_points=int(self._backend.option("max_polygon_points", 96)))
        seed_area = int(seed.sum())
        stability = float(np.clip(area / max(seed_area, 1) / 4.0, 0.0, 1.0))
        score = float(np.clip(0.55 + 0.35 * stability, 0.0, 1.0))
        return SegmentationResult(
            mask=artifact.with_kind(ArtifactKind.MASK),
            shape=shape,
            area_px=area,
            polygon=polygon,
            score=score,
            backend_id=self._backend.backend_id,
            duration_ms=elapsed_ms(started),
        )

    def _grow(self, image: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Expand the seed region while pixel colour stays close to the seed's own distribution."""
        seed_values = image[seed]
        mean = seed_values.mean(axis=0)
        spread = seed_values.std(axis=0) + 1e-3
        tolerance = float(self._backend.option("colour_tolerance", 2.4))
        distance = np.linalg.norm((image - mean) / np.maximum(spread, 0.05), axis=2)
        similar = distance <= tolerance
        border_guard = float(self._backend.option("border_guard", 0.35))
        border_values = np.concatenate(
            [
                image[:1, :].reshape(-1, 3),
                image[-1:, :].reshape(-1, 3),
                image[:, :1].reshape(-1, 3),
                image[:, -1:].reshape(-1, 3),
            ]
        )
        border_mean = border_values.mean(axis=0)
        background_like = np.linalg.norm(image - border_mean, axis=2) < border_guard
        candidate = similar & ~background_like
        candidate |= seed
        grown = _flood_from(candidate, seed)
        if int(grown.sum()) < int(seed.sum()):
            return seed
        return grown


def _seed_mask(request: SegmentationRequest, shape: ImageShape) -> np.ndarray:
    mask = np.zeros((shape.height, shape.width), dtype=bool)
    if request.bbox is not None:
        x0, y0, x1, y1 = request.bbox
        box = BoundingBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1).clipped(shape)
        mask[
            int(np.floor(box.y_min)) : int(np.ceil(box.y_max)),
            int(np.floor(box.x_min)) : int(np.ceil(box.x_max)),
        ] = True
    if request.point is not None:
        x, y = request.point
        column = int(np.clip(round(x), 0, shape.width - 1))
        row = int(np.clip(round(y), 0, shape.height - 1))
        mask[row, column] = True
    return mask


def _flood_from(allowed: np.ndarray, seed: np.ndarray, *, max_iterations: int = 4096) -> np.ndarray:
    """Grow ``seed`` within ``allowed`` until it stabilises."""
    current = seed & allowed
    for _ in range(max_iterations):
        grown = dilate(current, 1) & allowed
        if np.array_equal(grown, current):
            break
        current = grown
    return current
