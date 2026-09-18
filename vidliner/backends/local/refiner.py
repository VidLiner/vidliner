"""Local refinement: mask cleanup, alpha blending, and colour harmonisation.

Refinement is deliberately separate from generation. A generative model's raw output is composited
and matched against its surroundings by operators that can be replaced, measured, and switched off
independently. Three effects are implemented here, each with a measurable consequence:

* **mask edges** — remove speckle, close pinholes, feather the boundary;
* **alpha blend** — composite the generated region into the untouched source using the feathered
  matte, so nothing outside the mask can change;
* **harmonise** — match the region's mean colour and illumination to an annulus around it, at a
  configurable strength.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vidliner.backends.base import LocalBackend, elapsed_ms, timed
from vidliner.backends.imageops import (
    close,
    compose_over,
    dilate,
    feather,
    largest_component,
    load_image_array,
    open_mask,
    save_image_array,
)
from vidliner.capabilities.backend import (
    BackendProbe,
    CompositeResult,
    CompositingRequest,
    HarmonizationRequest,
    MaskRefinementRequest,
    MaskRefinementResult,
    PipelineContext,
)
from vidliner.capabilities.names import (
    CAP_IMAGE_COMPOSITING,
    CAP_IMAGE_HARMONIZATION,
    CAP_MASK_REFINEMENT,
)
from vidliner.core.errors import ErrorCode, OperatorFailure
from vidliner.core.results import ArtifactRef
from vidliner.domain.enums import ArtifactKind, Determinism

__all__ = ["LocalRefinerBackend"]


class LocalRefinerBackend(LocalBackend):
    """Clean masks, blend regions, and harmonise colour, locally and deterministically."""

    backend_id = "local_refiner"
    backend_version = "1.0.0"
    capabilities = (CAP_MASK_REFINEMENT, CAP_IMAGE_COMPOSITING, CAP_IMAGE_HARMONIZATION)
    determinism = Determinism.DETERMINISTIC
    declared_capabilities = (CAP_MASK_REFINEMENT, CAP_IMAGE_COMPOSITING, CAP_IMAGE_HARMONIZATION)

    def __init__(self, spec: Any, options: dict[str, Any], credentials: object | None = None) -> None:
        super().__init__(spec, options, credentials)
        self._sync = _RefinerSync(self)

    async def refine_mask(
        self, request: MaskRefinementRequest, context: PipelineContext
    ) -> MaskRefinementResult:
        """Protocol method; the runtime executes ``self._sync.refine_mask`` on a worker thread."""
        return self._sync.refine_mask(request, context)

    async def composite(self, request: CompositingRequest, context: PipelineContext) -> CompositeResult:
        """Protocol method; the runtime executes ``self._sync.composite`` on a worker thread."""
        return self._sync.composite(request, context)

    async def harmonize(self, request: HarmonizationRequest, context: PipelineContext) -> ArtifactRef:
        """Protocol method; the runtime executes ``self._sync.harmonize`` on a worker thread."""
        return self._sync.harmonize(request, context)

    async def probe(self) -> BackendProbe:
        """Report readiness."""
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
            message="deterministic mask, blend, and colour operations",
            details=probe.details,
        )


class _RefinerSync:
    """Synchronous implementations of the refinement operations."""

    def __init__(self, backend: LocalRefinerBackend) -> None:
        self._backend = backend

    def refine_mask(self, request: MaskRefinementRequest, context: PipelineContext) -> MaskRefinementResult:
        """Clean and feather a mask."""
        started = timed()
        io = context.require_io()
        mask = io.load_mask(request.mask)
        if mask.shape != (request.shape.height, request.shape.width):
            raise OperatorFailure(
                "mask shape does not match the declared image shape",
                code=ErrorCode.MASK_SHAPE_MISMATCH,
                detail={"mask": list(mask.shape), "declared": [request.shape.height, request.shape.width]},
            )
        before = int(mask.sum())
        refined = mask.copy()
        close_px = int(request.close_px)
        if close_px:
            refined = close(refined, close_px)
        refined = open_mask(refined, 1)
        refined = largest_component(refined)
        dilate_px = int(request.dilate_px)
        if dilate_px:
            refined = dilate(refined, dilate_px)
        feather_px = int(request.feather_px)
        if feather_px:
            alpha = feather(refined, feather_px)
            refined = alpha >= float(self._backend.option("alpha_cutoff", 0.35))
        if not refined.any():
            raise OperatorFailure(
                "mask refinement removed every pixel; refusing to continue with an empty mask",
                code=ErrorCode.MASK_EMPTY,
                detail={"before": before},
            )
        artifact = io.save_mask(refined).with_kind(ArtifactKind.MASK)
        return MaskRefinementResult(
            mask=artifact,
            area_px=int(refined.sum()),
            area_delta_px=int(refined.sum()) - before,
            backend_id=self._backend.backend_id,
            duration_ms=elapsed_ms(started),
        )

    def composite(self, request: CompositingRequest, context: PipelineContext) -> CompositeResult:
        """Blend a generated image into the source through the mask matte."""
        started = timed()
        io = context.require_io()
        source = load_image_array(io, request.source_image)
        generated = load_image_array(io, request.candidate_image)
        mask = io.load_mask(request.mask)
        if generated.shape != source.shape:
            raise OperatorFailure(
                "generated image shape does not match the source image",
                code=ErrorCode.MEDIA_CORRUPT,
                detail={"source": list(source.shape), "generated": list(generated.shape)},
            )
        if mask.shape != source.shape[:2]:
            raise OperatorFailure(
                "mask shape does not match the source image",
                code=ErrorCode.MASK_SHAPE_MISMATCH,
                detail={"mask": list(mask.shape), "source": list(source.shape[:2])},
            )
        alpha = feather(mask, max(0, request.feather_px)) if request.feather_px else mask.astype(np.float32)
        composited = compose_over(source, generated, alpha)
        unchanged_outside = np.abs(composited - source).max(axis=2) < 1.0 / 255.0
        outside = ~dilate(mask, max(1, request.feather_px))
        changed_ratio = 0.0
        if outside.any():
            changed_ratio = float((~unchanged_outside[outside]).mean())
        if request.preserve_outside_mask and changed_ratio > float(
            self._backend.option("outside_tolerance", 0.02)
        ):
            # The blend must not touch anything outside the mask; if it did, restore those pixels
            # from the source rather than reporting a success that the background gate will reject.
            restored = np.where(outside[..., None], source, composited)
            composited = restored
            changed_ratio = 0.0
        artifact = save_image_array(io, composited, kind=ArtifactKind.REFINED_IMAGE)
        return CompositeResult(
            image=artifact,
            changed_ratio_outside_mask=changed_ratio,
            backend_id=self._backend.backend_id,
            duration_ms=elapsed_ms(started),
        )

    def harmonize(self, request: HarmonizationRequest, context: PipelineContext) -> ArtifactRef:
        """Match colour and illumination between a region and its surroundings."""
        started = timed()
        del started
        io = context.require_io()
        image = load_image_array(io, request.image)
        mask = io.load_mask(request.mask)
        if mask.shape != image.shape[:2]:
            raise OperatorFailure(
                "mask shape does not match the image",
                code=ErrorCode.MASK_SHAPE_MISMATCH,
                detail={"mask": list(mask.shape), "image": list(image.shape[:2])},
            )
        strength = float(np.clip(request.strength, 0.0, 1.0))
        if strength <= 0.0 or not mask.any():
            return save_image_array(io, image, kind=ArtifactKind.REFINED_IMAGE)
        ring = dilate(mask, int(self._backend.option("ring_px", 6))) & ~mask
        if not ring.any():
            return save_image_array(io, image, kind=ArtifactKind.REFINED_IMAGE)
        region_mean = image[mask].mean(axis=0)
        ring_mean = image[ring].mean(axis=0)
        region_luminance = float(_luminance(image[mask]).mean())
        ring_luminance = float(_luminance(image[ring]).mean())
        gain = np.ones(3, dtype=np.float32)
        if region_luminance > 1e-3:
            luminance_gain = float(np.clip(ring_luminance / region_luminance, 0.6, 1.6))
            gain *= luminance_gain
        colour_shift = (ring_mean - region_mean) * 0.5
        adjusted = np.clip(image * gain[None, None, :] + colour_shift[None, None, :] * strength, 0.0, 1.0)
        blended = np.where(mask[..., None], adjusted, image)
        return save_image_array(io, blended.astype(np.float32), kind=ArtifactKind.REFINED_IMAGE)


def _luminance(pixels: np.ndarray) -> np.ndarray:
    return (pixels * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=-1)
