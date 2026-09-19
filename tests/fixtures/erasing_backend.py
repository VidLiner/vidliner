"""A deliberately broken replacement backend, used to prove the verification loop closes.

The backend removes the object instead of replacing it: it fills the masked region with the median
colour of the frame's border. Its output is a perfectly plausible image — the scene looks clean, the
background is byte-identical outside the mask, and the files are valid PNGs. What it is *not* is a
training sample: there is no object where the label says there is one.

This is the failure an open pipeline cannot see. Comparing the generated image against the mask that
was handed to the generator reports a perfect score, because the mask is exactly what was edited. Only
re-detecting the object in the output catches it, which is what ``verify.redetect`` does.

The module lives in ``tests/`` on purpose: it is a test double, not a shipped backend, and it exists
so that the property "an object must be found in the generated image" is asserted end to end rather
than only in a unit test with a stub.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vidliner.backends.base import LocalBackend, elapsed_ms, timed
from vidliner.backends.imageops import load_image_array, save_image_array
from vidliner.capabilities.backend import BackendProbe, GenerationOutcome, GenerationRequest, PipelineContext
from vidliner.capabilities.names import CAP_OBJECT_REPLACEMENT
from vidliner.core.errors import ErrorCode, OperatorFailure
from vidliner.domain.enums import ArtifactKind, Determinism

__all__ = ["ErasingReplacementBackend"]


class ErasingReplacementBackend(LocalBackend):
    """Fill the target region with background colour, removing the object."""

    backend_id = "erasing_replacement"
    backend_version = "1.0.0"
    # A test double is the definition of a demonstration stand-in: it produces a plausible picture
    # with no object where the label says there is one. Declaring that means the production guard
    # refuses to export a dataset built on it unless the recipe says, explicitly, that it knows.
    demo_only = True
    capabilities = (CAP_OBJECT_REPLACEMENT,)
    determinism = Determinism.DETERMINISTIC
    safe_to_retry = True
    declared_capabilities = (CAP_OBJECT_REPLACEMENT,)

    def __init__(self, spec: Any, options: dict[str, Any], credentials: object | None = None) -> None:
        super().__init__(spec, options, credentials)
        self._sync = _ErasingSync(self)

    async def replace(self, request: GenerationRequest, context: PipelineContext) -> GenerationOutcome:
        """Protocol method; the runtime executes ``self._sync.replace`` on a worker thread."""
        return self._sync.replace(request, context)

    async def probe(self) -> BackendProbe:
        """Report readiness, and be explicit that this backend is a test double."""
        probe = await super().probe()
        return BackendProbe(
            backend_id=probe.backend_id,
            version=probe.version,
            health="degraded",
            capabilities=self.capabilities,
            device=probe.device,
            determinism=self.determinism,
            safe_to_retry=True,
            external=False,
            message="test double: erases the target object instead of replacing it",
            details={"test_double": True, **probe.details},
        )


class _ErasingSync:
    """Synchronous implementation of the erasing generator."""

    def __init__(self, backend: ErasingReplacementBackend) -> None:
        self._backend = backend

    def replace(self, request: GenerationRequest, context: PipelineContext) -> GenerationOutcome:
        """Return the source frame with the masked region painted over."""
        started = timed()
        io = context.require_io()
        source = load_image_array(io, request.source_image)
        mask = io.load_mask(request.target_mask)
        if not mask.any():
            raise OperatorFailure("cannot erase an empty mask region", code=ErrorCode.MASK_EMPTY)
        if mask.shape[:2] != source.shape[:2]:
            raise OperatorFailure(
                "target mask shape does not match the source image",
                code=ErrorCode.MASK_SHAPE_MISMATCH,
                detail={"mask": list(mask.shape), "image": list(source.shape[:2])},
            )
        border = np.concatenate(
            [
                source[:2, :].reshape(-1, 3),
                source[-2:, :].reshape(-1, 3),
                source[:, :2].reshape(-1, 3),
                source[:, -2:].reshape(-1, 3),
            ]
        )
        background = np.median(border, axis=0).astype(np.float32)
        produced = source.copy()
        produced[mask] = background
        artifact = save_image_array(io, produced, kind=ArtifactKind.CANDIDATE)
        return GenerationOutcome(
            image=artifact,
            backend_id=self._backend.backend_id,
            model_id="erasing-test-double",
            seed=request.seed,
            duration_ms=elapsed_ms(started),
            cost_estimate=0.0,
            currency="USD",
            external=False,
            raw_metadata={"test_double": True, "erased": True, "mask_area_px": int(mask.sum())},
            applied_parameters={"intensity": request.intensity},
        )
