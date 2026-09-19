"""Perceptual fingerprint backend.

Duplicate control must work in every deployment, including one with no embedding model, so the
default binding for ``quality.embedding.v1`` is a perceptual hash exposed through the embedding
protocol. The returned "vector" is the 64 hash bits as floats, which makes cosine similarity between
two fingerprints equivalent to agreement between their bits; a learned embedding backend can be bound
instead and the duplicate stage keeps working, because it only ever calls
:meth:`EmbeddingResult.cosine`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from vidliner.backends.base import LocalBackend
from vidliner.backends.imageops import load_image_array
from vidliner.capabilities.backend import BackendProbe, EmbeddingRequest, EmbeddingResult, PipelineContext
from vidliner.capabilities.names import CAP_QUALITY_EMBEDDING
from vidliner.domain.enums import Determinism
from vidliner.quality.fingerprint import perceptual_hash

__all__ = ["PerceptualHashBackend"]


class PerceptualHashBackend(LocalBackend):
    """Expose perceptual hashing through the embedding protocol."""

    backend_id = "perceptual_hash"
    demo_only = True
    backend_version = "1.0.0"
    capabilities = (CAP_QUALITY_EMBEDDING,)
    determinism = Determinism.DETERMINISTIC
    declared_capabilities = (CAP_QUALITY_EMBEDDING,)

    def __init__(self, spec: Any, options: dict[str, Any], credentials: object | None = None) -> None:
        super().__init__(spec, options, credentials)
        self._sync = _FingerprintSync(self)

    async def embed(self, request: EmbeddingRequest, context: PipelineContext) -> EmbeddingResult:
        """Protocol method; the runtime executes ``self._sync.embed`` on a worker thread."""
        return self._sync.embed(request, context)

    async def probe(self) -> BackendProbe:
        """Report readiness, deriving from the base probe so the demo declaration is preserved."""
        probe = await super().probe()
        return replace(
            probe,
            message="64-bit perceptual hash exposed as an embedding",
            details={"algorithm": "phash", **probe.details},
        )

    def hash_hex(self, request: EmbeddingRequest, context: PipelineContext) -> str:
        """The hexadecimal fingerprint of an image, for storage in the sample table."""
        result = self._sync.embed(request, context)
        bits = "".join("1" if value > 0.5 else "0" for value in result.vector)
        return f"{int(bits, 2):016x}"


class _FingerprintSync:
    """Synchronous implementation of the fingerprint backend."""

    def __init__(self, backend: PerceptualHashBackend) -> None:
        self._backend = backend

    def embed(self, request: EmbeddingRequest, context: PipelineContext) -> EmbeddingResult:
        """Return the 64 hash bits of the image (or of a cropped region) as a vector."""
        io = context.require_io()
        image = load_image_array(io, request.image)
        if request.crop_bbox is not None:
            x0, y0, x1, y1 = request.crop_bbox
            height, width = image.shape[:2]
            crop = image[
                max(0, int(y0)) : min(height, int(np.ceil(y1))),
                max(0, int(x0)) : min(width, int(np.ceil(x1))),
            ]
            if crop.size:
                image = crop
        algorithm = str(self._backend.option("algorithm", "phash"))
        fingerprint = perceptual_hash(image, algorithm=algorithm)
        vector = tuple(float(bit) for bit in fingerprint.bits)
        return EmbeddingResult(
            vector=vector,
            backend_id=self._backend.backend_id,
            model_id=f"{self._backend.model_id()}:{algorithm}",
            dimension=len(vector),
        )
