"""Deterministic stand-in replacement generator.

This backend exists so the whole pipeline — generation included — can run and be tested without a
paid API or a downloaded model. It is a *real* implementation of the capability contract, not a
stub: it produces an image, reports its own metadata, honours the seed, and never claims more than
it did.

Because it cannot invent a plausible car, it does the two things it can do honestly:

1. it replaces the masked region with a seed-derived, category-conditioned appearance (a flat
   colour derived from the category name plus a deterministic low-frequency texture); and
2. it reports ``synthetic=true`` in ``raw_metadata``.

The consequence is important and intended: the quality gates will accept these candidates only when
the measured semantics, background preservation, geometry, and annotation consistency genuinely
hold. A mis-wired pipeline fails its own quality gate rather than producing a plausible-looking
dataset.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vidliner.backends.base import LocalBackend, elapsed_ms, timed
from vidliner.backends.imageops import dilate, load_image_array, save_image_array
from vidliner.capabilities.backend import BackendProbe, GenerationOutcome, GenerationRequest, PipelineContext
from vidliner.capabilities.names import CAP_OBJECT_REPLACEMENT
from vidliner.core.errors import ErrorCode, OperatorFailure
from vidliner.domain.enums import ArtifactKind, Determinism

__all__ = ["FakeReplacementBackend"]

#: Category → base colour, so a "sedan" candidate and an "suv" candidate differ visibly. Categories
#: that are not listed get a colour derived from their name, which keeps the mapping deterministic
#: without pretending to know what an unseen word looks like.
_CATEGORY_COLOURS: dict[str, tuple[float, float, float]] = {
    "sedan": (0.72, 0.16, 0.14),
    "suv": (0.86, 0.87, 0.88),
    "pickup": (0.16, 0.24, 0.52),
    "hatchback": (0.24, 0.52, 0.32),
    "van": (0.90, 0.78, 0.20),
    "truck": (0.32, 0.32, 0.34),
    "bus": (0.94, 0.62, 0.12),
    "motorcycle": (0.12, 0.12, 0.14),
    "bicycle": (0.20, 0.60, 0.70),
    "person": (0.78, 0.62, 0.50),
    "pedestrian": (0.44, 0.44, 0.60),
    "dog": (0.62, 0.46, 0.28),
    "cat": (0.70, 0.66, 0.62),
    "sign": (0.90, 0.90, 0.20),
    "tree": (0.20, 0.42, 0.20),
}


class FakeReplacementBackend(LocalBackend):
    """Produce a deterministic synthetic replacement inside the target mask."""

    backend_id = "fake_replacement"
    backend_version = "1.0.0"
    capabilities = (CAP_OBJECT_REPLACEMENT,)
    determinism = Determinism.SEEDED
    declared_capabilities = (CAP_OBJECT_REPLACEMENT,)

    def __init__(self, spec: Any, options: dict[str, Any], credentials: object | None = None) -> None:
        super().__init__(spec, options, credentials)
        self._sync = _FakeReplacementSync(self)

    async def replace(self, request: GenerationRequest, context: PipelineContext) -> GenerationOutcome:
        """Protocol method; the runtime executes ``self._sync.replace`` on a worker thread."""
        return self._sync.replace(request, context)

    async def probe(self) -> BackendProbe:
        """Report readiness and be explicit that this generator is synthetic."""
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
            message="synthetic generator: produces deterministic placeholder imagery, not photoreal edits",
            details={"synthetic": True, **probe.details},
        )


class _FakeReplacementSync:
    """Synchronous implementation of the stand-in generator."""

    def __init__(self, backend: FakeReplacementBackend) -> None:
        self._backend = backend

    def replace(self, request: GenerationRequest, context: PipelineContext) -> GenerationOutcome:
        """Render a replacement region and composite it into the source image."""
        started = timed()
        io = context.require_io()
        source = load_image_array(io, request.source_image)
        mask = io.load_mask(request.target_mask)
        if mask.shape[:2] != source.shape[:2]:
            raise OperatorFailure(
                "target mask shape does not match the source image",
                code=ErrorCode.MASK_SHAPE_MISMATCH,
                detail={"mask": list(mask.shape), "image": list(source.shape[:2])},
            )
        if not mask.any():
            raise OperatorFailure("cannot replace an empty mask region", code=ErrorCode.MASK_EMPTY)

        rng = np.random.default_rng(request.seed)
        colour = _category_colour(request.expected_category)
        region = dilate(mask, int(self._backend.option("edge_soften_px", 1)))
        produced = source.copy()
        shade = _shade(colour, request.seed, spread=float(self._backend.option("shade_spread", 0.12)))
        texture = _low_frequency_texture(
            rng, source.shape[:2], scale=int(self._backend.option("texture_scale", 48))
        )
        lighting = _source_lighting(source, mask)
        rendered = np.clip(
            shade[None, None, :] * lighting[..., None] * (0.85 + 0.3 * texture[..., None]),
            0.0,
            1.0,
        ).astype(np.float32)
        produced[region] = rendered[region]

        artifact = save_image_array(io, produced, kind=ArtifactKind.CANDIDATE)
        return GenerationOutcome(
            image=artifact,
            backend_id=self._backend.backend_id,
            model_id=self._backend.model_id(),
            seed=request.seed,
            duration_ms=elapsed_ms(started),
            cost_estimate=0.0,
            currency="USD",
            external=False,
            raw_metadata={
                "synthetic": True,
                "category": request.expected_category,
                "replacement_description": request.replacement_description,
                "prompt": request.positive_prompt,
                "negatives": list(request.negative_constraints),
                "mask_area_px": int(mask.sum()),
                "colour": [round(channel, 4) for channel in shade.tolist()],
                "honoured_parameters": ["seed", "expected_category", "target_mask"],
                "ignored_parameters": _ignored(request),
            },
            applied_parameters={"intensity": request.intensity},
        )


def _category_colour(category: str) -> np.ndarray:
    key = category.strip().lower()
    if key in _CATEGORY_COLOURS:
        return np.array(_CATEGORY_COLOURS[key], dtype=np.float32)
    digest = 0
    for char in key:
        digest = (digest * 131 + ord(char)) % 1000003
    hue = digest % 360
    return _hsv_to_rgb(hue / 360.0, 0.45, 0.62)


def _shade(colour: np.ndarray, seed: int, *, spread: float) -> np.ndarray:
    """Perturb a base colour deterministically from the seed, so candidates differ but repeat."""
    rng = np.random.default_rng(seed ^ 0x5EED)
    delta = rng.uniform(-spread, spread, size=3)
    return np.clip(colour + delta, 0.03, 0.97).astype(np.float32)


def _low_frequency_texture(rng: np.random.Generator, shape: tuple[int, int], *, scale: int) -> np.ndarray:
    """Coarse smooth noise in ``[0, 1]``, cheap and deterministic."""
    height, width = shape
    coarse_height = max(2, height // max(2, scale))
    coarse_width = max(2, width // max(2, scale))
    coarse = rng.random((coarse_height, coarse_width)).astype(np.float32)
    rows = (np.arange(height) * coarse_height / height).astype(int).clip(0, coarse_height - 1)
    columns = (np.arange(width) * coarse_width / width).astype(int).clip(0, coarse_width - 1)
    return coarse[rows][:, columns]


def _source_lighting(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A blurred luminance field of the source region, so the replacement inherits its shading."""
    luminance = (source * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=2)
    blurred = _box_blur(luminance, 9)
    mean = float(blurred[mask].mean()) if mask.any() else 0.5
    if mean <= 1e-3:
        return np.ones_like(blurred)
    return np.clip(blurred / mean, 0.55, 1.45).astype(np.float32)


def _box_blur(values: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    height, width = values.shape
    total = (
        integral[window : window + height, window : window + width]
        - integral[0:height, window : window + width]
        - integral[window : window + height, 0:width]
        + integral[0:height, 0:width]
    )
    return total / float(window * window)


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> np.ndarray:
    index = int(hue * 6.0) % 6
    fraction = hue * 6.0 - int(hue * 6.0)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    table = [
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q),
    ]
    return np.array(table[index], dtype=np.float32)


def _ignored(request: GenerationRequest) -> list[str]:
    """Parameters this generator cannot honour, reported rather than silently dropped."""
    ignored = ["reference_image"] if request.reference_image is not None else []
    if request.context_crop is not None:
        ignored.append("context_crop")
    return ignored
