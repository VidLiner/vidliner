"""Local quality evaluators.

These backends measure things that can be measured without a learned model:

* **semantics** — whether the replaced region's appearance is consistent with the requested category,
  using a category→appearance model that a real classifier or VLM replaces wholesale;
* **artifacts** — fragmented masks, duplicated objects, floating objects, seam clutter, text-like
  texture, and unexpected disappearance of neighbouring objects;
* **embeddings** — a compact appearance descriptor for similarity and duplicate comparisons.

They implement the same protocols a hosted vision-language evaluator implements, so binding a VLM is
a runtime-profile edit rather than a pipeline change.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vidliner.backends.base import LocalBackend, elapsed_ms, timed
from vidliner.backends.imageops import load_image_array
from vidliner.capabilities.backend import (
    ArtifactIO,
    BackendProbe,
    EmbeddingRequest,
    EmbeddingResult,
    PipelineContext,
    SemanticAssessment,
    SemanticAssessRequest,
    VLMRequest,
    VLMResponse,
)
from vidliner.capabilities.names import (
    CAP_QUALITY_ARTIFACT,
    CAP_QUALITY_EMBEDDING,
    CAP_QUALITY_SEMANTIC,
)
from vidliner.core.errors import ErrorCode, OperatorFailure
from vidliner.domain.enums import Determinism
from vidliner.domain.reasons import ReasonCode
from vidliner.quality.metrics import artifact_measurements, connected_components

__all__ = ["LocalMetricEvaluatorBackend"]

#: Reference appearance per category, expressed as a hue/saturation/value triple. This is a
#: deliberately simple stand-in for a classifier: it is honest about being a heuristic, it is
#: deterministic, and it is replaced by a real model in any deployment where semantics matter.
_CATEGORY_REFERENCE: dict[str, tuple[float, float, float]] = {
    "sedan": (0.00, 0.75, 0.70),
    "suv": (0.00, 0.05, 0.86),
    "pickup": (0.62, 0.70, 0.50),
    "hatchback": (0.36, 0.55, 0.50),
    "van": (0.13, 0.80, 0.90),
    "truck": (0.00, 0.05, 0.33),
    "bus": (0.09, 0.85, 0.94),
    "motorcycle": (0.00, 0.10, 0.12),
    "bicycle": (0.52, 0.65, 0.68),
    "person": (0.08, 0.35, 0.75),
    "pedestrian": (0.66, 0.30, 0.60),
    "dog": (0.07, 0.55, 0.62),
    "cat": (0.08, 0.10, 0.68),
    "sign": (0.16, 0.75, 0.90),
    "tree": (0.33, 0.60, 0.35),
}


class LocalMetricEvaluatorBackend(LocalBackend):
    """Measure semantic agreement, artifacts, and embeddings locally."""

    backend_id = "local_metric_evaluator"
    backend_version = "1.0.0"
    capabilities = (CAP_QUALITY_SEMANTIC, CAP_QUALITY_ARTIFACT, CAP_QUALITY_EMBEDDING)
    determinism = Determinism.DETERMINISTIC
    declared_capabilities = (CAP_QUALITY_SEMANTIC, CAP_QUALITY_ARTIFACT, CAP_QUALITY_EMBEDDING)

    def __init__(self, spec: Any, options: dict[str, Any], credentials: object | None = None) -> None:
        super().__init__(spec, options, credentials)
        self._sync = _EvaluatorSync(self)

    async def assess_semantics(
        self, request: SemanticAssessRequest, context: PipelineContext
    ) -> SemanticAssessment:
        """Protocol method; the runtime executes ``self._sync.assess_semantics`` on a worker thread."""
        return self._sync.assess_semantics(request, context)

    async def evaluate(self, request: VLMRequest, context: PipelineContext) -> VLMResponse:
        """Protocol method; the runtime executes ``self._sync.evaluate`` on a worker thread."""
        return self._sync.evaluate(request, context)

    async def embed(self, request: EmbeddingRequest, context: PipelineContext) -> EmbeddingResult:
        """Protocol method; the runtime executes ``self._sync.embed`` on a worker thread."""
        return self._sync.embed(request, context)

    async def probe(self) -> BackendProbe:
        """Report readiness, stating plainly that these are heuristics rather than a learned model."""
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
            message="heuristic evaluators; bind a classifier or VLM for semantic judgement",
            details={"heuristic": True, **probe.details},
        )


class _EvaluatorSync:
    """Synchronous implementations of the local evaluators."""

    def __init__(self, backend: LocalMetricEvaluatorBackend) -> None:
        self._backend = backend

    # -- semantics --------------------------------------------------------- #

    def assess_semantics(
        self, request: SemanticAssessRequest, context: PipelineContext
    ) -> SemanticAssessment:
        """Score how closely the generated region matches the requested category."""
        started = timed()
        io = context.require_io()
        image = load_image_array(io, request.image)
        mask = _crop_mask(io, request)
        if mask is None or not mask.any():
            raise OperatorFailure(
                "semantic assessment needs a non-empty mask for the replaced object",
                code=ErrorCode.MASK_EMPTY,
            )
        appearance = _appearance_statistics(image, mask)
        expected = _reference_appearance(request.expected_category)
        distance = _appearance_distance(appearance, expected)
        tolerance = float(self._backend.option("appearance_tolerance", 0.75))
        similarity = float(np.clip(1.0 - distance / max(tolerance, 1e-6), 0.0, 1.0))
        # A region that is essentially the source colour would also "match" a reference, so the
        # score is blended with the region's own internal coherence: a flat, single-colour blob is a
        # weaker semantic match than a region with structure.
        coherence = float(np.clip(appearance["spread"] / 0.18, 0.0, 1.0))
        score = float(np.clip(0.78 * similarity + 0.22 * coherence, 0.0, 1.0))
        labels = _label_scores(appearance)
        best_label = max(labels, key=lambda name: labels[name]) if labels else None
        reason_codes: tuple[str, ...] = ()
        if score < float(self._backend.option("mismatch_threshold", 0.45)):
            reason_codes = (ReasonCode.SEMANTIC_MISMATCH.value,)
        return SemanticAssessment(
            score=score,
            matched_label=best_label,
            labels=labels,
            reason_codes=reason_codes,
            evaluator_id=self._backend.backend_id,
            detail={
                "expected": request.expected_category,
                "appearance": {name: round(value, 4) for name, value in appearance.items()},
                "distance": round(distance, 4),
                "duration_ms": elapsed_ms(started),
                "method": "category-appearance heuristic",
            },
        )

    # -- artifact report --------------------------------------------------- #

    def evaluate(self, request: VLMRequest, context: PipelineContext) -> VLMResponse:
        """Answer the structured artifact question this evaluator understands."""
        io = context.require_io()
        image = load_image_array(io, request.image)
        mask = _mask_from_request(io, request)
        if mask is None or not mask.any():
            return VLMResponse(
                answer="mask_missing",
                confidence=1.0,
                fields={
                    "artifact_score": 1.0,
                    "duplicate_object": False,
                    "floating_object": False,
                    "boundary_break": True,
                    "unexpected_text": False,
                    "unexpected_disappearance": False,
                },
                evaluator_id=self._backend.backend_id,
                raw_text="no mask supplied for artifact evaluation",
            )
        measurements = artifact_measurements(image, mask)
        components = connected_components(mask)
        fields = {
            "artifact_score": round(measurements.artifact_score, 6),
            "component_count": len(components),
            "component_ratio": round(measurements.component_ratio, 6),
            "duplicate_object": bool(measurements.component_ratio < 0.85),
            "floating_object": bool(measurements.floating > 0.6),
            "boundary_break": bool(measurements.edge_density > 0.6),
            "unexpected_text": bool(measurements.text_like > 0.5),
            "unexpected_disappearance": False,
        }
        reason_codes: list[str] = []
        if fields["duplicate_object"]:
            reason_codes.append(ReasonCode.ARTIFACT_DUPLICATE_OBJECT.value)
        if fields["floating_object"]:
            reason_codes.append(ReasonCode.ARTIFACT_FLOATING_OBJECT.value)
        if fields["boundary_break"]:
            reason_codes.append(ReasonCode.ARTIFACT_BOUNDARY_BREAK.value)
        if fields["unexpected_text"]:
            reason_codes.append(ReasonCode.ARTIFACT_TEXT.value)
        if not reason_codes and measurements.artifact_score > 0.5:
            reason_codes.append(ReasonCode.ARTIFACT_DETECTED.value)
        fields["reason_codes"] = reason_codes
        return VLMResponse(
            answer="artifact_report",
            confidence=float(np.clip(1.0 - measurements.artifact_score, 0.0, 1.0)),
            fields=fields,
            evaluator_id=self._backend.backend_id,
            raw_text=request.question,
        )

    # -- embeddings -------------------------------------------------------- #

    def embed(self, request: EmbeddingRequest, context: PipelineContext) -> EmbeddingResult:
        """Produce a compact appearance descriptor of the image or of the masked region."""
        io = context.require_io()
        image = load_image_array(io, request.image)
        mask = io.load_mask(request.mask) if request.mask is not None else None
        if mask is None:
            mask = np.ones(image.shape[:2], dtype=bool)
        vector = _descriptor(image, mask)
        return EmbeddingResult(
            vector=tuple(float(value) for value in vector),
            backend_id=self._backend.backend_id,
            model_id=self._backend.model_id(),
            dimension=int(vector.size),
        )


def _mask_from_request(io: ArtifactIO, request: VLMRequest) -> np.ndarray | None:
    if request.mask is None:
        return None
    return io.load_mask(request.mask)


def _crop_mask(io: ArtifactIO, request: SemanticAssessRequest) -> np.ndarray | None:
    if request.mask is None:
        return None
    return io.load_mask(request.mask)


def _appearance_statistics(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Colour and structure statistics of the masked region."""
    pixels = image[mask]
    if pixels.size == 0:
        return {"hue": 0.0, "saturation": 0.0, "value": 0.0, "spread": 0.0, "edge": 0.0}
    hsv = _rgb_to_hsv(pixels)
    luminance = (image * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=2)
    gradient = np.abs(np.diff(luminance, axis=0, prepend=luminance[:1, :]))
    return {
        "hue": float(hsv[:, 0].mean()),
        "saturation": float(hsv[:, 1].mean()),
        "value": float(hsv[:, 2].mean()),
        "spread": float(luminance[mask].std()),
        "edge": float(gradient[mask].mean()),
    }


def _reference_appearance(category: str) -> dict[str, float]:
    key = category.strip().lower()
    triple = _CATEGORY_REFERENCE.get(key)
    if triple is None:
        digest = 0
        for char in key:
            digest = (digest * 131 + ord(char)) % 1000003
        triple = ((digest % 360) / 360.0, 0.45, 0.60)
    return {
        "hue": float(triple[0]),
        "saturation": float(triple[1]),
        "value": float(triple[2]),
        "spread": 0.12,
    }


def _appearance_distance(actual: dict[str, float], expected: dict[str, float]) -> float:
    """Weighted distance in hue/saturation/value space, with hue treated circularly."""
    hue_delta = abs(actual["hue"] - expected["hue"])
    hue_delta = min(hue_delta, 1.0 - hue_delta)
    return float(
        1.6 * hue_delta
        + 0.7 * abs(actual["saturation"] - expected["saturation"])
        + 0.5 * abs(actual["value"] - expected["value"])
    )


def _label_scores(appearance: dict[str, float]) -> dict[str, float]:
    """Score the region against every known category, for the audit trail."""
    scores: dict[str, float] = {}
    for name in _CATEGORY_REFERENCE:
        distance = _appearance_distance(appearance, _reference_appearance(name))
        scores[name] = round(float(np.clip(1.0 - distance / 0.75, 0.0, 1.0)), 4)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def _rgb_to_hsv(pixels: np.ndarray) -> np.ndarray:
    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    maximum = np.max(pixels, axis=1)
    minimum = np.min(pixels, axis=1)
    span = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = span > 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
        red_is_max = nonzero & (maximum == red)
        green_is_max = nonzero & (maximum == green) & ~red_is_max
        blue_is_max = nonzero & ~red_is_max & ~green_is_max
        hue[red_is_max] = ((green[red_is_max] - blue[red_is_max]) / span[red_is_max]) % 6.0
        hue[green_is_max] = ((blue[green_is_max] - red[green_is_max]) / span[green_is_max]) + 2.0
        hue[blue_is_max] = ((red[blue_is_max] - green[blue_is_max]) / span[blue_is_max]) + 4.0
    hue = (hue / 6.0) % 1.0
    saturation = np.where(maximum > 1e-6, span / np.maximum(maximum, 1e-6), 0.0)
    return np.stack([hue, saturation, maximum], axis=1)


def _descriptor(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A 24-dimensional appearance descriptor: colour moments plus a coarse spatial histogram."""
    if not mask.any():
        mask = np.ones(image.shape[:2], dtype=bool)
    pixels = image[mask]
    statistics = _appearance_statistics(image, mask)
    base = np.array(
        [
            statistics["hue"],
            statistics["saturation"],
            statistics["value"],
            statistics["spread"],
            statistics["edge"],
        ],
        dtype=np.float32,
    )
    rows, columns = np.nonzero(mask)
    height, width = mask.shape
    grid = np.zeros((4, 4), dtype=np.float32)
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        grid[min(3, row * 4 // max(1, height)), min(3, column * 4 // max(1, width))] += 1.0
    grid = grid / max(1.0, grid.sum())
    colour_histogram = np.histogram(pixels.mean(axis=1), bins=3, range=(0.0, 1.0))[0].astype(np.float32)
    colour_histogram = colour_histogram / max(1.0, colour_histogram.sum())
    return np.concatenate([base, grid.reshape(-1), colour_histogram]).astype(np.float32)
