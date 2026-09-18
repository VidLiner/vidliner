"""Backend contracts: the ports every capability adapter must satisfy.

These are protocols, not base classes with behaviour, so a backend can be a plain class, an adapter
around an HTTP client, a wrapper around a model runtime, or a test double. The only requirement is
that it exposes the members below and that :meth:`Backend.probe` tells the truth about readiness.

Two design points are load-bearing:

* **Requests carry artifacts, not pixels.** A backend receives digest-addressed
  :class:`~vidliner.core.results.ArtifactRef` values and reads the bytes through the artifact store
  it is given. That keeps the pipeline free of dtype/colour-space assumptions and makes every
  request cacheable.
* **`safe_to_retry` is a contract, not a hope.** A generative backend that is not idempotent says
  so, and the engine will not retry it without an explicit policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from vidliner.core.results import ArtifactRef
from vidliner.domain.enums import ArtifactKind, Determinism, MediaKind
from vidliner.domain.instances import ObjectInstance
from vidliner.domain.masks import MaskRef
from vidliner.domain.replacement import (
    GenerationOutcome,
    GenerationRequest,
    ReplacementIntent,
    ReplacementPlan,
)
from vidliner.domain.scene import SceneContext
from vidliner.domain.shapes import ImageShape
from vidliner.domain.tracks import ObjectTrack

__all__ = [
    "ArtifactIO",
    "Backend",
    "BackendHealth",
    "BackendProbe",
    "CompositeResult",
    "CompositingRequest",
    "DepthBackend",
    "DepthResult",
    "DetectionRequest",
    "DetectionResult",
    "EmbeddingBackend",
    "EmbeddingRequest",
    "EmbeddingResult",
    "HarmonizationRequest",
    "ImageCompositingBackend",
    "ImageEditBackend",
    "ImageHarmonizationBackend",
    "InstanceSegmentationBackend",
    "MaskRefinementBackend",
    "MaskRefinementRequest",
    "MaskRefinementResult",
    "ObjectDetectorBackend",
    "ObjectReplacementBackend",
    "PipelineContext",
    "PlanningRequest",
    "QualityEvaluatorBackend",
    "ReplacementPlannerBackend",
    "SceneAnalysisBackend",
    "SceneAnalysisRequest",
    "SegmentationRequest",
    "SegmentationResult",
    "SemanticAssessRequest",
    "SemanticAssessment",
    "TrackerBackend",
    "TrackingRequest",
    "TrackingResult",
    "VLMRequest",
    "VLMResponse",
    "VisionEvaluationBackend",
]


@runtime_checkable
class ArtifactIO(Protocol):
    """Artifact access a backend is allowed: read what it was given, write what it produces.

    A backend receives this and nothing else about storage. It cannot resolve workspace paths, open
    the state database, or reach the network on its own, which is what keeps a capability adapter
    replaceable and testable.
    """

    def load(self, artifact: ArtifactRef) -> bytes:
        """Read the bytes of an artifact, verifying its digest."""
        ...

    def load_json(self, artifact: ArtifactRef) -> object:
        """Read and parse a JSON artifact."""
        ...

    def image(self, artifact: ArtifactRef) -> Any:
        """Load an artifact as a Pillow image."""
        ...

    def load_mask(self, artifact: ArtifactRef) -> Any:
        """Load a mask artifact as a boolean numpy array."""
        ...

    def save_image(self, image: Any, *, kind: ArtifactKind = ...) -> ArtifactRef:
        """Store a Pillow image as an artifact."""
        ...

    def save_mask(self, mask: Any) -> ArtifactRef:
        """Store a boolean mask as an artifact."""
        ...

    def save_json(self, payload: object, *, kind: ArtifactKind = ...) -> ArtifactRef:
        """Store a JSON payload as an artifact."""
        ...

    @property
    def store(self) -> Any:
        """The underlying artifact store, for the few backends that need catalogue access."""
        ...


class BackendHealth(str):
    """Readiness reported by a probe."""

    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class BackendProbe:
    """The result of asking a backend whether it can work right now.

    A probe must be cheap and must never perform a paid or generative call: it checks credential
    presence, endpoint reachability, model files, and device availability only.
    """

    backend_id: str
    version: str
    health: str = BackendHealth.READY
    capabilities: tuple[str, ...] = ()
    device: str | None = None
    determinism: Determinism = Determinism.NONDETERMINISTIC
    safe_to_retry: bool = False
    external: bool = False
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """True when the backend can serve requests, possibly with reduced quality."""
        return self.health in {BackendHealth.READY, BackendHealth.DEGRADED}


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """Everything a backend is allowed to know about the run it is serving.

    Deliberately narrow: a backend gets an identity for logging, a seed, a device hint, and a place
    to read and write artifacts. It does not get the recipe, the job store, or the workspace root,
    so it cannot take a shortcut around the pipeline.
    """

    job_id: str
    node_id: str
    seed: int
    device: str = "cpu"
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    io: ArtifactIO | None = None
    """Artifact access for this invocation.

    A backend reads and writes artifacts through this object and never through the workspace, the
    state store, or the filesystem. ``None`` only in unit tests that exercise a backend with no
    storage."""
    model_id: str | None = None

    def option(self, name: str, default: Any = None) -> Any:
        """Read a backend option, preferring per-node config over profile options."""
        if name in self.config:
            return self.config[name]
        options = self.metadata.get("options")
        if isinstance(options, dict):
            return options.get(name, default)
        return default

    def require_io(self) -> ArtifactIO:
        """Artifact access, or an explicit error when the invocation has none.

        Raises:
            BackendFailure: when the context carries no artifact I/O, which means the node was
                invoked outside a runtime and is a wiring bug that must be loud.
        """
        if self.io is None:
            from vidliner.core.errors import BackendFailure, ErrorCode

            raise BackendFailure(
                f"backend invocation for node {self.node_id} has no artifact I/O",
                backend_id=str(self.metadata.get("backend_id", "unknown")),
                code=ErrorCode.BACKEND_UNAVAILABLE,
                detail={"node_id": self.node_id},
            )
        return self.io


@runtime_checkable
class Backend(Protocol):
    """The minimum every backend exposes."""

    @property
    def backend_id(self) -> str:
        """Stable identifier used in manifests and cache keys."""
        ...

    @property
    def backend_version(self) -> str:
        """Version of the adapter itself, not of the model it wraps."""
        ...

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Capability names this backend serves, from :mod:`vidliner.capabilities.names`."""
        ...

    async def probe(self) -> BackendProbe:
        """Report readiness without performing real work."""
        ...


# --------------------------------------------------------------------------- #
# Perception
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DetectionRequest:
    """Ask for objects in an image.

    ``known_classes`` is the vocabulary the *source dataset* actually labels. A detector with a fixed
    label set cannot honestly report a class the dataset never contained, so the harness passes the
    vocabulary and the detector refuses to invent a label outside it. That is what makes a class
    filter above this stage meaningful instead of decorative.
    """

    image: ArtifactRef
    classes: tuple[str, ...] = ()
    min_score: float = 0.0
    max_objects: int = 100
    media_kind: MediaKind = MediaKind.IMAGE
    frame_index: int = 0
    known_classes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Detected objects plus the identity of what produced them."""

    instances: tuple[ObjectInstance, ...]
    backend_id: str
    model_id: str | None = None
    duration_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ObjectDetectorBackend(Protocol):
    """Locate objects and report class, confidence, and box."""

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    async def probe(self) -> BackendProbe: ...

    async def detect(self, request: DetectionRequest, context: PipelineContext) -> DetectionResult:
        """Return every object the backend finds for ``request``."""
        ...


@dataclass(frozen=True, slots=True)
class SegmentationRequest:
    """Ask for a pixel-level mask for one object.

    Exactly one prompt kind is set. ``text`` prompting is optional and a backend that cannot honour
    it must raise :class:`~vidliner.core.errors.BackendFailure` rather than fall back silently.
    """

    image: ArtifactRef
    image_shape: ImageShape
    bbox: tuple[float, float, float, float] | None = None
    point: tuple[float, float] | None = None
    text: str | None = None
    object_id: str = ""
    guidance: str = "auto"


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """A mask plus its vector outline when the backend can produce one."""

    mask: ArtifactRef
    shape: ImageShape
    area_px: int
    polygon: Any | None = None
    score: float = 1.0
    backend_id: str = ""
    duration_ms: int = 0

    @property
    def coverage(self) -> float:
        """Fraction of the image covered by the mask."""
        return self.area_px / self.shape.pixel_count if self.shape.pixel_count else 0.0

    @property
    def mask_ref(self) -> MaskRef:
        """Wrap the produced artifact as a domain mask reference."""
        return MaskRef(
            artifact=self.mask.with_kind(ArtifactKind.MASK),
            shape=self.shape,
            area_px=self.area_px,
        )


@runtime_checkable
class InstanceSegmentationBackend(Protocol):
    """Produce a pixel-level mask for a prompted object."""

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    async def probe(self) -> BackendProbe: ...

    async def segment(self, request: SegmentationRequest, context: PipelineContext) -> SegmentationResult:
        """Return a mask for the prompted object."""
        ...


@dataclass(frozen=True, slots=True)
class TrackingRequest:
    """Ask for tracks across the frames of one media asset."""

    asset_digest: str
    frames: tuple[int, ...] = ()
    classes: tuple[str, ...] = ()
    initial_instances: tuple[ObjectInstance, ...] = ()


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """Tracks plus the identity of what produced them."""

    tracks: tuple[ObjectTrack, ...]
    backend_id: str = ""
    duration_ms: int = 0


@runtime_checkable
class TrackerBackend(Protocol):
    """Associate one object across frames. Phase 2 capability; the contract exists now."""

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    async def probe(self) -> BackendProbe: ...

    async def track(self, request: TrackingRequest, context: PipelineContext) -> TrackingResult:
        """Return tracks for the requested frames."""
        ...


@dataclass(frozen=True, slots=True)
class DepthResult:
    """A depth estimate plus the identity of what produced it."""

    depth: ArtifactRef
    shape: ImageShape
    min_depth: float
    max_depth: float
    backend_id: str = ""
    duration_ms: int = 0


@runtime_checkable
class DepthBackend(Protocol):
    """Estimate per-pixel relative depth. Phase 2 capability; the contract exists now."""

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    async def probe(self) -> BackendProbe: ...

    async def estimate_depth(self, image: ArtifactRef, context: PipelineContext) -> DepthResult:
        """Return a relative depth map for ``image``."""
        ...


@dataclass(frozen=True, slots=True)
class SceneAnalysisRequest:
    """Ask for measured facts about one target object and its surroundings.

    Pixels are attached as runtime-only fields (``exclude=True`` semantics) rather than loaded by the
    backend: the operator that builds the request already had to decode the frame and the mask, so
    handing them over avoids a second decode and keeps the backend free of storage concerns.
    """

    image: ArtifactRef
    image_shape: ImageShape
    target: ObjectInstance
    neighbours: tuple[ObjectInstance, ...] = ()
    mask_array: Any = None
    """Decoded boolean mask of the target object, in image coordinates."""
    luminance: Any = None
    """Decoded luminance array in ``[0, 1]`` for the same frame."""


@runtime_checkable
class SceneAnalysisBackend(Protocol):
    """Measure lighting, orientation, scale, occlusion, and ground contact."""

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    async def probe(self) -> BackendProbe: ...

    async def analyse(self, request: SceneAnalysisRequest, context: PipelineContext) -> SceneContext:
        """Return the measured scene context for the request's target object."""
        ...


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PlanningRequest:
    """Ask for a replacement plan for one target object.

    ``candidate_values`` is the recipe's list of replacements; a planner assigns them to candidates in
    order, which is what makes "sedan, suv, pickup" produce one of each rather than three of the
    first. An empty list means the intent's own category is the only value.
    """

    sample_id: str
    intent: ReplacementIntent
    scene: SceneContext
    candidates_per_object: int
    prompt_template: str
    negative_constraints: tuple[str, ...] = ()
    mask_expansion_px: int = 0
    candidate_values: tuple[str, ...] = ()


@runtime_checkable
class ReplacementPlannerBackend(Protocol):
    """Turn a recipe and a scene context into an executable plan. Never calls a generative model."""

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    async def probe(self) -> BackendProbe: ...

    async def plan(self, request: PlanningRequest, context: PipelineContext) -> ReplacementPlan:
        """Return the plan for the request's target object."""
        ...


# --------------------------------------------------------------------------- #
# Generation and refinement
# --------------------------------------------------------------------------- #


@runtime_checkable
class ObjectReplacementBackend(Protocol):
    """Generate a replacement object inside a masked region. The headline generative port."""

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    async def probe(self) -> BackendProbe: ...

    async def replace(self, request: GenerationRequest, context: PipelineContext) -> GenerationOutcome:
        """Return one generated candidate image."""
        ...


@dataclass(frozen=True, slots=True)
class CompositingRequest:
    """Blend a generated region into its source image."""

    source_image: ArtifactRef
    candidate_image: ArtifactRef
    mask: ArtifactRef
    shape: ImageShape
    feather_px: int = 0
    preserve_outside_mask: bool = True


@dataclass(frozen=True, slots=True)
class CompositeResult:
    """The composited image plus the evidence of how much changed outside the mask."""

    image: ArtifactRef
    changed_ratio_outside_mask: float = 0.0
    backend_id: str = ""
    duration_ms: int = 0


@runtime_checkable
class ImageCompositingBackend(Protocol):
    """Blend a generated region into an image without changing the rest."""

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    async def probe(self) -> BackendProbe: ...

    async def composite(self, request: CompositingRequest, context: PipelineContext) -> CompositeResult:
        """Return the composited image."""
        ...


@dataclass(frozen=True, slots=True)
class HarmonizationRequest:
    """Match colour and illumination between a region and its surroundings."""

    image: ArtifactRef
    mask: ArtifactRef
    shape: ImageShape
    strength: float = 0.45


@runtime_checkable
class ImageHarmonizationBackend(Protocol):
    """Match colour and illumination between a generated region and its surroundings."""

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    async def probe(self) -> BackendProbe: ...

    async def harmonize(self, request: HarmonizationRequest, context: PipelineContext) -> ArtifactRef:
        """Return the harmonised image."""
        ...


@dataclass(frozen=True, slots=True)
class MaskRefinementRequest:
    """Clean a mask edge before compositing."""

    mask: ArtifactRef
    shape: ImageShape
    feather_px: int = 3
    close_px: int = 5
    dilate_px: int = 0


@dataclass(frozen=True, slots=True)
class MaskRefinementResult:
    """The refined mask and how its area changed."""

    mask: ArtifactRef
    area_px: int
    area_delta_px: int = 0
    backend_id: str = ""
    duration_ms: int = 0


@runtime_checkable
class MaskRefinementBackend(Protocol):
    """Clean and feather a mask edge."""

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    async def probe(self) -> BackendProbe: ...

    async def refine_mask(
        self, request: MaskRefinementRequest, context: PipelineContext
    ) -> MaskRefinementResult:
        """Return the refined mask."""
        ...


@runtime_checkable
class ImageEditBackend(Protocol):
    """A general instruction-driven image edit, subsuming replacement and harmonisation.

    Declared so a hosted image-editing service can serve several capabilities through one adapter
    without the core knowing which wire format it speaks.
    """

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    async def probe(self) -> BackendProbe: ...

    async def edit(self, request: GenerationRequest, context: PipelineContext) -> GenerationOutcome:
        """Return one edited image."""
        ...


# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SemanticAssessRequest:
    """Ask whether a generated object matches the requested category."""

    image: ArtifactRef
    mask: ArtifactRef
    bbox: tuple[float, float, float, float]
    expected_category: str
    description: str = ""
    negative_constraints: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticAssessment:
    """The measured semantic agreement, with per-label evidence."""

    score: float
    matched_label: str | None = None
    labels: dict[str, float] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    evaluator_id: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class QualityEvaluatorBackend(Protocol):
    """Measure one or more named quality metrics for a candidate."""

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    async def probe(self) -> BackendProbe: ...

    async def assess_semantics(
        self, request: SemanticAssessRequest, context: PipelineContext
    ) -> SemanticAssessment:
        """Measure semantic agreement between the generated object and the request."""
        ...


@dataclass(frozen=True, slots=True)
class VLMRequest:
    """A structured visual question for a vision-language evaluator."""

    image: ArtifactRef
    mask: ArtifactRef | None = None
    question: str = ""
    choices: tuple[str, ...] = ()
    schema_name: str = "artifact_report"


@dataclass(frozen=True, slots=True)
class VLMResponse:
    """The evaluator's structured answer."""

    answer: str
    confidence: float = 0.0
    fields: dict[str, Any] = field(default_factory=dict)
    evaluator_id: str = ""
    raw_text: str = ""


@runtime_checkable
class VisionEvaluationBackend(Protocol):
    """Answer a structured visual question, used by the artifact and semantic gates."""

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    async def probe(self) -> BackendProbe: ...

    async def evaluate(self, request: VLMRequest, context: PipelineContext) -> VLMResponse:
        """Return a structured answer to ``request.question``."""
        ...


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """Ask for an embedding of an image or of a region of it."""

    image: ArtifactRef
    mask: ArtifactRef | None = None
    crop_bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """An embedding vector plus the identity of the model that produced it."""

    vector: tuple[float, ...]
    backend_id: str = ""
    model_id: str | None = None
    dimension: int = 0

    def cosine(self, other: EmbeddingResult) -> float:
        """Cosine similarity with another embedding of the same dimension."""
        if self.dimension != other.dimension:
            raise ValueError("cannot compare embeddings of different dimensions")
        dot = sum(left * right for left, right in zip(self.vector, other.vector, strict=True))
        left_norm = sum(value * value for value in self.vector) ** 0.5
        right_norm = sum(value * value for value in other.vector) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Produce an embedding for similarity comparison (duplicate control, semantic checks)."""

    @property
    def backend_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    async def probe(self) -> BackendProbe: ...

    async def embed(self, request: EmbeddingRequest, context: PipelineContext) -> EmbeddingResult:
        """Return an embedding for ``request``."""
        ...


class AbstractBackend(ABC):
    """Convenience base class for backends that prefer inheritance over duck typing.

    It exists so a backend author can get a clear error for a missing member instead of a
    protocol-mismatch at call time. It carries no behaviour that the protocols do not describe.
    """

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Stable identifier used in manifests and cache keys."""

    @property
    @abstractmethod
    def backend_version(self) -> str:
        """Version of the adapter itself."""

    @property
    @abstractmethod
    def capabilities(self) -> tuple[str, ...]:
        """Capability names this backend serves."""

    @abstractmethod
    async def probe(self) -> BackendProbe:
        """Report readiness without performing real work."""
