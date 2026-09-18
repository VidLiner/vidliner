"""Replacement intent, plan, candidate specification, and generation contracts.

Three separations matter here and they are the reason these are four models rather than one:

* **Intent** is what the *user* asked for. It knows nothing about models or masks.
* **Plan** is what the *planner* decided for one target object given the scene. It knows nothing
  about which backend will execute it.
* **GenerationRequest** is the *uniform* request every replacement backend must accept, so that
  swapping a hosted service for a local model requires no recipe change and no core change.
* **ReplacementCandidate** is the thing that lives, gets evaluated, gets accepted or rejected, and
  ends up (or does not end up) in the dataset.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidliner.core.results import ArtifactRef, PortValue, utc_now
from vidliner.domain.enums import (
    AugmentationMode,
    CandidateState,
    DecisionState,
    ReplacementStrategy,
)
from vidliner.domain.quality import QualityReport
from vidliner.domain.shapes import BoundingBox, ImageShape

__all__ = [
    "AugmentedSample",
    "CandidateSpec",
    "GenerationOutcome",
    "GenerationRequest",
    "GeometryBudget",
    "PreservationRules",
    "RefinementRecipe",
    "RefinementStep",
    "ReplacementCandidate",
    "ReplacementIntent",
    "ReplacementPlan",
]


class PreservationRules(BaseModel):
    """Which properties of the source object must survive the replacement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    background: str = "strict"
    """``strict`` (a large background change rejects), ``balanced``, or ``loose``."""
    geometry: bool = True
    lighting: bool = True
    pose: bool = True
    scale: bool = True
    position: bool = True
    occlusion: bool = True

    @model_validator(mode="after")
    def _check_background(self) -> Self:
        if self.background not in {"strict", "balanced", "loose"}:
            raise ValueError("background preservation must be strict, balanced, or loose")
        return self


class ReplacementIntent(BaseModel):
    """What the user wants to happen to one target object.

    Attributes:
        target_object: identity of the instance being replaced.
        replacement_category: the class the new object should belong to.
        replacement_description: free-form description used for prompt-driven strategies.
        strategy: how the desired object was named.
        mode: ``strict`` for dataset production, ``creative`` for exploration.
        allowed_geometry_change: when true the geometry gate tolerance is widened.
        seed: explicit seed for this intent, overriding the derived one.
        reference: a user-supplied reference image, when the strategy is ``reference``.
        negatives: user-supplied constraints that must not appear.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_object: str
    replacement_category: str
    replacement_description: str = ""
    strategy: ReplacementStrategy = ReplacementStrategy.CATEGORY
    mode: AugmentationMode = AugmentationMode.STRICT
    preserve_pose: bool = True
    preserve_scale: bool = True
    preserve_position: bool = True
    preserve_lighting: bool = True
    preserve_occlusion: bool = True
    allowed_geometry_change: bool = False
    seed: int | None = None
    reference: ArtifactRef | None = None
    negatives: tuple[str, ...] = ()

    @property
    def is_creative(self) -> bool:
        """True when the intent allows creative deviation."""
        return self.mode is AugmentationMode.CREATIVE

    def with_seed(self, seed: int) -> ReplacementIntent:
        """Return a copy with an explicit seed."""
        return self.model_copy(update={"seed": seed})


class GeometryBudget(BaseModel):
    """How far the replacement may deviate geometrically before the geometry gate objects.

    Values are expressed relative to the source object so that one recipe works across image sizes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    centroid_shift_max: float = Field(default=0.08, gt=0.0, le=1.0)
    """Maximum centroid movement as a fraction of the image diagonal."""
    area_ratio_min: float = Field(default=0.70, gt=0.0)
    area_ratio_max: float = Field(default=1.45, gt=0.0)
    aspect_ratio_delta_max: float = Field(default=0.30, ge=0.0, le=1.0)
    ground_contact_max_px: float = Field(default=24.0, ge=0.0)

    @model_validator(mode="after")
    def _check_ratio(self) -> Self:
        if self.area_ratio_max < self.area_ratio_min:
            raise ValueError("area ratio maximum must not be below the minimum")
        return self


class CandidateSpec(BaseModel):
    """One variation the planner wants generated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_key: str
    """Stable key, e.g. ``sedan-1``. Part of the candidate identity and of the seed path."""
    category: str
    description: str
    seed: int
    ordinal: int = Field(default=0, ge=0)
    reference: ArtifactRef | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_key(self) -> Self:
        if not self.candidate_key or "/" in self.candidate_key or ":" in self.candidate_key:
            raise ValueError(
                f"candidate key {self.candidate_key!r} must be non-empty and free of ':' and '/'"
            )
        return self


class ReplacementPlan(BaseModel):
    """The planner's decision for one target object.

    The plan states *what changes* (the positive prompt and expected category), *what must remain*
    (preservation rules and geometry budget), and *how many variations* to generate. It is produced
    without calling any model, which is what makes ``plan`` and ``--dry-run`` free.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str
    sample_id: str
    intent: ReplacementIntent
    expected_category: str
    positive_prompt: str
    negative_constraints: tuple[str, ...] = ()
    mask_expansion_px: int = Field(default=0, ge=0)
    depth_expansion_px: int = Field(default=0, ge=0)
    preservation: PreservationRules = Field(default_factory=PreservationRules)
    geometry_budget: GeometryBudget = Field(default_factory=GeometryBudget)
    candidate_specs: tuple[CandidateSpec, ...] = ()
    plan_seed: int
    planner_id: str
    notes: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_candidates(self) -> Self:
        keys = [spec.candidate_key for spec in self.candidate_specs]
        if len(set(keys)) != len(keys):
            raise ValueError("replacement plan contains duplicate candidate keys")
        return self

    @property
    def candidate_count(self) -> int:
        """Number of candidates the plan asks for."""
        return len(self.candidate_specs)

    @property
    def target_object(self) -> str:
        """Identity of the object this plan applies to."""
        return self.intent.target_object


class GenerationRequest(BaseModel):
    """The uniform request every replacement backend must accept.

    A backend may ignore fields it does not support, but it must never reinterpret them: the fields
    it cannot honour are reported through ``GenerationOutcome.raw_metadata`` so the limitation is
    visible in the audit trail instead of hidden behind a plausible-looking image.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str
    target_object: str
    candidate_key: str
    source_image: ArtifactRef
    target_mask: ArtifactRef
    mask_shape: ImageShape
    target_bbox: BoundingBox
    context_crop: ArtifactRef | None = None
    positive_prompt: str
    negative_constraints: tuple[str, ...] = ()
    expected_category: str
    replacement_description: str = ""
    seed: int
    intensity: float = Field(default=1.0, ge=0.0, le=1.0)
    """How strongly to apply the edit; strict mode keeps this at or below 1.0."""
    reference_image: ArtifactRef | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    attempt: int = Field(default=0, ge=0)

    def cache_signature(self) -> dict[str, Any]:
        """The subset of the request that must be recorded for provenance, credentials excluded."""
        return {
            "sample": self.sample_id,
            "object": self.target_object,
            "candidate": self.candidate_key,
            "source": self.source_image.digest,
            "mask": self.target_mask.digest,
            "prompt": self.positive_prompt,
            "negatives": list(self.negative_constraints),
            "expected_category": self.expected_category,
            "seed": self.seed,
            "intensity": self.intensity,
            "reference": self.reference_image.digest if self.reference_image else None,
        }


class GenerationOutcome(BaseModel):
    """What a replacement backend returns.

    ``raw_metadata`` is recorded for audit and is never an input to a quality gate — a generator's
    own opinion of its output is not evidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: ArtifactRef
    backend_id: str
    model_id: str | None = None
    seed: int
    duration_ms: int = Field(default=0, ge=0)
    cost_estimate: float | None = None
    currency: str | None = None
    external: bool = False
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    applied_parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_image(self) -> Self:
        if self.image.width is None or self.image.height is None:
            raise ValueError("a generated image must report its dimensions")
        return self


class RefinementStep(BaseModel):
    """One post-processing step, resolved at run time through the operator registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operator: str
    config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class RefinementRecipe(BaseModel):
    """The ordered refinement pipeline applied to every generated candidate.

    Refinement is deliberately *not* folded into the generation backend. Keeping it as separate
    operators means a better harmoniser can be dropped in without regenerating anything, and that
    the effect of each step is separately measurable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: tuple[RefinementStep, ...] = ()

    @property
    def is_empty(self) -> bool:
        """True when no refinement is configured."""
        return not any(step.enabled for step in self.steps)

    def enabled_steps(self) -> tuple[RefinementStep, ...]:
        """The steps that will actually run, in order."""
        return tuple(step for step in self.steps if step.enabled)


class ReplacementCandidate(BaseModel):
    """One generated candidate and its complete lifecycle.

    This object is mutable-by-copy: each stage returns a new instance with one more field filled in.
    The final instance is what gets persisted alongside its quality report and its annotation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    job_id: str
    sample_id: str
    target_object: str
    spec: CandidateSpec
    state: CandidateState = CandidateState.GENERATED
    plan_id: str = ""
    generated: GenerationOutcome | None = None
    refined: ArtifactRef | None = None
    mask_used: ArtifactRef | None = None
    quality: QualityReport | None = None
    annotation: PortValue | None = None
    decision: DecisionState | None = None
    reason_codes: tuple[str, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    seed: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def final_image(self) -> ArtifactRef | None:
        """The image that should be exported: the refined one when refinement ran."""
        return self.refined or (self.generated.image if self.generated else None)

    @property
    def overall_score(self) -> float | None:
        """Overall quality score, when the candidate has been evaluated."""
        return self.quality.overall_score if self.quality else None

    @property
    def is_accepted(self) -> bool:
        """True when this candidate may enter the accepted dataset."""
        return self.state is CandidateState.ACCEPTED

    def evolve(self, **changes: Any) -> ReplacementCandidate:
        """Return a copy with ``changes`` applied and ``updated_at`` refreshed."""
        return self.model_copy(update={**changes, "updated_at": utc_now()})


class AugmentedSample(BaseModel):
    """A produced sample ready for export (or for review, when the decision says so)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str
    source_sample_id: str
    candidate_id: str
    image: ArtifactRef
    source_image: ArtifactRef
    source_mask: ArtifactRef
    annotation: PortValue
    decision: DecisionState
    quality: QualityReport | None = None
    split: str = "train"
    lineage_root: str
    augmentation_depth: int = Field(default=1, ge=1)
    artifacts: tuple[ArtifactRef, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def is_accepted(self) -> bool:
        """True when the decision permits export."""
        return self.decision is DecisionState.ACCEPTED
