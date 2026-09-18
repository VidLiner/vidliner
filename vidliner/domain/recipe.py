"""The declarative recipe schema.

A recipe says *what* the user wants. It never names a model, a service, or a file path that the
runtime should have chosen. Every field is validated with Pydantic v2, and ``Recipe.model_json_schema()``
is the published contract (`vidliner recipe schema`).
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AcceptanceSpec",
    "DatasetSpec",
    "DuplicatesSpec",
    "EstimatesSpec",
    "ExportSpec",
    "LimitsSpec",
    "PreserveSpec",
    "QualitySpec",
    "Recipe",
    "RefineSpec",
    "RefinementStepSpec",
    "ReplacementSpec",
    "SplitSpec",
    "TargetSpec",
    "TemporalGatesSpec",
]


class SplitSpec(BaseModel):
    """How split membership is discovered for a dataset directory."""

    model_config = ConfigDict(extra="forbid")

    mode: str = "none"
    """``none`` (everything is train), ``directory``, ``filename``, or ``manifest``."""
    names: tuple[str, ...] = ("train", "val", "test")
    """Directory names (or filename prefixes) that identify each split, in priority order."""
    manifest: str | None = None
    """Path to a JSON mapping ``relative_path -> split`` when ``mode`` is ``manifest``."""

    @model_validator(mode="after")
    def _check_mode(self) -> Self:
        if self.mode not in {"none", "directory", "filename", "manifest"}:
            raise ValueError("split mode must be none, directory, filename, or manifest")
        if self.mode == "manifest" and not self.manifest:
            raise ValueError("split mode 'manifest' requires a manifest path")
        return self


class DatasetSpec(BaseModel):
    """Where the source data is and how it is annotated."""

    model_config = ConfigDict(extra="forbid")

    input: str
    format: str | None = None
    """Annotation format of the source set. ``None`` means "images only, no annotations"."""
    splits: SplitSpec = Field(default_factory=SplitSpec)
    augment_splits: tuple[str, ...] = ("train",)
    allow_test_augmentation: bool = False
    recursive: bool = True
    max_samples: int = Field(default=0, ge=0)
    """Cap on discovered source samples; ``0`` means no cap."""


class TargetSpec(BaseModel):
    """Which objects in each sample should be replaced."""

    model_config = ConfigDict(extra="forbid")

    classes: tuple[str, ...] = Field(min_length=1)
    min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    min_area_px: int = Field(default=1024, ge=0)
    max_per_sample: int = Field(default=2, ge=1, le=64)
    top_k_by: str = "score"
    strategy: str = "all"
    object_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_strategy(self) -> Self:
        if self.top_k_by not in {"score", "area"}:
            raise ValueError("top_k_by must be 'score' or 'area'")
        if self.strategy not in {"all", "largest", "first", "explicit"}:
            raise ValueError("target strategy must be all, largest, first, or explicit")
        if self.strategy == "explicit" and not self.object_ids:
            raise ValueError("target strategy 'explicit' requires object_ids")
        if not self.classes:
            raise ValueError("target classes must not be empty")
        cleaned = tuple(sorted({name.strip() for name in self.classes if name.strip()}))
        if not cleaned:
            raise ValueError("target classes must contain at least one non-empty name")
        object.__setattr__(self, "classes", cleaned)
        return self


class ReplacementSpec(BaseModel):
    """What should replace the target object, and how many variations to try."""

    model_config = ConfigDict(extra="forbid")

    mode: str = "strict"
    strategy: str = "category"
    values: tuple[str, ...] = ()
    description: str = ""
    references: tuple[str, ...] = ()
    candidates_per_object: int = Field(default=3, ge=1, le=16)
    seed: int | None = None
    mask_expansion_px: int = Field(default=0, ge=0, le=256)
    context_padding_px: int = Field(default=32, ge=0, le=2048)
    prompt_template: str = (
        "Replace the {category} in the masked region with {description}. "
        "Keep the background, camera viewpoint, lighting direction, shadows, and ground contact unchanged."
    )
    negative_constraints: tuple[str, ...] = (
        "no duplicated object",
        "no floating object",
        "no text or watermark",
        "no change to the background outside the mask",
    )

    @model_validator(mode="after")
    def _check_values(self) -> Self:
        if self.mode not in {"strict", "creative"}:
            raise ValueError("replacement mode must be strict or creative")
        if self.strategy not in {"category", "description", "reference"}:
            raise ValueError("replacement strategy must be category, description, or reference")
        if self.strategy == "category":
            cleaned = tuple(dict.fromkeys(value.strip() for value in self.values if value.strip()))
            if not cleaned:
                raise ValueError("replacement strategy 'category' requires a non-empty values list")
            object.__setattr__(self, "values", cleaned)
        if self.strategy == "description" and not self.description.strip():
            raise ValueError("replacement strategy 'description' requires a description")
        if self.strategy == "reference" and not self.references:
            raise ValueError("replacement strategy 'reference' requires at least one reference path")
        if "{category}" not in self.prompt_template and "{description}" not in self.prompt_template:
            raise ValueError("prompt_template must reference at least one of {category} or {description}")
        return self


class PreserveSpec(BaseModel):
    """What must not change."""

    model_config = ConfigDict(extra="forbid")

    background: str = "strict"
    geometry: bool = True
    lighting: bool = True
    pose: bool = True
    scale: bool = True
    position: bool = True
    occlusion: bool = True
    geometry_tolerance_centroid: float = Field(default=0.08, gt=0.0, le=1.0)
    geometry_tolerance_area_min: float = Field(default=0.70, gt=0.0)
    geometry_tolerance_area_max: float = Field(default=1.45, gt=0.0)
    geometry_tolerance_aspect: float = Field(default=0.30, ge=0.0, le=1.0)
    geometry_tolerance_ground_px: float = Field(default=24.0, ge=0.0)

    @model_validator(mode="after")
    def _check_background(self) -> Self:
        if self.background not in {"strict", "balanced", "loose"}:
            raise ValueError("background preservation must be strict, balanced, or loose")
        if self.geometry_tolerance_area_max < self.geometry_tolerance_area_min:
            raise ValueError("area tolerance maximum must not be below the minimum")
        return self


class RefinementStepSpec(BaseModel):
    """One post-processing step referenced by registered operator name."""

    model_config = ConfigDict(extra="forbid")

    operator: str
    config: dict[str, object] = Field(default_factory=dict)
    enabled: bool = True


class RefineSpec(BaseModel):
    """The ordered refinement pipeline applied to every candidate."""

    model_config = ConfigDict(extra="forbid")

    steps: tuple[RefinementStepSpec, ...] = (
        RefinementStepSpec(operator="refine.mask_edges", config={"feather_px": 3, "close_px": 5}),
        RefinementStepSpec(operator="refine.alpha_blend", config={}),
    )


class TemporalGatesSpec(BaseModel):
    """Video-only gates. Reserved: recorded, validated, and ignored for image inputs."""

    model_config = ConfigDict(extra="forbid")

    minimum_identity_stability: float = Field(default=0.90, ge=0.0, le=1.0)
    minimum_mask_stability: float = Field(default=0.90, ge=0.0, le=1.0)
    maximum_flicker: float = Field(default=0.10, ge=0.0, le=1.0)
    maximum_position_jump_px: float = Field(default=48.0, ge=0.0)


class QualitySpec(BaseModel):
    """The acceptance policy in recipe form."""

    model_config = ConfigDict(extra="forbid")

    minimum_overall: float = Field(default=0.82, ge=0.0, le=1.0)
    hard_gates: dict[str, float] = Field(
        default_factory=lambda: {
            "semantic_match": 0.90,
            "background_preservation": 0.93,
            "annotation_consistency": 0.95,
        }
    )
    warn_gates: dict[str, float] = Field(default_factory=dict)
    maximum_artifact_score: float = Field(default=0.15, ge=0.0, le=1.0)
    minimum_target_presence: float = Field(default=0.60, ge=0.0, le=1.0)
    review_band: float = Field(default=0.03, ge=0.0, le=0.5)
    background_change_ceiling: float = Field(default=0.06, ge=0.0, le=1.0)
    weights: dict[str, float] = Field(default_factory=dict)
    evaluators: tuple[str, ...] = ()
    allow_missing_metrics: bool = True
    temporal: TemporalGatesSpec = Field(default_factory=TemporalGatesSpec)

    @model_validator(mode="after")
    def _check_gate_names(self) -> Self:
        from vidliner.domain.enums import MetricName

        known = {metric.value for metric in MetricName}
        for name in (*self.hard_gates, *self.warn_gates, *self.weights):
            if name not in known:
                raise ValueError(f"unknown quality metric {name!r}; known metrics are {sorted(known)}")
        overlap = set(self.hard_gates) & set(self.warn_gates)
        if overlap:
            raise ValueError(f"metric(s) declared as both hard and warn gates: {sorted(overlap)}")
        for name, value in {**self.hard_gates, **self.warn_gates}.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"gate threshold for {name} must be between 0 and 1")
        return self


class AcceptanceSpec(BaseModel):
    """What to do with candidates that are not clearly accepted."""

    model_config = ConfigDict(extra="forbid")

    on_quality_reject: str = "keep_diagnostics"
    export_review: bool = False
    allow_creative_in_dataset: bool = False

    @model_validator(mode="after")
    def _check_on_reject(self) -> Self:
        if self.on_quality_reject not in {"keep_diagnostics", "discard"}:
            raise ValueError("on_quality_reject must be keep_diagnostics or discard")
        return self


class DuplicatesSpec(BaseModel):
    """Near-duplicate control."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    perceptual_hash: str = "phash"
    hamming_threshold: int = Field(default=6, ge=0, le=64)
    compare_against_source: bool = True
    embedding_backend: str | None = None
    embedding_threshold: float = Field(default=0.98, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_hash(self) -> Self:
        if self.perceptual_hash not in {"phash", "ahash", "dhash", "none"}:
            raise ValueError("perceptual_hash must be phash, ahash, dhash, or none")
        return self


class ExportSpec(BaseModel):
    """How accepted samples are written out."""

    model_config = ConfigDict(extra="forbid")

    format: str = "coco-instance"
    path: str = "dataset"
    splits: dict[str, str] = Field(default_factory=dict)
    copy_images: bool = True
    image_format: str = "same"
    include_rejected: bool = False
    include_provenance: bool = True
    include_review_report: bool = True
    min_annotation_area_px: int = Field(default=4, ge=0)

    @model_validator(mode="after")
    def _check_format(self) -> Self:
        from vidliner.domain.annotations import ANNOTATION_FORMATS

        if self.format not in ANNOTATION_FORMATS:
            raise ValueError(f"export format must be one of {ANNOTATION_FORMATS}")
        if self.image_format not in {"same", "png", "jpg"}:
            raise ValueError("image_format must be same, png, or jpg")
        return self


class LimitsSpec(BaseModel):
    """Execution limits."""

    model_config = ConfigDict(extra="forbid")

    max_samples: int = Field(default=0, ge=0)
    max_candidates_total: int = Field(default=0, ge=0)
    per_node_timeout_s: float = Field(default=600.0, gt=0.0)
    job_timeout_s: float = Field(default=0.0, ge=0.0)
    fail_fast: bool = False
    resume: bool = True
    cache: bool = True
    workers: int = Field(default=4, ge=1, le=256)


class EstimatesSpec(BaseModel):
    """User-supplied cost/time overrides for planning. Excluded from the recipe hash."""

    model_config = ConfigDict(extra="forbid")

    currency: str = "USD"
    generation_unit_cost: float = Field(default=0.0, ge=0.0)
    generation_unit_seconds: float = Field(default=0.0, ge=0.0)


class Recipe(BaseModel):
    """A complete augmentation recipe."""

    model_config = ConfigDict(extra="forbid")

    recipe: str
    description: str = ""
    dataset: DatasetSpec
    target: TargetSpec
    replacement: ReplacementSpec
    preserve: PreserveSpec = Field(default_factory=PreserveSpec)
    refine: RefineSpec = Field(default_factory=RefineSpec)
    quality: QualitySpec = Field(default_factory=QualitySpec)
    acceptance: AcceptanceSpec = Field(default_factory=AcceptanceSpec)
    duplicates: DuplicatesSpec = Field(default_factory=DuplicatesSpec)
    export: ExportSpec = Field(default_factory=ExportSpec)
    limits: LimitsSpec = Field(default_factory=LimitsSpec)
    estimates: EstimatesSpec = Field(default_factory=EstimatesSpec)

    @model_validator(mode="after")
    def _check_identity(self) -> Self:
        if not self.recipe.strip():
            raise ValueError("recipe name must not be empty")
        if any(char.isspace() for char in self.recipe):
            raise ValueError("recipe name must not contain whitespace")
        return self

    @model_validator(mode="after")
    def _check_splits(self) -> Self:
        blocked = {name for name in self.dataset.augment_splits if name in {"test", "validation", "val"}}
        if blocked and not self.dataset.allow_test_augmentation:
            raise ValueError(
                f"augmenting split(s) {sorted(blocked)} is refused by default; "
                "set dataset.allow_test_augmentation: true to override deliberately"
            )
        if blocked and self.replacement.mode == "strict":
            raise ValueError(
                "strict replacement mode does not permit test/validation augmentation; "
                "leakage protection is not opt-out inside strict mode"
            )
        unknown = set(self.dataset.augment_splits) - set(self.dataset.splits.names) - {"train", "unassigned"}
        if self.dataset.splits.mode != "none" and unknown:
            raise ValueError(
                f"augment_splits names {sorted(unknown)} are not among dataset.splits.names "
                f"{list(self.dataset.splits.names)}"
            )
        return self

    @model_validator(mode="after")
    def _check_creative(self) -> Self:
        permitting = self.replacement.mode == "creative" and self.acceptance.allow_creative_in_dataset
        unguarded = not self.acceptance.export_review and self.quality.minimum_overall < 0.5
        if permitting and unguarded:
            raise ValueError(
                "creative replacements with a very low minimum_overall and no review export "
                "would silently degrade the dataset; raise minimum_overall or export review"
            )
        return self

    @property
    def name(self) -> str:
        """The recipe name (alias of ``recipe``)."""
        return self.recipe

    @property
    def is_strict(self) -> bool:
        """True when the recipe produces dataset-grade strict replacements."""
        return self.replacement.mode == "strict"

    def snapshot(self) -> dict[str, object]:
        """Full canonical-ready snapshot, including estimates, for the job manifest."""
        return self.model_dump(mode="json")

    def hashed_snapshot(self) -> dict[str, object]:
        """Snapshot used for the recipe hash: estimates are excluded as they are labelling aids."""
        return self.model_dump(mode="json", exclude={"estimates"})

    def recipe_hash(self) -> str:
        """Stable digest identifying this recipe exactly."""
        from vidliner.core.canonical import digest_json

        return digest_json(self.hashed_snapshot())

    def required_capabilities(self) -> tuple[str, ...]:
        """Capabilities this recipe needs regardless of dataset contents.

        Deriving the list from the *recipe* (rather than from the compiled graph) is what lets
        ``plan`` explain a missing binding before a graph exists, and lets ``backend check`` report
        which configured backend would serve each requirement.
        """
        from vidliner.capabilities.names import capability_for_refinement

        capabilities = [
            "vision.object_detection.v1",
            "vision.instance_segmentation.v1",
            "vision.scene_analysis.v1",
            "planning.replacement.v1",
            "generation.object_replacement.v1",
            "quality.semantic_match.v1",
            "quality.background_preservation.v1",
            "quality.artifact_detection.v1",
        ]
        for step in self.refine.steps:
            if step.enabled:
                capabilities.append(capability_for_refinement(step.operator))
        return tuple(dict.fromkeys(capabilities))
