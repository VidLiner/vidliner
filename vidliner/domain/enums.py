"""Closed vocabularies used across the VidLiner domain.

Using ``StrEnum`` keeps every value JSON- and YAML-friendly while still giving the type checker a
closed set to work with.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "ArtifactKind",
    "AugmentationMode",
    "CandidateState",
    "Comparison",
    "DecisionState",
    "Determinism",
    "GateStatus",
    "JobState",
    "MediaKind",
    "MetricName",
    "NodeStatus",
    "PortType",
    "ReplacementStrategy",
    "SampleState",
    "SceneScaleClass",
    "Severity",
    "SplitName",
    "StageName",
]


class MediaKind(StrEnum):
    """Kind of source media VidLiner can ingest."""

    IMAGE = "image"
    VIDEO = "video"


class SplitName(StrEnum):
    """Dataset split a sample belongs to."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    UNASSIGNED = "unassigned"


class AugmentationMode(StrEnum):
    """How aggressively a replacement may deviate from the source object."""

    STRICT = "strict"
    """Preserve pose, scale, position, lighting, occlusion; used for training data."""

    CREATIVE = "creative"
    """Allow larger deviations; never enters the accepted dataset by default."""


class ReplacementStrategy(StrEnum):
    """How the recipe names the objects that should appear."""

    CATEGORY = "category"
    DESCRIPTION = "description"
    REFERENCE = "reference"


class SceneScaleClass(StrEnum):
    """Rough size of a target object inside the frame."""

    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class ArtifactKind(StrEnum):
    """What a stored artifact contains."""

    SOURCE_MEDIA = "source_media"
    IMAGE = "image"
    MASK = "mask"
    POLYGON = "polygon"
    DEPTH = "depth"
    TRACK = "track"
    CANDIDATE = "candidate"
    REFINED_IMAGE = "refined"
    DIFFERENCE = "difference"
    ANNOTATION = "annotation"
    METRICS = "metrics"
    REPORT = "report"
    DATASET_SLICE = "dataset_slice"
    MANIFEST = "manifest"


class Determinism(StrEnum):
    """How repeatable a backend or operator is for identical inputs and config."""

    DETERMINISTIC = "deterministic"
    SEEDED = "seeded"
    NONDETERMINISTIC = "nondeterministic"


class PortType(StrEnum):
    """The declared type of a graph port.

    Port types are intentionally coarse: they describe the *shape of the data* crossing an edge,
    which is what graph validation needs. Fine-grained validation happens inside each operator's
    Pydantic config and result models.
    """

    ANY = "any"
    FLAG = "flag"
    """A boolean outcome, such as "was the object found in the generated image?"."""
    MEDIA_ASSET = "media_asset"
    IMAGE_REF = "image_ref"
    INSTANCES = "instances"
    MASK_REF = "mask_ref"
    SCENE_CONTEXT = "scene_context"
    PLAN = "plan"
    CANDIDATE_REF = "candidate_ref"
    QUALITY_REPORT = "quality_report"
    ANNOTATION = "annotation"
    DATASET_SLICE = "dataset_slice"
    DECISION = "decision"
    ARTIFACT_LIST = "artifact_list"


class StageName(StrEnum):
    """Pipeline stage a node belongs to; used for reporting and ordering."""

    INGEST = "ingest"
    DETECT = "detect"
    SELECT = "select"
    SEGMENT = "segment"
    SCENE = "scene"
    PLAN = "plan"
    GENERATE = "generate"
    REFINE = "refine"
    VERIFY = "verify"
    EVALUATE = "evaluate"
    ANNOTATE = "annotate"
    GATE = "gate"
    EXPORT = "export"


class JobState(StrEnum):
    """Lifecycle of one job. Persisted, never inferred (requirement §15)."""

    CREATED = "created"
    PLANNED = "planned"
    RUNNING = "running"
    WAITING = "waiting"
    FINALIZING = "finalizing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(StrEnum):
    """Lifecycle of one graph node execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class CandidateState(StrEnum):
    """Lifecycle of one generated candidate."""

    GENERATED = "generated"
    EVALUATING = "evaluating"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVIEW = "review"


class SampleState(StrEnum):
    """State of a dataset sample row."""

    SOURCE = "source"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVIEW = "review"


class DecisionState(StrEnum):
    """Outcome of the acceptance gate."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class MetricName(StrEnum):
    """Quality metric vocabulary."""

    SEMANTIC_MATCH = "semantic_match"
    TARGET_PRESENCE = "target_presence"
    BACKGROUND_PRESERVATION = "background_preservation"
    MASK_BOUNDARY = "mask_boundary"
    GEOMETRY = "geometry"
    ARTIFACT_FREE = "artifact_free"
    ANNOTATION_CONSISTENCY = "annotation_consistency"
    TEMPORAL_CONSISTENCY = "temporal_consistency"


class Comparison(StrEnum):
    """Direction of a metric."""

    AT_LEAST = "at_least"
    AT_MOST = "at_most"


class Severity(StrEnum):
    """How a gate affects acceptance when it fails."""

    HARD = "hard"
    WARN = "warn"
    INFO = "info"


class GateStatus(StrEnum):
    """Result of evaluating one gate."""

    PASSED = "passed"
    FAILED = "failed"
    WARNED = "warned"
    SKIPPED = "skipped"
    ERROR = "error"
