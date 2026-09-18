"""Operator results, artifact references, evidence records, and job summaries.

These objects cross the operator boundary, so they are plain Pydantic models: they serialize,
they validate, and they can be cached. They never hold live file handles or image buffers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vidliner.domain.enums import ArtifactKind, Determinism, JobState, NodeStatus

__all__ = [
    "ArtifactRef",
    "JobCounters",
    "JobSummary",
    "NodeResult",
    "NodeTelemetry",
    "OperationEvidence",
    "PortValue",
    "Timestamped",
    "utc_now",
]


def utc_now() -> datetime:
    """Current time in UTC. One helper so every timestamp has the same source."""
    from datetime import UTC
    from datetime import datetime as _datetime

    return _datetime.now(UTC)


class Timestamped(BaseModel):
    """Base model with a creation timestamp."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    created_at: datetime = Field(default_factory=utc_now)


class ArtifactRef(BaseModel):
    """A digest-addressed reference to a stored artifact.

    The digest is the SHA-256 of the artifact bytes, lower-case hex. Two artifacts with the same
    digest are the same bytes, which is what makes the cache and provenance links meaningful.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ArtifactKind
    digest: str = Field(min_length=16, max_length=64)
    media_type: str
    size_bytes: int = Field(ge=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    suffix: str = ""
    """File extension used in the store, without the dot (``png``, ``npy``, ``json``)."""

    @property
    def short_digest(self) -> str:
        """First 12 characters of the digest, for logs and console output."""
        return self.digest[:12]

    def with_kind(self, kind: ArtifactKind) -> ArtifactRef:
        """Return a copy with a different artifact kind (used when re-labelling a refinement)."""
        return self.model_copy(update={"kind": kind})


class PortValue(BaseModel):
    """Anything that travels along a graph edge.

    A port value is always one of: an artifact reference, a plain JSON value produced by a
    deterministic operator (a plan, a decision, a metric list), or nothing at all. Keeping this
    union explicit is what lets the engine serialize node outputs into the cache without knowing
    what any individual operator does.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact: ArtifactRef | None = None
    payload: dict[str, Any] | None = None
    label: str = ""

    @classmethod
    def of_artifact(cls, artifact: ArtifactRef, label: str = "") -> PortValue:
        """Wrap a single artifact reference."""
        return cls(artifact=artifact, label=label)

    @classmethod
    def of_payload(cls, payload: dict[str, Any], label: str = "") -> PortValue:
        """Wrap a JSON-serializable payload."""
        return cls(payload=payload, label=label)

    @classmethod
    def of_artifacts(cls, artifacts: list[ArtifactRef], label: str = "") -> PortValue:
        """Wrap a homogeneous list of artifacts as a payload."""
        return cls(payload={"artifacts": [item.model_dump(mode="json") for item in artifacts]}, label=label)

    def require_artifact(self) -> ArtifactRef:
        """Return the wrapped artifact, raising if this port carries something else."""
        if self.artifact is None:
            raise ValueError(f"port {self.label or '<unnamed>'} does not carry an artifact")
        return self.artifact


class NodeTelemetry(BaseModel):
    """Cost and resource telemetry reported by one node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cost: float | None = None
    currency: str | None = None
    external_calls: int = 0
    device: str | None = None


class NodeResult(BaseModel):
    """The outcome of executing one node.

    ``payload`` is stored in the node cache; ``artifacts`` is stored in the artifact catalogue.
    A failed node still produces a result object so the failure is recorded uniformly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    operator: str
    operator_version: str
    status: NodeStatus
    outputs: dict[str, PortValue] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    finished_at: datetime
    duration_ms: int = Field(ge=0)
    backend_id: str | None = None
    attempt: int = Field(default=0, ge=0)
    cache_hit: bool = False
    resumed: bool = False
    determinism: Determinism = Determinism.DETERMINISTIC
    telemetry: NodeTelemetry = Field(default_factory=NodeTelemetry)
    error_class: str | None = None
    error_code: str | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        """True when the node completed its work."""
        return self.status is NodeStatus.SUCCEEDED


class OperationEvidence(BaseModel):
    """The auditable record of one node execution, written to ``state.db`` and the run log."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    node_id: str
    operator: str
    operator_version: str
    stage: str
    lineage: str
    status: NodeStatus
    attempt: int = Field(default=0, ge=0)
    cache_hit: bool = False
    resumed: bool = False
    backend_id: str | None = None
    config_hash: str
    seed: int
    input_digests: list[str] = Field(default_factory=list)
    output_digests: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
    error_class: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class JobCounters(BaseModel):
    """Separate counting of system failures and sample rejections (requirement §16)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: int = 0
    targets: int = 0
    generated: int = 0
    accepted: int = 0
    rejected: int = 0
    review: int = 0
    duplicates: int = 0
    failed: int = 0
    nodes_succeeded: int = 0
    nodes_failed: int = 0
    nodes_cached: int = 0
    nodes_resumed: int = 0

    @property
    def evaluated(self) -> int:
        """Candidates that reached a decision."""
        return self.accepted + self.rejected + self.review

    @property
    def acceptance_rate(self) -> float:
        """Accepted candidates divided by decided candidates, or ``0.0`` when nothing was decided."""
        decided = self.evaluated
        return self.accepted / decided if decided else 0.0


class JobSummary(BaseModel):
    """Human- and machine-readable roll-up of one job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    state: JobState
    created_at: datetime
    finished_at: datetime | None = None
    duration_s: float | None = None
    counters: JobCounters = Field(default_factory=JobCounters)
    average_quality: float | None = None
    estimated_cost: float | None = None
    actual_cost: float | None = None
    currency: str | None = None
    backends: list[str] = Field(default_factory=list)
    reason_code_histogram: dict[str, int] = Field(default_factory=dict)
    failure_class: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
