"""Job manifests, operator/backend version records, and the compiled plan.

A job manifest is the artifact an auditor reads. It must answer, without any other file: *what
recipe ran, with which seeds, against which runtime, using which operator and backend versions, and
what came out*. It must never contain a credential value; the profile snapshot it stores is already
redacted, and a test asserts that no secret can reach it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidliner.core.results import JobCounters, JobSummary, utc_now
from vidliner.domain.enums import Determinism, JobState, StageName

__all__ = [
    "BackendVersion",
    "JobManifest",
    "JobPlan",
    "OperatorVersion",
    "PlanEstimate",
    "StageStat",
]


class OperatorVersion(BaseModel):
    """Identity and version of one operator implementation used by a job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operator: str
    version: str
    name: str = ""
    capabilities: tuple[str, ...] = ()


class BackendVersion(BaseModel):
    """Identity and declared properties of one backend bound during a job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend_id: str
    version: str
    kind: str = ""
    capabilities: tuple[str, ...] = ()
    determinism: Determinism = Determinism.NONDETERMINISTIC
    device: str | None = None
    external: bool = False

    @property
    def label(self) -> str:
        """``id@version`` for display."""
        return f"{self.backend_id}@{self.version}"


class StageStat(BaseModel):
    """Node count and estimated resource use for one pipeline stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: StageName
    nodes: int = Field(ge=0)
    external_nodes: int = Field(default=0, ge=0)
    accelerator_nodes: int = Field(default=0, ge=0)
    estimated_seconds: float = Field(default=0.0, ge=0.0)
    estimated_cost: float = Field(default=0.0, ge=0.0)


class PlanEstimate(BaseModel):
    """Resource and cost estimate produced by planning, never by execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: int = Field(default=0, ge=0)
    targets: int = Field(default=0, ge=0)
    candidates: int = Field(default=0, ge=0)
    nodes: int = Field(default=0, ge=0)
    external_calls: int = Field(default=0, ge=0)
    accelerator_operations: int = Field(default=0, ge=0)
    estimated_seconds: float = Field(default=0.0, ge=0.0)
    estimated_cost: float = Field(default=0.0, ge=0.0)
    currency: str | None = None
    stages: tuple[StageStat, ...] = ()
    per_backend: dict[str, int] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def has_cost(self) -> bool:
        """True when at least one backend declared a monetary estimate."""
        return self.estimated_cost > 0.0


class JobPlan(BaseModel):
    """The compiled graph plus its estimate and the capability bindings it resolved."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    recipe_hash: str
    graph_hash: str
    node_count: int = Field(ge=0)
    capability_bindings: dict[str, str] = Field(default_factory=dict)
    unmet_capabilities: tuple[str, ...] = ()
    estimate: PlanEstimate = Field(default_factory=PlanEstimate)
    created_at: datetime = Field(default_factory=utc_now)
    dry_run: bool = False

    @property
    def is_runnable(self) -> bool:
        """True when every required capability resolved to a backend."""
        return not self.unmet_capabilities


class JobManifest(BaseModel):
    """Immutable record of one job attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    state: JobState = JobState.CREATED
    recipe_name: str
    recipe_hash: str
    recipe_snapshot: dict[str, Any]
    runtime_snapshot: dict[str, Any]
    runtime_profile_name: str = ""
    seed: int
    workspace_root: str
    run_dir: str
    dataset_input: str = ""
    output_path: str = ""
    operator_versions: tuple[OperatorVersion, ...] = ()
    backend_versions: tuple[BackendVersion, ...] = ()
    capability_bindings: dict[str, str] = Field(default_factory=dict)
    seed_tree: dict[str, int] = Field(default_factory=dict)
    """Sample-level seeds, recorded so a single sample can be reproduced in isolation."""
    counters: JobCounters = Field(default_factory=JobCounters)
    node_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    summary: JobSummary | None = None
    failure_class: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    demo_backends: tuple[str, ...] = ()
    """Production-critical capabilities served by demonstration backends during this job.

    Recorded even when the recipe allowed the export: a manifest that does not say a dataset was
    produced with stand-ins is a manifest that cannot be audited.
    """
    demo_backends_allowed: bool = False
    vidliner_version: str = "0.1.0"

    @model_validator(mode="after")
    def _check_finished(self) -> Self:
        if self.finished_at is not None and self.finished_at < self.created_at:
            raise ValueError("job finished_at precedes created_at")
        return self

    @property
    def is_terminal(self) -> bool:
        """True when the job has reached a final state."""
        return self.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}

    @property
    def duration_s(self) -> float | None:
        """Wall-clock duration in seconds, once the job has finished."""
        if self.finished_at is None:
            return None
        return (self.finished_at - self.created_at).total_seconds()
