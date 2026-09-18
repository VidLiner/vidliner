"""Runtime profile: which backend serves which capability on this machine.

The profile is the only place a vendor is named. It is validated strictly, it never contains a
resolved credential value, and it can be serialized into a job manifest after redaction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vidliner.capabilities.names import KNOWN_CAPABILITIES
from vidliner.core.errors import ErrorCode, ValidationFailure

__all__ = [
    "BackendSpec",
    "CapacitySpec",
    "ConcurrencySpec",
    "CredentialRef",
    "EstimateSpec",
    "RuntimeProfile",
    "StorageSpec",
    "default_profile",
    "load_profile",
    "profile_from_mapping",
    "redact_profile",
]


class CredentialRef(BaseModel):
    """A *reference* to a credential. Never the credential itself inside a repository.

    ``source`` is one of:

    * ``env`` — read the environment variable named by ``name``;
    * ``file`` — read a file whose path is ``name`` (used for mounted secrets);
    * ``value`` — an inline literal. This is rejected for profiles stored inside the workspace and
      flagged by ``vidliner backend check``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    name: str = ""
    required: bool = True

    @model_validator(mode="after")
    def _check_source(self) -> Self:
        if self.source not in {"env", "file", "value"}:
            raise ValueError("credential source must be env, file, or value")
        if self.source in {"env", "file"} and not self.name:
            raise ValueError(f"credential source {self.source!r} requires a name")
        return self

    def is_inline(self) -> bool:
        """True when the reference carries the secret value directly."""
        return self.source == "value"

    def redacted(self) -> dict[str, str]:
        """Serializable form with no secret material."""
        return {"source": self.source, "name": self.name if self.source != "value" else "<redacted>"}


class CapacitySpec(BaseModel):
    """How much concurrent and how much rate-limited traffic a backend accepts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    limit: int = Field(default=2, ge=1, le=512)
    period_s: float | None = Field(default=None, gt=0.0)
    weight: int = Field(default=1, ge=1)


class EstimateSpec(BaseModel):
    """Cost and latency estimates for planning. Never used during execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    unit_cost: float = Field(default=0.0, ge=0.0)
    unit_seconds: float = Field(default=0.0, ge=0.0)
    currency: str = "USD"
    external: bool = False
    accelerator: bool = False


class BackendSpec(BaseModel):
    """One configured backend instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    use: str
    options: dict[str, Any] = Field(default_factory=dict)
    credentials: dict[str, CredentialRef] = Field(default_factory=dict)
    capacity: CapacitySpec = Field(default_factory=CapacitySpec)
    estimates: EstimateSpec = Field(default_factory=EstimateSpec)
    enabled: bool = True
    description: str = ""

    @model_validator(mode="after")
    def _check_use(self) -> Self:
        if ":" not in self.use and "." not in self.use:
            raise ValueError(
                f"backend 'use' must be an import path like 'package.module:ClassName', got {self.use!r}"
            )
        return self


class ConcurrencySpec(BaseModel):
    """Bounds on parallelism."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workers: int = Field(default=4, ge=1, le=256)
    per_backend: int = Field(default=2, ge=1, le=256)
    external_requests: int = Field(default=3, ge=1, le=256)


class StorageSpec(BaseModel):
    """Where the workspace keeps its data and how large artifacts may be."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: str = "."
    keep_rejected: bool = True
    max_artifact_mb: int = Field(default=256, ge=1)
    state_filename: str = "state.db"


class RuntimeProfile(BaseModel):
    """The complete runtime configuration for one machine (or one execution environment)."""

    model_config = ConfigDict(extra="forbid")

    profile: str = "default"
    device: str = "cpu"
    storage: StorageSpec = Field(default_factory=StorageSpec)
    concurrency: ConcurrencySpec = Field(default_factory=ConcurrencySpec)
    backends: dict[str, BackendSpec] = Field(default_factory=dict)
    bindings: dict[str, str] = Field(default_factory=dict)
    source_path: Path | None = Field(default=None, exclude=True)

    @field_validator("device")
    @classmethod
    def _check_device(cls, value: str) -> str:
        if value not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("device must be cpu, cuda, mps, or auto")
        return value

    @model_validator(mode="after")
    def _check_bindings(self) -> Self:
        for capability, backend in self.bindings.items():
            if capability not in KNOWN_CAPABILITIES:
                raise ValueError(
                    f"binding for unknown capability {capability!r}; known: {', '.join(KNOWN_CAPABILITIES)}"
                )
            if backend not in self.backends:
                raise ValueError(f"binding for {capability} names unconfigured backend {backend!r}")
        for name, spec in self.backends.items():
            if not name:
                raise ValueError("backend names must not be empty")
            inline = any(credential.is_inline() for credential in spec.credentials.values())
            repository_local = self.source_path is not None and _looks_like_repository(self.source_path)
            if spec.enabled and inline and repository_local:
                raise ValueError(
                    f"backend {name!r} stores an inline credential in a repository-local profile; "
                    "use an env: or file: reference instead"
                )
        return self

    def enabled_backends(self) -> dict[str, BackendSpec]:
        """Configured backends that are switched on."""
        return {name: spec for name, spec in self.backends.items() if spec.enabled}

    def backend_names(self) -> tuple[str, ...]:
        """Sorted names of enabled backends."""
        return tuple(sorted(self.enabled_backends()))

    def binding_for(self, capability: str) -> str | None:
        """Explicit binding for a capability, if one is declared."""
        return self.bindings.get(capability)

    def device_for(self, backend: str) -> str:
        """Device hint for a backend: its option overrides the profile default."""
        spec = self.backends.get(backend)
        if spec is None:
            return self.device
        option = spec.options.get("device")
        if isinstance(option, str) and option:
            return option
        if self.device == "auto":
            return "cpu"
        return self.device

    def snapshot(self) -> dict[str, Any]:
        """Redacted, JSON-ready snapshot for the job manifest."""
        return redact_profile(self)


def _looks_like_repository(path: Path) -> bool:
    """Heuristic: a profile inside a directory that also holds a VCS marker is repository-local."""
    for candidate in (path.parent, *path.parents):
        if (candidate / ".git").exists() or (candidate / ".hg").exists():
            return True
    return False


def redact_profile(profile: RuntimeProfile) -> dict[str, Any]:
    """Serialize a profile with every credential replaced by its reference, never its value."""
    payload = profile.model_dump(mode="json", exclude={"source_path"})
    for spec in payload.get("backends", {}).values():
        credentials = spec.get("credentials", {})
        spec["credentials"] = {
            name: {
                "source": ref.get("source"),
                "name": "<redacted>" if ref.get("source") == "value" else ref.get("name"),
            }
            for name, ref in credentials.items()
        }
    return payload


def profile_from_mapping(payload: dict[str, Any], *, source_path: Path | None = None) -> RuntimeProfile:
    """Build and validate a profile from a plain mapping."""
    try:
        return RuntimeProfile.model_validate({**payload, "source_path": source_path})
    except Exception as exc:  # pydantic ValidationError
        raise ValidationFailure(
            f"runtime profile is invalid: {exc}",
            code=ErrorCode.RUNTIME_INVALID,
            detail={"source": str(source_path) if source_path else None},
        ) from exc


def load_profile(path: Path) -> RuntimeProfile:
    """Load a runtime profile from a YAML or JSON document.

    Raises:
        ValidationFailure: when the file is missing, unparseable, or fails validation.
    """
    import json

    import yaml

    if not path.is_file():
        raise ValidationFailure(f"runtime profile not found: {path}", code=ErrorCode.RUNTIME_INVALID)
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except Exception as exc:
        raise ValidationFailure(
            f"runtime profile {path} could not be parsed: {exc}",
            code=ErrorCode.RUNTIME_INVALID,
        ) from exc
    if not isinstance(payload, dict):
        raise ValidationFailure(
            f"runtime profile {path} must contain a mapping at the top level",
            code=ErrorCode.RUNTIME_INVALID,
        )
    return profile_from_mapping(payload, source_path=path)


def default_profile(*, device: str = "cpu") -> RuntimeProfile:
    """A working, dependency-free profile: built-in local backends and no external service.

    This is what ``vidliner init`` writes and what makes the whole pipeline runnable — and testable —
    without a paid API or a downloaded model.
    """
    return RuntimeProfile(
        profile="local",
        device=device,
        backends={
            "builtin_detector": BackendSpec(
                use="vidliner.backends.heuristic.detector:HeuristicDetectorBackend",
                options={"score_floor": 0.30},
                estimates=EstimateSpec(unit_seconds=0.02),
            ),
            "builtin_segmenter": BackendSpec(
                use="vidliner.backends.heuristic.segmentation:HeuristicSegmentationBackend",
                options={"dilate_px": 2},
                estimates=EstimateSpec(unit_seconds=0.03),
            ),
            "builtin_scene": BackendSpec(
                use="vidliner.backends.heuristic.scene:HeuristicSceneBackend",
                estimates=EstimateSpec(unit_seconds=0.02),
            ),
            "builtin_planner": BackendSpec(
                use="vidliner.backends.heuristic.planner:RuleBasedPlannerBackend",
                estimates=EstimateSpec(unit_seconds=0.01),
            ),
            "builtin_replacement": BackendSpec(
                use="vidliner.backends.fake.replacement:FakeReplacementBackend",
                options={"variation": "tint"},
                estimates=EstimateSpec(unit_cost=0.0, unit_seconds=0.05),
            ),
            "builtin_metrics": BackendSpec(
                use="vidliner.backends.local.evaluator:LocalMetricEvaluatorBackend",
                estimates=EstimateSpec(unit_seconds=0.05),
            ),
            "builtin_refiner": BackendSpec(
                use="vidliner.backends.local.refiner:LocalRefinerBackend",
                estimates=EstimateSpec(unit_seconds=0.02),
            ),
            "builtin_phash": BackendSpec(
                use="vidliner.backends.local.fingerprint:PerceptualHashBackend",
                estimates=EstimateSpec(unit_seconds=0.01),
            ),
        },
        bindings={
            "vision.object_detection.v1": "builtin_detector",
            "vision.instance_segmentation.v1": "builtin_segmenter",
            "vision.scene_analysis.v1": "builtin_scene",
            "planning.replacement.v1": "builtin_planner",
            "generation.object_replacement.v1": "builtin_replacement",
            "quality.semantic_match.v1": "builtin_metrics",
            "quality.artifact_detection.v1": "builtin_metrics",
            "quality.embedding.v1": "builtin_phash",
            "generation.mask_refinement.v1": "builtin_refiner",
            "generation.image_compositing.v1": "builtin_refiner",
            "generation.image_harmonization.v1": "builtin_refiner",
        },
    )
