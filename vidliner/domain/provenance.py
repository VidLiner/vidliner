"""Provenance: the link from an exported sample back to how it was produced.

Requirement §11 lists exactly what must be traceable, and every field below exists because of it.
The model is deliberately flat and JSON-only: provenance ends up inside a dataset that is shared
with other people, so it must not carry object references, file handles, or credential material.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidliner.core.results import utc_now
from vidliner.domain.enums import AugmentationMode, DecisionState

__all__ = ["ProvenanceRecord", "SeedRecord", "assert_no_secrets"]

_SECRET_KEY_MARKERS = ("key", "token", "secret", "password", "credential", "authorization")
_ALLOWED_KEY_EXCEPTIONS = {
    "keyframe",
    "keys",
    "candidate_key",
    "object_key",
    "sort_key",
    "primary_key",
    "api_key_ref",
}


class SeedRecord(BaseModel):
    """The seed tree for one accepted sample, so the sample can be reproduced in isolation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_seed: int
    sample_seed: int
    target_seed: int | None = None
    candidate_seed: int | None = None
    node_seeds: dict[str, int] = Field(default_factory=dict)

    def as_flat(self) -> dict[str, int]:
        """Flatten into the ``path -> seed`` mapping stored in a manifest."""
        flat = {"job": self.job_seed, "sample": self.sample_seed}
        if self.target_seed is not None:
            flat["target"] = self.target_seed
        if self.candidate_seed is not None:
            flat["candidate"] = self.candidate_seed
        flat.update({f"node.{name}": seed for name, seed in sorted(self.node_seeds.items())})
        return flat


class ProvenanceRecord(BaseModel):
    """Everything needed to explain and reproduce one accepted output sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Where it came from
    source_sample_id: str
    source_digest: str = Field(min_length=16, max_length=64)
    source_path: str = ""
    split: str = "train"
    lineage_root: str = ""
    ancestors: tuple[str, ...] = ()
    augmentation_depth: int = Field(default=1, ge=1)

    # What produced it
    recipe_name: str
    recipe_hash: str
    job_id: str
    job_seed: int
    target_object_id: str
    target_class: str
    replacement_category: str
    replacement_description: str = ""
    replacement_mode: AugmentationMode = AugmentationMode.STRICT
    intent: dict[str, Any] = Field(default_factory=dict)
    seeds: SeedRecord

    # Which code produced it
    operator_versions: dict[str, str] = Field(default_factory=dict)
    backend_ids: dict[str, str] = Field(default_factory=dict)
    model_ids: dict[str, str] = Field(default_factory=dict)
    generation_parameters: dict[str, Any] = Field(default_factory=dict)

    # What came out
    output_digest: str = Field(min_length=16, max_length=64)
    output_path: str = ""
    output_shape: tuple[int, int] | None = None
    annotation_path: str = ""
    annotation_format: str = ""
    annotation_object_count: int = Field(default=0, ge=0)

    # How good it is and what was decided
    quality_scores: dict[str, float] = Field(default_factory=dict)
    overall_score: float | None = Field(default=None, ge=0.0, le=1.0)
    decision: DecisionState = DecisionState.ACCEPTED
    reason_codes: tuple[str, ...] = ()
    policy_hash: str = ""
    duplicate_of: str | None = None
    perceptual_hash: str | None = None

    # When
    created_at: datetime = Field(default_factory=utc_now)
    vidliner_version: str = "0.1.0"

    @model_validator(mode="after")
    def _reject_secrets(self) -> Self:
        offending = find_secret_keys(self.model_dump(mode="json"))
        if offending:
            raise ValueError(
                f"provenance record contains credential-like fields: {', '.join(sorted(offending))}"
            )
        return self


def find_secret_keys(payload: Any, prefix: str = "") -> set[str]:
    """Return the dotted paths of keys that look like credentials.

    The check is intentionally name-based and slightly over-eager: a false positive costs one
    rename, a false negative costs a leaked API key in a distributed dataset.
    """
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            path = f"{prefix}.{key}" if prefix else str(key)
            if lowered not in _ALLOWED_KEY_EXCEPTIONS and any(
                marker in lowered for marker in _SECRET_KEY_MARKERS
            ):
                if isinstance(value, str) and value.startswith(("env:", "file:", "<redacted>")):
                    continue
                if value is None:
                    continue
                if lowered.endswith("_ref"):
                    continue
                found.add(path)
            found |= find_secret_keys(value, path)
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            found |= find_secret_keys(item, f"{prefix}[{index}]")
    return found


def assert_no_secrets(payload: Any, *, context: str) -> None:
    """Raise :class:`ValueError` when ``payload`` contains a credential-like field."""
    offending = find_secret_keys(payload)
    if offending:
        raise ValueError(f"{context} contains credential-like fields: {', '.join(sorted(offending))}")
