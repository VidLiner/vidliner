"""Duplicate control.

An augmentation pipeline that floods a dataset with near-identical variants makes the dataset worse,
not better: the model memorises the duplicate instead of learning the concept. This module provides
two complementary checks:

* **perceptual hashing** — always available, no model required, catches recompression, scaling, and
  mild colour shifts through Hamming distance;
* **embedding similarity** — optional, used when an ``quality.embedding.v1`` backend is bound, for
  the semantic duplicates a perceptual hash cannot see.

Both are advisory to the *exporter*: the decision to drop a candidate is recorded with its distance
and its reference, so the dataset report can explain exactly what was removed and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vidliner.quality.fingerprint import Fingerprint, hamming_distance, perceptual_hash

__all__ = ["DuplicateDecision", "DuplicateIndex", "DuplicateKind"]


class DuplicateKind:
    """How a duplicate was detected."""

    EXACT = "exact"
    NEAR = "near"


@dataclass(frozen=True, slots=True)
class DuplicateDecision:
    """The verdict for one candidate."""

    is_duplicate: bool
    kind: str
    reference: str
    distance: float

    @classmethod
    def unique(cls) -> DuplicateDecision:
        """The verdict for a candidate with no match."""
        return cls(is_duplicate=False, kind="", reference="", distance=0.0)


@dataclass
class DuplicateIndex:
    """Tracks perceptual fingerprints of sources and accepted outputs."""

    enabled: bool = True
    algorithm: str = "phash"
    hamming_threshold: int = 6
    compare_against_source: bool = True
    embedding_threshold: float = 0.98
    existing: dict[str, str] = field(default_factory=dict)
    _fingerprints: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _accepted: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _embeddings: dict[str, tuple[float, ...]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for sample, hex_digest in self.existing.items():
            try:
                self._fingerprints[sample] = int(hex_digest, 16)
            except (TypeError, ValueError):
                continue

    # -- registration ------------------------------------------------------ #

    def fingerprint_of(self, image: np.ndarray) -> Fingerprint:
        """Compute a fingerprint with the configured algorithm."""
        return perceptual_hash(image, algorithm=self.algorithm)

    def register_source(self, sample_id: str, image: np.ndarray) -> Fingerprint:
        """Register a source sample's fingerprint, computed from its pixels."""
        return self.register_source_value(sample_id, self.fingerprint_of(image).value)

    def register_source_value(self, sample_id: str, value: int) -> Fingerprint:
        """Register an already-computed fingerprint value."""
        self._fingerprints[sample_id] = value & ((1 << 64) - 1)
        return Fingerprint(self._fingerprints[sample_id], self.algorithm)

    def register_accepted(self, candidate_id: str, image: np.ndarray) -> Fingerprint:
        """Register an accepted output's fingerprint, computed from its pixels."""
        return self.register_accepted_value(candidate_id, self.fingerprint_of(image).value)

    def register_accepted_value(self, candidate_id: str, value: int) -> Fingerprint:
        """Register an already-computed fingerprint value."""
        self._accepted[candidate_id] = value & ((1 << 64) - 1)
        return Fingerprint(self._accepted[candidate_id], self.algorithm)

    def check_value(self, value: int, *, candidate_id: str = "") -> DuplicateDecision:
        """Decide whether an already-computed fingerprint duplicates a known sample."""
        if not self.enabled or self.algorithm == "none":
            return DuplicateDecision.unique()
        best_reference = ""
        best_distance = self.hamming_threshold + 1
        pools: list[dict[str, int]] = []
        if self.compare_against_source:
            pools.append(self._fingerprints)
        pools.append(self._accepted)
        for pool in pools:
            for reference, other in pool.items():
                if reference == candidate_id:
                    continue
                distance = hamming_distance(value & ((1 << 64) - 1), other)
                if distance < best_distance:
                    best_distance = distance
                    best_reference = reference
        if not best_reference or best_distance > self.hamming_threshold:
            return DuplicateDecision.unique()
        kind = DuplicateKind.EXACT if best_distance == 0 else DuplicateKind.NEAR
        return DuplicateDecision(
            is_duplicate=True,
            kind=kind,
            reference=best_reference,
            distance=float(best_distance),
        )

    def register_embedding(self, candidate_id: str, vector: tuple[float, ...]) -> None:
        """Register an accepted output's embedding, when an embedding backend is bound."""
        self._embeddings[candidate_id] = vector

    # -- queries ----------------------------------------------------------- #

    def check(self, image: np.ndarray, *, candidate_id: str = "") -> DuplicateDecision:
        """Decide whether an image duplicates a known source or an already accepted output."""
        if not self.enabled or self.algorithm == "none":
            return DuplicateDecision.unique()
        return self.check_value(self.fingerprint_of(image).value, candidate_id=candidate_id)

    def check_embedding(self, vector: tuple[float, ...], *, candidate_id: str = "") -> DuplicateDecision:
        """Decide whether an embedding is too similar to an accepted output's embedding."""
        if not self.enabled or not self._embeddings:
            return DuplicateDecision.unique()
        best_reference = ""
        best_similarity = -1.0
        norm = _norm(vector)
        for reference, other in self._embeddings.items():
            if reference == candidate_id:
                continue
            similarity = _cosine(vector, other, norm=norm)
            if similarity > best_similarity:
                best_similarity = similarity
                best_reference = reference
        if best_similarity >= self.embedding_threshold and best_reference:
            return DuplicateDecision(
                is_duplicate=True,
                kind=DuplicateKind.NEAR,
                reference=best_reference,
                distance=float(1.0 - best_similarity),
            )
        return DuplicateDecision.unique()

    @property
    def tracked(self) -> int:
        """How many fingerprints the index holds."""
        return len(self._fingerprints) + len(self._accepted)


def _norm(vector: tuple[float, ...]) -> float:
    return float(np.sqrt(sum(value * value for value in vector)))


def _cosine(left: tuple[float, ...], right: tuple[float, ...], *, norm: float | None = None) -> float:
    if len(left) != len(right):
        return 0.0
    right_norm = float(np.sqrt(sum(value * value for value in right)))
    denominator = (norm if norm is not None else _norm(left)) * right_norm
    if denominator == 0.0:
        return 0.0
    return float(sum(a * b for a, b in zip(left, right, strict=True)) / denominator)
