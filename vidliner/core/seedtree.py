"""Deterministic seed derivation.

VidLiner never calls module-level randomness. Every random decision a backend makes is driven by a
seed that was derived from the job seed along the identity path of the work being done:

``job_seed → sample_seed → target_seed → candidate_seed``

Deriving by *label* rather than by *counter* is what makes the tree stable: adding a fourth
candidate value to a recipe does not change the seed of the first three, so their cached outputs
remain valid.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

__all__ = ["MAX_SEED", "SeedTree", "derive_seed"]

MAX_SEED: Final = 2**63 - 1


def derive_seed(parent: int, *labels: str | int) -> int:
    """Derive a child seed from a parent seed and an ordered label path.

    The derivation is a keyed hash, not an arithmetic operation on the parent, so sibling labels
    cannot be guessed from one another and the result is stable across processes and platforms.
    """
    if not 0 <= parent <= MAX_SEED:
        raise ValueError(f"parent seed {parent} is outside the supported range")
    material = "\u0000".join([str(parent), *(str(label) for label in labels)])
    digest = hashlib.blake2b(material.encode("utf-8"), digest_size=8, person=b"vidliner").digest()
    return int.from_bytes(digest, "big") % (MAX_SEED + 1)


@dataclass(frozen=True, slots=True)
class SeedTree:
    """A node in the derived-seed tree.

    Attributes:
        seed: the seed at this node, suitable for handing to a random number generator.
        path: the label path from the job root, used for logging and for the ``seed_tree`` record
            in the job manifest.
    """

    seed: int
    path: tuple[str, ...] = ()

    def child(self, *labels: str | int) -> SeedTree:
        """Return the child node reached by appending ``labels`` to the path."""
        return SeedTree(
            seed=derive_seed(self.seed, *labels), path=(*self.path, *(str(label) for label in labels))
        )

    @property
    def path_label(self) -> str:
        """The label path joined for display."""
        return "/".join(self.path) if self.path else "<root>"

    def for_sample(self, sample: str) -> SeedTree:
        """Derive the seed for one dataset sample."""
        return self.child("sample", sample)

    def for_object(self, sample: str, target_object: str) -> SeedTree:
        """Derive the seed for one target object inside a sample."""
        return self.for_sample(sample).child("object", target_object)

    def for_candidate(self, sample: str, target_object: str, candidate_key: str) -> SeedTree:
        """Derive the seed for one replacement candidate of one target object."""
        return self.for_object(sample, target_object).child("candidate", candidate_key)

    def for_node(self, node: str) -> SeedTree:
        """Derive the seed for one graph node execution."""
        return self.child("node", node)
