"""Node result caching.

A node's cache key is a pure function of what would change its output (ADR-005): the operator, its
implementation version, the identity of the backend that ran it, the resolved node configuration,
the digests of every input artifact, and the seed the node would use. Anything not in that list —
wall-clock time, output paths, log verbosity — cannot change a cached result, so it is not in the
key.

Nodes that are not deterministic for a given seed are simply never cached.
"""

from __future__ import annotations

from typing import Final

from vidliner.core.canonical import canonical_json, digest_bytes
from vidliner.core.graph import OperationNode

__all__ = ["CacheDecision", "cache_key", "cacheable", "node_deterministic"]

CACHE_KEY_VERSION: Final = "vidliner-cache-1"


class CacheDecision:
    """Outcome of a cache lookup."""

    __slots__ = ("hit", "outputs", "payload")

    def __init__(
        self, *, hit: bool, payload: dict[str, object] | None = None, outputs: dict[str, str] | None = None
    ) -> None:
        self.hit = hit
        self.payload = payload or {}
        self.outputs = outputs or {}


def node_deterministic(node: OperationNode) -> bool:
    """Whether repeating this node with identical inputs is guaranteed to reproduce its output.

    A *seeded* node is reproducible only because the engine supplies a derived seed; that is enough
    for caching as long as the seed is part of the key, which it is.
    """
    return node.determinism.value in {"deterministic", "seeded"}


def cacheable(node: OperationNode) -> bool:
    """Whether the node's result may be stored in, and served from, the node cache."""
    return node.cacheable and node_deterministic(node)


def cache_key(
    node: OperationNode,
    *,
    implementation: str,
    input_digests: dict[str, str],
    seed: int,
) -> str:
    """Compute the cache key for one node execution.

    Args:
        node: the node being executed.
        implementation: identity of the code that will run, e.g. ``"operator:ingest.image@1.0.0"``
            or a backend id when the node delegates to a backend. Two different implementations of
            the same operator must not share cache entries.
        input_digests: port name to artifact digest, for every artifact-valued input.
        seed: the derived seed the node would run with.

    Returns:
        Lower-case hex digest usable as a primary key.
    """
    material = canonical_json(
        {
            "version": CACHE_KEY_VERSION,
            "operator": node.operator,
            "operator_version": node.operator_version,
            "implementation": implementation,
            "config": node.config,
            "inputs": dict(sorted(input_digests.items())),
            "seed": seed,
        }
    )
    return digest_bytes(material.encode("utf-8"))
