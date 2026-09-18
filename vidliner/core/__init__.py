"""Core primitives shared by every VidLiner layer.

Nothing in this package knows about images, models, vendors, or files. It provides:

* :mod:`vidliner.core.canonical` — canonical JSON, digests, identifiers.
* :mod:`vidliner.core.identity` — derived identities for samples, objects, candidates, nodes.
* :mod:`vidliner.core.seedtree` — deterministic seed derivation.
* :mod:`vidliner.core.errors` — the failure taxonomy.
* :mod:`vidliner.core.graph` — the operation graph and its validation.
* :mod:`vidliner.core.results` — artifact references, node results, evidence, job summaries.
* :mod:`vidliner.core.cache` — node cache keys.
"""

from __future__ import annotations

from vidliner.core.errors import (
    BackendFailure,
    Cancellation,
    ErrorCode,
    FailureClass,
    InfrastructureFailure,
    OperatorFailure,
    QualityReject,
    ValidationFailure,
    VidlinerError,
    describe_failure,
)
from vidliner.core.graph import OperationGraph, OperationNode, RetryPolicy
from vidliner.core.results import ArtifactRef, JobCounters, JobSummary, NodeResult, PortValue, utc_now
from vidliner.core.seedtree import SeedTree, derive_seed

__all__ = [
    "ArtifactRef",
    "BackendFailure",
    "Cancellation",
    "ErrorCode",
    "FailureClass",
    "InfrastructureFailure",
    "JobCounters",
    "JobSummary",
    "NodeResult",
    "OperationGraph",
    "OperationNode",
    "OperatorFailure",
    "PortValue",
    "QualityReject",
    "RetryPolicy",
    "SeedTree",
    "ValidationFailure",
    "VidlinerError",
    "derive_seed",
    "describe_failure",
    "utc_now",
]
