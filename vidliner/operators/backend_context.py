"""Construction of the backend-facing context.

A backend is handed a narrow view of the run: its own identity, the node's seed and device, the
node's configuration, and an artifact I/O object. It does not receive the workspace, the job store,
the recipe, or any credential beyond what its own spec resolves.

This lives in its own module so every operator builds the same view, and so the shape of that view is
reviewable in one place rather than scattered across a dozen operators.
"""

from __future__ import annotations

from typing import Any

from vidliner.capabilities.backend import PipelineContext
from vidliner.operators.base import ExecutionContext

__all__ = ["backend_context"]


def backend_context(context: ExecutionContext, **extra: Any) -> PipelineContext:
    """Build the :class:`PipelineContext` a backend receives for one node execution."""
    provider = context.backends
    metadata = dict(getattr(provider, "metadata", {}) or {}) if provider is not None else {}
    return PipelineContext(
        job_id=context.job_id,
        node_id=context.node.node_id,
        seed=context.seed,
        device=str(metadata.get("device", "cpu")),
        config=dict(context.config),
        metadata={**metadata, **extra},
        io=context.io,
    )
