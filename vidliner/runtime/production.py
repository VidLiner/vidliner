"""The production guard: a demonstration stack may run a job, but it may not produce training data.

VidLiner ships a profile that works out of the box: a saliency detector, a box-prompt segmenter, a
synthetic generator, and colour-appearance evaluators. Together they exercise every stage, every
cache path, and every gate, which is what makes the demo and the test suite possible without a model
download or a paid API.

They are also stand-ins. A saliency detector reports "something is here" rather than a class, a
synthetic generator paints a flat colour rather than a car, and a colour heuristic cannot judge
whether an SUV looks like an SUV. A dataset produced with them is *shaped* like training data and is
not training data.

The guard therefore separates two things that used to be the same verdict:

* **can the job run?** — yes, with any bound backend. Nothing here blocks execution, because the
  demo, the tests, and a first look at a new dataset all depend on it.
* **may the result be exported?** — only when the four capabilities whose output *becomes* the
  training data are served by production backends, or when the recipe explicitly says it knows what
  it is asking for.

The check runs before generation, so a job that could never be exported fails in a second rather
than after spending money on candidates nobody may keep.
"""

from __future__ import annotations

from dataclasses import dataclass

from vidliner.capabilities.names import PRODUCTION_CAPABILITIES, describe_capability
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.runtime.registry import BackendRegistry

__all__ = ["DemoUsage", "ProductionVerdict", "assert_production_ready", "assess_production_readiness"]


@dataclass(frozen=True, slots=True)
class DemoUsage:
    """One production-critical capability that is served by a demonstration backend."""

    capability: str
    backend: str
    reason: str
    """Where the demo declaration came from: the profile entry or the backend's own probe."""

    @property
    def description(self) -> str:
        """What the capability is for, for the refusal message."""
        try:
            return describe_capability(self.capability)
        except KeyError:  # pragma: no cover - the vocabulary is closed
            return "unknown capability"


@dataclass(frozen=True, slots=True)
class ProductionVerdict:
    """The result of asking whether a job's bindings may produce training data."""

    demo_usage: tuple[DemoUsage, ...] = ()
    considered: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def is_production_ready(self) -> bool:
        """True when every production-critical capability is served by a real implementation."""
        return not self.demo_usage

    @property
    def capabilities_involved(self) -> tuple[str, ...]:
        """The production-critical capabilities with a demonstration backend."""
        return tuple(usage.capability for usage in self.demo_usage)

    def explanation(self) -> tuple[str, ...]:
        """Human-readable lines explaining every demonstration binding that was found."""
        return tuple(
            f"{usage.capability} ({usage.description}) is served by {usage.backend!r}, "
            f"which declares itself a demonstration stand-in ({usage.reason})"
            for usage in self.demo_usage
        )


def assess_production_readiness(
    registry: BackendRegistry,
    bindings: dict[str, str],
) -> ProductionVerdict:
    """Decide whether the bindings behind a job may produce training data.

    Args:
        registry: the runtime profile's backend registry, used for declared capabilities.
        bindings: ``capability → backend`` for the job about to run.

    Returns:
        The verdict, listing every production-critical capability that resolved to a demonstration
        backend. Capabilities outside :data:`PRODUCTION_CAPABILITIES` are not considered: a
        demonstration planner or refiner changes *how* a sample was made, not whether its label
        describes the picture, and refusing those would make the guard unusable.
    """
    profile = registry.profile
    usage: list[DemoUsage] = []
    for capability in PRODUCTION_CAPABILITIES:
        backend = bindings.get(capability)
        if backend is None:
            continue
        spec = profile.backends.get(backend)
        if spec is not None and spec.demo_only:
            usage.append(DemoUsage(capability, backend, reason="declared in the runtime profile"))
            continue
        # A backend may instead declare it about itself, which the block below reads from its class
        # rather than from its probe: probing imports and constructs backends, and the guard runs
        # before generation precisely to avoid that cost.
        if _class_declares_demo(registry, backend):
            usage.append(DemoUsage(capability, backend, reason="declared by the backend itself"))
    return ProductionVerdict(demo_usage=tuple(usage), considered=PRODUCTION_CAPABILITIES)


def assert_production_ready(verdict: ProductionVerdict) -> None:
    """Refuse to continue when a demonstration binding would produce training data.

    Raises:
        ValidationFailure: listing every offending capability, so one edit to the profile fixes all
            of them instead of surfacing them one job at a time.
    """
    if verdict.is_production_ready:
        return
    raise ValidationFailure(
        "this job would export a dataset produced with demonstration backends: "
        + "; ".join(verdict.explanation()),
        code=ErrorCode.DEMO_BACKEND_NOT_ALLOWED,
        detail={
            "capabilities": list(verdict.capabilities_involved),
            "bindings": {usage.capability: usage.backend for usage in verdict.demo_usage},
            "remedy": (
                "bind production backends for these capabilities in the runtime profile, or set "
                "acceptance.allow_demo_backends: true to export a demonstration dataset deliberately"
            ),
        },
    )


def _class_declares_demo(registry: BackendRegistry, backend: str) -> bool:
    """Whether the backend class itself declares the demo-only marker.

    Read from the class rather than from a probe so the guard stays cheap: `vidliner plan` and the
    pre-flight both run before anything is instantiated or downloaded.
    """
    spec = registry.profile.backends.get(backend)
    if spec is None:
        return False
    from vidliner.runtime.registry import import_attribute

    try:
        factory = import_attribute(spec.use)
    except ValidationFailure:
        return False
    return bool(getattr(factory, "demo_only", False))
