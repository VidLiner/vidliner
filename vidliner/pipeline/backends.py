"""Capability brokering: from a capability name to a live backend call.

Operators ask for a capability; this module turns that into a concrete backend, enforces the
backend's capacity limits, and records which backend served the node so the run's evidence can name
it. It is the only bridge between the capability vocabulary and the runtime profile, and it is
constructed per job so resolution failures are reported once, at pre-flight, rather than per node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vidliner.capabilities.backend import Backend
from vidliner.capabilities.names import describe_capability
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.runtime.io import BackendLimiter
from vidliner.runtime.registry import BackendRegistry, ResolutionReport
from vidliner.storage.workspace import ArtifactStore

__all__ = ["CapabilityBroker", "ResolvedCapability"]


@dataclass(frozen=True, slots=True)
class ResolvedCapability:
    """A capability bound to a specific backend."""

    capability: str
    backend_name: str
    backend: Backend


@dataclass
class CapabilityHandle:
    """A resolved capability that an operator can call."""

    capability: str
    backend_name: str
    backend: Backend
    broker: CapabilityBroker

    @property
    def backend_id(self) -> str:
        """Identifier of the resolved backend."""
        return str(self.backend.backend_id)

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Capabilities the backend advertises."""
        return tuple(self.backend.capabilities)

    async def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke one capability method under the backend's capacity budget.

        Raises:
            ValidationFailure: when the backend does not implement ``method``, which means the
                profile bound a backend that does not satisfy the operator's contract.
        """
        function = getattr(self.backend, method, None)
        if function is None or not callable(function):
            raise ValidationFailure(
                f"backend {self.backend_name!r} bound to {self.capability} does not implement "
                f"{method!r}; it advertises {sorted(self.backend.capabilities)}",
                code=ErrorCode.CAPABILITY_UNBOUND,
                detail={
                    "capability": self.capability,
                    "backend": self.backend_name,
                    "method": method,
                },
            )
        self.broker.record_use(self.backend_name)
        return await self.broker.limiter.run(self.backend_name, self.backend, function, *args, **kwargs)


@dataclass
class CapabilityBroker:
    """Resolves capabilities and invokes backends for one job."""

    registry: BackendRegistry
    limiter: BackendLimiter = field(init=False)
    last_backend_id: str | None = None
    _handles: dict[str, CapabilityHandle] = field(default_factory=dict, init=False, repr=False)
    _used: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.limiter = BackendLimiter(self.registry)

    # -- resolution -------------------------------------------------------- #

    def resolve(self, capabilities: tuple[str, ...]) -> ResolutionReport:
        """Resolve every capability a graph requires."""
        return self.registry.resolve(capabilities, instantiate=True)

    def handle(self, capability: str) -> CapabilityHandle:
        """Return a handle for one capability.

        Raises:
            ValidationFailure: when the capability cannot be resolved, with a message that names the
                capability, what it means, and what the profile currently offers.
        """
        handle = self._handles.get(capability)
        if handle is not None:
            return handle
        report = self.registry.resolve((capability,), instantiate=True)
        backend_name = report.backend_for(capability)
        if backend_name is None:
            available = sorted(self.registry.capability_index(instantiate=True))
            raise ValidationFailure(
                f"no backend is bound to capability {capability} ({describe_capability(capability)}); "
                f"the runtime profile provides {available}",
                code=ErrorCode.CAPABILITY_UNBOUND,
                detail={"capability": capability, "available": available},
            )
        backend = self.registry.backend(backend_name)
        created = CapabilityHandle(
            capability=capability,
            backend_name=backend_name,
            backend=backend,
            broker=self,
        )
        self._handles[capability] = created
        return created

    # -- bookkeeping ------------------------------------------------------- #

    def record_use(self, backend_name: str) -> None:
        """Remember that a backend served a call, for the job manifest."""
        self.last_backend_id = backend_name
        if backend_name not in self._used:
            self._used.append(backend_name)

    @property
    def used_backends(self) -> tuple[str, ...]:
        """Every backend that served at least one call during this job."""
        return tuple(self._used)

    @property
    def metadata(self) -> dict[str, Any]:
        """Extra context passed to backends, currently the resolved device."""
        return {"device": self.registry.profile.device}

    def binding_report(self) -> dict[str, str]:
        """``capability → backend`` for every capability resolved so far."""
        return {capability: handle.backend_name for capability, handle in sorted(self._handles.items())}


def build_broker(
    profile_registry: BackendRegistry, *, store: ArtifactStore | None = None
) -> CapabilityBroker:
    """Construct a broker for a job, optionally verifying the artifact store is usable."""
    if store is not None and not store.workspace.is_initialized():
        raise ValidationFailure(
            f"workspace {store.workspace.root} is not initialised; run 'vidliner init' first",
            code=ErrorCode.WORKSPACE_INVALID,
            detail={"workspace": str(store.workspace.root)},
        )
    return CapabilityBroker(registry=profile_registry)
