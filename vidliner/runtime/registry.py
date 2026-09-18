"""Backend registry and capability resolution.

Resolution rules, in order, and never by guessing:

1. an explicit ``bindings[capability]`` entry wins;
2. otherwise a backend whose *declared* capabilities contain the capability is used, provided
   exactly one matches;
3. otherwise the registry instantiates enabled backends and asks them what they serve;
4. if zero or more than one candidate remains, resolution fails with a precise error.

Rule 2 exists so that ``plan`` can report a fully bound graph without importing or instantiating a
single backend, which is what keeps dry runs free and fast.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Protocol

from vidliner.capabilities.backend import Backend, BackendProbe, PipelineContext
from vidliner.capabilities.names import is_known_capability
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.runtime.profile import BackendSpec, RuntimeProfile
from vidliner.runtime.secrets import CredentialResolver

__all__ = [
    "BackendRegistry",
    "ResolutionReport",
    "ResolvedBinding",
    "import_attribute",
]


class BackendFactory(Protocol):
    """Callable that builds a backend from its spec, options, and credential resolver."""

    def __call__(
        self, spec: BackendSpec, options: dict[str, Any], credentials: CredentialResolver
    ) -> Backend: ...


def import_attribute(path: str) -> Any:
    """Import ``package.module:Attribute`` or ``package.module.Attribute``.

    Raises:
        ValidationFailure: when the module or attribute cannot be imported.
    """
    module_path, separator, attribute = path.partition(":")
    if not separator:
        module_path, _, attribute = path.rpartition(".")
    if not module_path or not attribute:
        raise ValidationFailure(
            f"backend import path {path!r} must look like 'package.module:Attribute'",
            code=ErrorCode.BACKEND_NOT_FOUND,
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValidationFailure(
            f"backend module {module_path!r} could not be imported: {exc}",
            code=ErrorCode.BACKEND_NOT_FOUND,
            detail={"use": path},
        ) from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ValidationFailure(
            f"backend module {module_path!r} has no attribute {attribute!r}",
            code=ErrorCode.BACKEND_NOT_FOUND,
            detail={"use": path},
        ) from exc


@dataclass(frozen=True, slots=True)
class ResolvedBinding:
    """One capability resolved to a backend name."""

    capability: str
    backend: str
    reason: str


@dataclass(frozen=True, slots=True)
class ResolutionReport:
    """The outcome of resolving a set of required capabilities."""

    bindings: dict[str, ResolvedBinding] = field(default_factory=dict)
    unmet: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        """True when every requested capability resolved."""
        return not self.unmet

    def backend_for(self, capability: str) -> str | None:
        """Backend name bound to ``capability``, if any."""
        binding = self.bindings.get(capability)
        return binding.backend if binding else None

    def as_mapping(self) -> dict[str, str]:
        """``capability -> backend`` mapping for manifests."""
        return {capability: binding.backend for capability, binding in sorted(self.bindings.items())}


class BackendRegistry:
    """Instantiates and caches backend objects for one runtime profile."""

    def __init__(self, profile: RuntimeProfile, credentials: CredentialResolver | None = None) -> None:
        self._profile = profile
        self._credentials = credentials or CredentialResolver()
        self._instances: dict[str, Backend] = {}
        self._capability_index: dict[str, list[str]] | None = None

    @property
    def profile(self) -> RuntimeProfile:
        """The profile this registry was built from."""
        return self._profile

    @property
    def credentials(self) -> CredentialResolver:
        """The credential resolver handed to backends."""
        return self._credentials

    def _instantiate(self, name: str, spec: BackendSpec) -> Backend:
        if name in self._instances:
            return self._instances[name]
        factory = import_attribute(spec.use)
        if not callable(factory):
            raise ValidationFailure(
                f"backend {name!r} resolves to a non-callable object at {spec.use!r}",
                code=ErrorCode.BACKEND_NOT_FOUND,
            )
        options = {**spec.options, "device": self._profile.device_for(name)}
        try:
            backend = factory(spec, options, self._credentials)
        except ValidationFailure:
            raise
        except Exception as exc:
            raise ValidationFailure(
                f"backend {name!r} could not be constructed from {spec.use!r}: {exc}",
                code=ErrorCode.BACKEND_UNAVAILABLE,
                detail={"backend": name, "use": spec.use},
            ) from exc
        backend_id = getattr(backend, "backend_id", None)
        if not isinstance(backend_id, str) or not backend_id:
            raise ValidationFailure(
                f"backend {name!r} does not expose a non-empty backend_id",
                code=ErrorCode.BACKEND_NOT_FOUND,
                detail={"backend": name},
            )
        self._instances[name] = backend
        self._capability_index = None
        return backend

    def backend(self, name: str) -> Backend:
        """Return the instantiated backend registered under ``name``.

        Raises:
            ValidationFailure: when the name is not configured or is disabled.
        """
        spec = self._profile.enabled_backends().get(name)
        if spec is None:
            raise ValidationFailure(
                f"backend {name!r} is not configured or is disabled in profile {self._profile.profile!r}",
                code=ErrorCode.BACKEND_NOT_FOUND,
                detail={"backend": name},
            )
        return self._instantiate(name, spec)

    def declared_capabilities(self, name: str) -> tuple[str, ...]:
        """Capabilities a backend declares without being instantiated.

        An explicit ``capabilities`` list in the backend spec is authoritative; otherwise the
        backend class may expose a ``declared_capabilities`` class attribute.
        """
        spec = self._profile.backends.get(name)
        if spec is None:
            return ()
        declared = spec.options.get("declared_capabilities")
        if isinstance(declared, (list, tuple)) and all(isinstance(item, str) for item in declared):
            return tuple(declared)
        try:
            factory = import_attribute(spec.use)
        except ValidationFailure:
            return ()
        attribute = getattr(factory, "declared_capabilities", ())
        if isinstance(attribute, (list, tuple)):
            return tuple(str(item) for item in attribute)
        return ()

    def capability_index(self, *, instantiate: bool = True) -> dict[str, list[str]]:
        """Map every available capability to the backends that advertise it.

        Args:
            instantiate: when true, backends without a declared capability list are instantiated so
                they can be asked. When false, only declared capabilities are considered and the
                result may be incomplete — callers that must not import heavy modules pass ``False``.
        """
        if self._capability_index is not None and instantiate:
            return self._capability_index
        index: dict[str, list[str]] = {}
        for name in self._profile.backend_names():
            capabilities = self.declared_capabilities(name)
            if not capabilities and instantiate:
                backend = self.backend(name)
                capabilities = tuple(backend.capabilities)
            for capability in capabilities:
                index.setdefault(capability, []).append(name)
        if instantiate:
            self._capability_index = index
        return index

    def resolve(self, required: tuple[str, ...], *, instantiate: bool = True) -> ResolutionReport:
        """Resolve every capability in ``required`` to a backend name.

        Args:
            required: capability names the job needs.
            instantiate: whether unresolved capabilities may be discovered by instantiating
                backends. ``plan`` passes ``False`` when it must not import backend modules.

        Raises:
            ValidationFailure: when a capability name is not part of the vocabulary.
        """
        for capability in required:
            if not is_known_capability(capability):
                raise ValidationFailure(
                    f"{capability!r} is not a known capability",
                    code=ErrorCode.CAPABILITY_UNBOUND,
                    detail={"capability": capability},
                )
        index = self.capability_index(instantiate=instantiate) if instantiate else self._declared_index()
        bindings: dict[str, ResolvedBinding] = {}
        unmet: list[str] = []
        notes: list[str] = []
        for capability in required:
            explicit = self._profile.binding_for(capability)
            if explicit is not None:
                bindings[capability] = ResolvedBinding(capability, explicit, "explicit binding")
                continue
            candidates = index.get(capability, [])
            if len(candidates) == 1:
                bindings[capability] = ResolvedBinding(capability, candidates[0], "sole provider")
                continue
            if not candidates:
                unmet.append(capability)
                continue
            raise ValidationFailure(
                f"capability {capability} is advertised by {len(candidates)} backends "
                f"({', '.join(sorted(candidates))}); add an explicit binding to choose one",
                code=ErrorCode.BINDING_AMBIGUOUS,
                detail={"capability": capability, "candidates": sorted(candidates)},
            )
        if unmet and not instantiate:
            notes.append(
                "capabilities were resolved from declared capability lists only; "
                "run 'vidliner backend check' to confirm by probing backends"
            )
        return ResolutionReport(bindings=bindings, unmet=tuple(unmet), notes=tuple(notes))

    def _declared_index(self) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for name in self._profile.backend_names():
            for capability in self.declared_capabilities(name):
                index.setdefault(capability, []).append(name)
        return index

    async def probe_all(self) -> dict[str, BackendProbe]:
        """Probe every enabled backend, converting construction failures into unavailable probes."""
        results: dict[str, BackendProbe] = {}
        for name in self._profile.backend_names():
            try:
                backend = self.backend(name)
            except ValidationFailure as exc:
                results[name] = BackendProbe(
                    backend_id=name,
                    version="unknown",
                    health="unavailable",
                    message=exc.message,
                    details={"code": exc.code.value},
                )
                continue
            try:
                results[name] = await backend.probe()
            except Exception as exc:  # a broken probe must not break the report
                results[name] = BackendProbe(
                    backend_id=name,
                    version=getattr(backend, "backend_version", "unknown"),
                    health="unavailable",
                    message=f"probe failed: {type(exc).__name__}: {exc}",
                )
        return results

    def context(
        self, *, job_id: str, node_id: str, seed: int, backend: str, config: dict[str, Any]
    ) -> PipelineContext:
        """Build a :class:`PipelineContext` for one backend invocation."""
        spec = self._profile.backends.get(backend)
        return PipelineContext(
            job_id=job_id,
            node_id=node_id,
            seed=seed,
            device=self._profile.device_for(backend),
            config=config,
            metadata={
                "options": dict(spec.options) if spec else {},
                "capacity": {"limit": spec.capacity.limit, "period_s": spec.capacity.period_s}
                if spec
                else {},
                "estimates": spec.estimates.model_dump() if spec else {},
            },
        )
