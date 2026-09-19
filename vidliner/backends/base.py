"""Shared plumbing for the built-in backends.

Every built-in backend is *blocking*: it does local numpy/Pillow work. They all implement
``probe()`` the same way, and they all keep their synchronous implementations on ``self._sync`` so
the runtime can run them on a worker thread.
"""

from __future__ import annotations

import time
from typing import Any

from vidliner.capabilities.backend import BackendProbe
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.enums import Determinism
from vidliner.runtime.profile import BackendSpec

__all__ = ["LocalBackend", "SyncCalls", "timed"]


def timed() -> float:
    """Start a monotonic timer, for reporting execution duration."""
    return time.perf_counter()


def elapsed_ms(started: float) -> int:
    """Milliseconds since ``started``."""
    return int((time.perf_counter() - started) * 1000)


class SyncCalls:
    """Namespace holding a backend's synchronous implementations.

    The runtime looks up ``backend._sync.<method>`` by protocol method name, so a backend declares
    its capabilities once and does not repeat itself for the thread offload path.
    """

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - defensive
        raise AttributeError(f"this backend has no synchronous implementation named {name!r}")


class LocalBackend:
    """Base class for the built-in local backends.

    Subclasses set :attr:`backend_id`, :attr:`backend_version`, :attr:`capabilities`,
    :attr:`determinism`, and bind their synchronous methods onto ``self._sync``.
    """

    backend_id: str = "local"
    backend_version: str = "1.0.0"
    capabilities: tuple[str, ...] = ()
    determinism: Determinism = Determinism.DETERMINISTIC
    safe_to_retry: bool = True
    external: bool = False
    blocking: bool = True
    device: str = "cpu"
    demo_only: bool = False
    """Whether this backend is a demonstration stand-in rather than a production implementation.

    Declared on the class *and* in the runtime profile, deliberately: the class states the truth about
    the implementation, and the profile entry lets a deployment record which binding it selected. The
    production guard treats either declaration as authoritative, so a backend cannot be promoted to
    production by editing only one of them.
    """

    def __init__(self, spec: BackendSpec, options: dict[str, Any], credentials: object | None = None) -> None:
        self.spec = spec
        self.options = dict(options or {})
        self.credentials = credentials
        self._sync = SyncCalls()

    # -- helpers used by subclasses ---------------------------------------- #

    def option(self, name: str, default: Any = None) -> Any:
        """Read a backend option with a default."""
        return self.options.get(name, default)

    def require_option(self, name: str) -> Any:
        """Read a required backend option.

        Raises:
            ValidationFailure: when the option is absent, so a misconfigured backend fails at
                pre-flight rather than mid-job.
        """
        value = self.options.get(name)
        if value is None:
            raise ValidationFailure(
                f"backend {self.backend_id} requires the {name!r} option",
                code=ErrorCode.BACKEND_NOT_FOUND,
                detail={"backend": self.backend_id, "option": name},
            )
        return value

    def model_id(self) -> str:
        """Identifier of the model this adapter wraps, for provenance."""
        return str(self.option("model_id", f"{self.backend_id}-builtin"))

    # -- protocol ---------------------------------------------------------- #

    async def probe(self) -> BackendProbe:
        """Report readiness. Local backends are ready unless a declared option is unusable."""
        return BackendProbe(
            backend_id=self.backend_id,
            version=self.backend_version,
            health="ready",
            capabilities=self.capabilities,
            device=str(self.option("device", self.device)),
            determinism=self.determinism,
            safe_to_retry=self.safe_to_retry,
            external=self.external,
            demo_only=self.demo_only,
            message="local built-in backend",
            details={"adapter": type(self).__name__},
        )
