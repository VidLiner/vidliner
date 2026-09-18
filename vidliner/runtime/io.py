"""Blocking-work dispatch and backend capacity control.

Backends come in two shapes and the engine treats both identically:

* **native async** — an HTTP client or remote service that spends its life awaiting I/O;
* **blocking** — a local CPU/GPU model, a decoder, a subprocess.

A blocking backend still implements the same async protocol methods (so call sites are uniform) but
declares ``blocking = True`` and stores its synchronous implementation on ``_sync``. The runtime then
calls ``_sync`` on a worker thread instead of awaiting the coroutine, which keeps the job's event
loop free without nesting event loops in a thread.

The declaration is explicit. Guessing that a coroutine is "really synchronous" would be exactly the
kind of implicit behaviour that makes a pipeline unpredictable.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from vidliner.capabilities.backend import Backend, BackendProbe
from vidliner.runtime.registry import BackendRegistry

__all__ = ["BackendLimiter", "CapacityBudget", "run_blocking"]

T = TypeVar("T")

_SYNC_ATTRIBUTE = "_sync"


def _callable_name(function: Any) -> str:
    """A callable's name for diagnostics.

    ``__name__`` exists on functions and methods but not on every callable (a partial, a bound
    instance, a mock), so it is read defensively rather than assumed.
    """
    name = getattr(function, "__name__", None)
    return name if isinstance(name, str) else type(function).__name__


async def _invoke(
    backend: Backend,
    function: Callable[..., Awaitable[T]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> T:
    """Invoke one backend capability method, honouring the backend's blocking declaration."""
    if getattr(backend, "blocking", False):
        sync_callable = getattr(backend, _SYNC_ATTRIBUTE, None)
        target = getattr(sync_callable, _callable_name(function), None) if sync_callable is not None else None
        if not callable(target):
            raise _missing_sync(backend, function)
        produced = await asyncio.to_thread(target, *args, **kwargs)
        return cast("T", produced)
    result = function(*args, **kwargs)
    if hasattr(result, "__await__"):
        return await result
    raise _not_awaitable(backend, function)


def _missing_sync(backend: Backend, function: Callable[..., Any]) -> Exception:
    from vidliner.core.errors import ErrorCode, InfrastructureFailure

    return InfrastructureFailure(
        f"blocking backend {backend.backend_id} does not provide _sync.{_callable_name(function)}",
        code=ErrorCode.BACKEND_UNAVAILABLE,
        detail={"backend": backend.backend_id, "method": _callable_name(function)},
    )


def _not_awaitable(backend: Backend, function: Callable[..., Any]) -> Exception:
    from vidliner.core.errors import ErrorCode, InfrastructureFailure

    return InfrastructureFailure(
        f"backend {backend.backend_id} returned a non-awaitable from {_callable_name(function)}",
        code=ErrorCode.BACKEND_RESPONSE_INVALID,
        detail={"backend": backend.backend_id, "method": _callable_name(function)},
    )


def run_blocking(function: Callable[..., T], /, *args: Any, **kwargs: Any) -> Awaitable[T]:
    """Run a synchronous function on a worker thread and await its result."""
    return asyncio.to_thread(function, *args, **kwargs)


@dataclass
class CapacityBudget:
    """A concurrency limit plus an optional replenishing rate budget.

    Attributes:
        limit: maximum concurrent units.
        period_s: when set, at most ``limit`` units may start per period. A rate budget is what
            respects a hosted service's requests-per-minute policy without serialising the client.
    """

    limit: int = 1
    period_s: float | None = None
    _semaphore: asyncio.Semaphore | None = field(default=None, repr=False, compare=False)
    _starts: list[float] = field(default_factory=list, repr=False, compare=False)

    def _sem(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, self.limit))
        return self._semaphore

    async def acquire(self) -> None:
        """Wait until one unit of budget is available."""
        await self._sem().acquire()
        if self.period_s is None:
            return
        while True:
            now = time.monotonic()
            self._starts = [start for start in self._starts if now - start < self.period_s]
            if len(self._starts) < self.limit:
                self._starts.append(now)
                return
            oldest = min(self._starts)
            await asyncio.sleep(max(0.01, self.period_s - (now - oldest)))

    def release(self) -> None:
        """Return one unit of budget."""
        self._sem().release()


class BackendLimiter:
    """Per-backend capacity across one job run."""

    def __init__(self, registry: BackendRegistry) -> None:
        self._registry = registry
        self._budgets: dict[str, CapacityBudget] = {}

    def budget_for(self, name: str) -> CapacityBudget:
        """The budget for a configured backend, created on first use from its spec."""
        budget = self._budgets.get(name)
        if budget is None:
            spec = self._registry.profile.backends.get(name)
            default_limit = self._registry.profile.concurrency.per_backend
            if spec is not None:
                limit, period = spec.capacity.limit, spec.capacity.period_s
            else:
                limit, period = default_limit, None
            budget = CapacityBudget(limit=max(1, limit), period_s=period)
            self._budgets[name] = budget
        return budget

    async def run(
        self,
        name: str,
        backend: Backend,
        function: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run a backend capability call under its budget."""
        budget = self.budget_for(name)
        await budget.acquire()
        try:
            return await _invoke(backend, function, args, kwargs)
        finally:
            budget.release()

    async def probe(self, backend: Backend) -> BackendProbe:
        """Probe a backend off the event loop when it is a blocking implementation."""
        if getattr(backend, "blocking", False):
            return await asyncio.to_thread(lambda: asyncio.run(backend.probe()))
        return await backend.probe()
