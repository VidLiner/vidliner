"""Shared helpers for operators.

Operators share four concerns and nothing else:

* **backend access** — an operator asks the context for a handle on the backend bound to a
  capability; the handle already enforces that backend's capacity limits;
* **storage access** — artifacts go in and out of the store through the context's I/O object;
* **configuration** — node config is validated once at plan time by the registry, and read at run
  time through the context;
* **port payloads** — the internal IR crosses edges in a form that survives a persisted node result.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vidliner.core.errors import ErrorCode, OperatorFailure, ValidationFailure
from vidliner.domain.annotations import AnnotationBundle
from vidliner.domain.instances import ObjectInstance
from vidliner.operators.base import ExecutionContext

__all__ = [
    "BackendHandle",
    "annotation_from_payload",
    "annotation_to_payload",
    "backend_handle",
    "instance_from_payload",
    "instance_to_payload",
    "require_instances",
    "store",
]


@runtime_checkable
class BackendHandle(Protocol):
    """A capability already resolved to a concrete backend, with its capacity enforced."""

    @property
    def backend_id(self) -> str:
        """Identifier of the resolved backend, for evidence and provenance."""
        ...

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Capabilities the resolved backend advertises."""
        ...

    async def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke one capability method."""
        ...


async def backend_handle(context: ExecutionContext, capability: str) -> BackendHandle:
    """Resolve a capability to a backend handle.

    Raises:
        OperatorFailure: when the execution context carries no backend resolver, which means the
            node ran outside a runtime and is a wiring bug that must be loud.
    """
    resolver = context.backends
    if resolver is None:
        raise OperatorFailure(
            f"operator {context.node.operator} needs capability {capability} but no backend resolver "
            "was attached to the execution context",
            code=ErrorCode.CAPABILITY_UNBOUND,
            detail={"node_id": context.node.node_id, "capability": capability},
        )
    handle = resolver(capability) if callable(resolver) else None
    if handle is None or not isinstance(handle, BackendHandle):  # pragma: no cover - defensive
        raise OperatorFailure(
            f"capability {capability} did not resolve to a usable backend handle",
            code=ErrorCode.CAPABILITY_UNBOUND,
            detail={"node_id": context.node.node_id, "capability": capability},
        )
    return handle


def store(context: ExecutionContext) -> Any:
    """The artifact store behind the context's I/O object."""
    return context.io.store


def instance_to_payload(instance: ObjectInstance) -> dict[str, Any]:
    """Serialize an object instance for a port payload."""
    return instance.model_dump(mode="json")


def instance_from_payload(payload: dict[str, Any]) -> ObjectInstance:
    """Rebuild an object instance from a port payload.

    Raises:
        ValidationFailure: when the payload is not a valid instance, so a corrupted upstream output
            is reported rather than silently treated as an empty list.
    """
    try:
        return ObjectInstance.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError
        raise ValidationFailure(
            f"port payload is not a valid object instance: {exc}",
            code=ErrorCode.PORT_TYPE_MISMATCH,
        ) from exc


def require_instances(value: Any, *, where: str) -> tuple[ObjectInstance, ...]:
    """Coerce a value on an ``instances`` port into a tuple of :class:`ObjectInstance`.

    Accepts both the runtime form (models) and the persisted form (mappings), so the same operator
    works whether its input came from a live upstream node or from a resumed run.
    """
    if value is None:
        raise ValidationFailure(
            f"{where} received no value on its instances port",
            code=ErrorCode.PORT_UNBOUND,
            detail={"where": where},
        )
    items = list(value) if isinstance(value, (tuple, list)) else [value]
    resolved: list[ObjectInstance] = []
    for item in items:
        if isinstance(item, ObjectInstance):
            resolved.append(item)
        elif isinstance(item, dict):
            resolved.append(instance_from_payload(item))
        else:
            raise ValidationFailure(
                f"{where} received an instances port value of unsupported type {type(item).__name__}",
                code=ErrorCode.PORT_TYPE_MISMATCH,
                detail={"where": where, "type": type(item).__name__},
            )
    return tuple(resolved)


def annotation_to_payload(bundle: AnnotationBundle) -> dict[str, Any]:
    """Serialize an annotation bundle for a port payload."""
    return bundle.model_dump(mode="json")


def annotation_from_payload(payload: Any) -> AnnotationBundle:
    """Rebuild an annotation bundle from a port payload.

    Raises:
        ValidationFailure: when the payload is not a valid bundle.
    """
    if isinstance(payload, AnnotationBundle):
        return payload
    if not isinstance(payload, dict):
        raise ValidationFailure(
            f"annotation port expects a mapping, received {type(payload).__name__}",
            code=ErrorCode.PORT_TYPE_MISMATCH,
        )
    try:
        return AnnotationBundle.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError
        raise ValidationFailure(
            f"annotation port payload is not a valid bundle: {exc}",
            code=ErrorCode.ANNOTATION_INVALID,
        ) from exc
