"""The operator catalogue.

Registration is explicit: importing :mod:`vidliner.operators` builds the default catalogue once.
Nothing scans the filesystem, and nothing is registered by import side effect on an arbitrary
module, so a deployment reproduces the same catalogue as a test run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from pydantic import ValidationError

from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.core.graph import OperationNode
from vidliner.operators.base import Operator, OperatorSpec

__all__ = [
    "OperatorRegistry",
    "build_default_registry",
    "create_operator",
    "default_registry",
    "describe_operator",
    "operator_names",
    "validate_node_config",
]

OperatorFactory = Callable[[], Operator]


class OperatorRegistry:
    """A name→factory catalogue of operators."""

    def __init__(self) -> None:
        self._factories: dict[str, OperatorFactory] = {}
        self._specs: dict[str, OperatorSpec] = {}

    def register(self, factory: OperatorFactory) -> OperatorFactory:
        """Register an operator factory.

        The spec is read by instantiating the factory once at registration time, so a malformed
        operator fails loudly during catalogue construction rather than mid-job.

        Raises:
            ValidationFailure: when the operator name is already registered with a different
                version, which would make node identity ambiguous.
        """
        probe = factory()
        spec = probe.spec
        existing = self._specs.get(spec.name)
        if existing is not None and existing.version != spec.version:
            raise ValidationFailure(
                f"operator {spec.name!r} is already registered at version {existing.version} "
                f"and cannot also be registered at {spec.version}",
                code=ErrorCode.OPERATOR_UNKNOWN,
                detail={"operator": spec.name, "registered": existing.version, "offered": spec.version},
            )
        self._factories[spec.name] = factory
        self._specs[spec.name] = spec
        return factory

    def operator(self, name: str) -> OperatorFactory:
        """Return the factory for ``name``.

        Raises:
            ValidationFailure: when the operator is unknown.
        """
        try:
            return self._factories[name]
        except KeyError as exc:
            raise ValidationFailure(
                f"unknown operator {name!r}; registered operators are {', '.join(sorted(self._factories))}",
                code=ErrorCode.OPERATOR_UNKNOWN,
                detail={"operator": name},
            ) from exc

    def describe(self, name: str, version: str | None = None) -> OperatorSpec:
        """Return the spec of a registered operator.

        Raises:
            ValidationFailure: when the operator is unknown, or when an explicit ``version`` does
                not match the registered one.
        """
        try:
            spec = self._specs[name]
        except KeyError as exc:
            raise ValidationFailure(
                f"unknown operator {name!r}",
                code=ErrorCode.OPERATOR_UNKNOWN,
                detail={"operator": name},
            ) from exc
        if version is not None and version != spec.version:
            raise ValidationFailure(
                f"operator {name!r} is registered at version {spec.version}, not {version}",
                code=ErrorCode.OPERATOR_UNKNOWN,
                detail={"operator": name, "registered": spec.version, "requested": version},
            )
        return spec

    def create(self, name: str, config: dict[str, Any] | None = None) -> Operator:
        """Instantiate an operator, validating its configuration."""
        spec = self.describe(name)
        if config is not None:
            try:
                spec.config_model.model_validate(config)
            except ValidationError as exc:
                raise ValidationFailure(
                    f"invalid config for operator {name!r}: {exc}",
                    code=ErrorCode.OPERATOR_CONFIG_INVALID,
                    detail={"operator": name},
                ) from exc
        return self.operator(name)()

    def names(self) -> tuple[str, ...]:
        """Every registered operator name, sorted."""
        return tuple(sorted(self._factories))

    def specs(self) -> tuple[OperatorSpec, ...]:
        """Every registered spec, ordered by name."""
        return tuple(self._specs[name] for name in self.names())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._factories

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())


_DEFAULT: OperatorRegistry | None = None


def default_registry() -> OperatorRegistry:
    """The process-wide default catalogue, built on first use."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = build_default_registry()
    return _DEFAULT


def build_default_registry() -> OperatorRegistry:
    """Build a fresh catalogue containing every built-in operator."""
    from vidliner.operators import annotation as annotation_ops
    from vidliner.operators import detection as detection_ops
    from vidliner.operators import evaluation as evaluation_ops
    from vidliner.operators import export as export_ops
    from vidliner.operators import ingest as ingest_ops
    from vidliner.operators import planning as planning_ops
    from vidliner.operators import refinement as refinement_ops
    from vidliner.operators import replacement as replacement_ops
    from vidliner.operators import scene as scene_ops
    from vidliner.operators import segmentation as segmentation_ops
    from vidliner.operators import selection as selection_ops
    from vidliner.operators import verification as verification_ops

    registry = OperatorRegistry()
    for module in (
        ingest_ops,
        detection_ops,
        selection_ops,
        segmentation_ops,
        scene_ops,
        planning_ops,
        replacement_ops,
        refinement_ops,
        verification_ops,
        evaluation_ops,
        annotation_ops,
        export_ops,
    ):
        module.register_all(registry)
    return registry


def describe_operator(name: str, version: str | None = None) -> OperatorSpec:
    """Describe an operator using the default catalogue."""
    return default_registry().describe(name, version)


def create_operator(name: str, config: dict[str, Any] | None = None) -> Operator:
    """Instantiate an operator from the default catalogue."""
    return default_registry().create(name, config)


def operator_names() -> tuple[str, ...]:
    """Every built-in operator name."""
    return default_registry().names()


def validate_node_config(node: OperationNode) -> None:
    """Validate one node's config against its operator's schema.

    Raises:
        ValidationFailure: when the operator is unknown or its config is invalid.
    """
    spec = describe_operator(node.operator, node.operator_version)
    try:
        spec.config_model.model_validate(node.config)
    except ValidationError as exc:
        raise ValidationFailure(
            f"node {node.node_id} has an invalid config for {node.operator!r}: {exc}",
            code=ErrorCode.OPERATOR_CONFIG_INVALID,
            detail={"node_id": node.node_id, "operator": node.operator},
        ) from exc
