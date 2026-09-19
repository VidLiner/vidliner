"""Structured error taxonomy.

The single most important distinction in this module is between *the system went wrong* and *this
generated sample is not good enough to be training data*. ``QualityReject`` is the second kind: it
never fails a job, it produces a ``REJECTED`` candidate. Everything else is a real failure with a
distinct class so that operators, backends, validation problems, and infrastructure problems can be
counted separately in a job summary.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

__all__ = [
    "BackendFailure",
    "Cancellation",
    "ErrorCode",
    "FailureClass",
    "InfrastructureFailure",
    "OperatorFailure",
    "QualityReject",
    "ValidationFailure",
    "VidlinerError",
    "describe_failure",
]


class FailureClass(StrEnum):
    """Top-level classification of a failure."""

    OPERATOR = "operator"
    BACKEND = "backend"
    VALIDATION = "validation"
    QUALITY = "quality"
    INFRASTRUCTURE = "infrastructure"
    CANCELLATION = "cancellation"


class ErrorCode(StrEnum):
    """Stable machine-readable error identifiers.

    Codes are part of the public contract: they appear in manifests, logs, and job summaries, so
    they are never renamed without a migration note.
    """

    # Configuration and pre-flight
    RECIPE_INVALID = "RECIPE_INVALID"
    RUNTIME_INVALID = "RUNTIME_INVALID"
    CAPABILITY_UNBOUND = "CAPABILITY_UNBOUND"
    BINDING_AMBIGUOUS = "BINDING_AMBIGUOUS"
    BACKEND_NOT_FOUND = "BACKEND_NOT_FOUND"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    DEVICE_UNAVAILABLE = "DEVICE_UNAVAILABLE"
    CREDENTIAL_MISSING = "CREDENTIAL_MISSING"
    DEMO_BACKEND_NOT_ALLOWED = "DEMO_BACKEND_NOT_ALLOWED"

    # Graph and planning
    GRAPH_INVALID = "GRAPH_INVALID"
    GRAPH_CYCLE = "GRAPH_CYCLE"
    PORT_TYPE_MISMATCH = "PORT_TYPE_MISMATCH"
    PORT_UNBOUND = "PORT_UNBOUND"
    OPERATOR_UNKNOWN = "OPERATOR_UNKNOWN"
    OPERATOR_CONFIG_INVALID = "OPERATOR_CONFIG_INVALID"
    OPERATOR_NOT_IMPLEMENTED = "OPERATOR_NOT_IMPLEMENTED"

    # Execution
    NODE_TIMEOUT = "NODE_TIMEOUT"
    JOB_TIMEOUT = "JOB_TIMEOUT"
    JOB_CANCELLED = "JOB_CANCELLED"
    DEPENDENCY_FAILED = "DEPENDENCY_FAILED"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    CONCURRENCY_LIMIT = "CONCURRENCY_LIMIT"

    # Media handling
    MEDIA_UNSUPPORTED = "MEDIA_UNSUPPORTED"
    MEDIA_CORRUPT = "MEDIA_CORRUPT"
    MEDIA_TOO_LARGE = "MEDIA_TOO_LARGE"
    MEDIA_TYPE_REJECTED = "MEDIA_TYPE_REJECTED"
    VIDEO_NOT_SUPPORTED = "VIDEO_NOT_SUPPORTED"
    PATH_OUTSIDE_WORKSPACE = "PATH_OUTSIDE_WORKSPACE"
    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    ARTIFACT_DIGEST_MISMATCH = "ARTIFACT_DIGEST_MISMATCH"

    # Detection / segmentation
    NO_TARGET_FOUND = "NO_TARGET_FOUND"
    MASK_EMPTY = "MASK_EMPTY"
    MASK_INVALID = "MASK_INVALID"
    MASK_SHAPE_MISMATCH = "MASK_SHAPE_MISMATCH"
    ANNOTATION_INVALID = "ANNOTATION_INVALID"
    ANNOTATION_TOO_SMALL = "ANNOTATION_TOO_SMALL"

    # Backends
    BACKEND_REQUEST_FAILED = "BACKEND_REQUEST_FAILED"
    BACKEND_RESPONSE_INVALID = "BACKEND_RESPONSE_INVALID"
    BACKEND_RATE_LIMITED = "BACKEND_RATE_LIMITED"
    BACKEND_AUTH_FAILED = "BACKEND_AUTH_FAILED"
    BACKEND_RESPONSE_TOO_LARGE = "BACKEND_RESPONSE_TOO_LARGE"

    # Dataset integrity
    SPLIT_LEAKAGE = "SPLIT_LEAKAGE"
    SPLIT_NOT_AUGMENTABLE = "SPLIT_NOT_AUGMENTABLE"
    DUPLICATE_CANDIDATE = "DUPLICATE_CANDIDATE"
    EXPORT_INVALID = "EXPORT_INVALID"

    # Storage / state
    STATE_STORE_FAILED = "STATE_STORE_FAILED"
    WORKSPACE_INVALID = "WORKSPACE_INVALID"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_STATE_INVALID = "JOB_STATE_INVALID"

    # Quality
    QUALITY_REJECT = "QUALITY_REJECT"
    QUALITY_EVALUATOR_FAILED = "QUALITY_EVALUATOR_FAILED"


class VidlinerError(Exception):
    """Base class for every VidLiner failure.

    Args:
        message: human-readable description, safe to show a user.
        code: machine-readable :class:`ErrorCode`.
        detail: optional structured context. Must never contain credential values.
    """

    failure_class: FailureClass = FailureClass.OPERATOR
    default_code: ErrorCode = ErrorCode.OPERATOR_CONFIG_INVALID

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code: ErrorCode = code or self.default_code
        self.detail: dict[str, Any] = dict(detail or {})

    @property
    def failure_class_name(self) -> str:
        """Name of the failure class, for logs and dashboards."""
        return self.failure_class.value

    def to_dict(self) -> dict[str, Any]:
        """Serialize the failure for a manifest or a log event."""
        return {
            "failure_class": self.failure_class.value,
            "code": self.code.value,
            "message": self.message,
            "detail": self.detail,
        }

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"


class OperatorFailure(VidlinerError):
    """An operator could not complete its own work."""

    failure_class = FailureClass.OPERATOR
    default_code = ErrorCode.OPERATOR_CONFIG_INVALID


class BackendFailure(VidlinerError):
    """A capability backend failed, was unreachable, or returned something unusable.

    Args:
        safe_to_retry: the backend's declaration of whether retrying the same request is
            idempotent. Generative requests default to ``False`` because a retry may produce a
            different image and consume another paid call.
    """

    failure_class = FailureClass.BACKEND
    default_code = ErrorCode.BACKEND_REQUEST_FAILED

    def __init__(
        self,
        message: str,
        *,
        backend_id: str | None = None,
        code: ErrorCode | None = None,
        detail: dict[str, Any] | None = None,
        safe_to_retry: bool = False,
    ) -> None:
        super().__init__(message, code=code, detail=detail)
        self.backend_id = backend_id
        self.safe_to_retry = safe_to_retry
        if backend_id is not None:
            self.detail.setdefault("backend_id", backend_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the failure, including the retry-safety declaration."""
        payload = super().to_dict()
        payload["safe_to_retry"] = self.safe_to_retry
        return payload


class ValidationFailure(VidlinerError):
    """Input did not satisfy a declared contract (recipe, annotation, geometry, dataset)."""

    failure_class = FailureClass.VALIDATION
    default_code = ErrorCode.RECIPE_INVALID


class QualityReject(VidlinerError):
    """A generated sample did not meet the acceptance policy.

    This is *not* a system error. It travels as an exception only inside the quality stage, where
    the gate operator converts it into a ``REJECTED`` decision. It must never propagate to the job
    state machine as a failure.
    """

    failure_class = FailureClass.QUALITY
    default_code = ErrorCode.QUALITY_REJECT

    def __init__(
        self,
        message: str,
        *,
        reason_codes: list[str] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=ErrorCode.QUALITY_REJECT, detail=detail)
        self.reason_codes = list(reason_codes or [])
        if self.reason_codes:
            self.detail.setdefault("reason_codes", self.reason_codes)


class InfrastructureFailure(VidlinerError):
    """Storage, state, filesystem, or environment failure."""

    failure_class = FailureClass.INFRASTRUCTURE
    default_code = ErrorCode.STATE_STORE_FAILED


class Cancellation(VidlinerError):
    """The job or node was cancelled by the user or by the engine shutting down."""

    failure_class = FailureClass.CANCELLATION
    default_code = ErrorCode.JOB_CANCELLED


def describe_failure(error: BaseException) -> dict[str, Any]:
    """Describe any exception in the manifest/log shape, without leaking non-VidLiner internals."""
    if isinstance(error, VidlinerError):
        return error.to_dict()
    return {
        "failure_class": FailureClass.OPERATOR.value,
        "code": ErrorCode.OPERATOR_CONFIG_INVALID.value,
        "message": f"{type(error).__name__}: {error}",
        "detail": {},
    }
