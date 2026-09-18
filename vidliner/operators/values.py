"""Serialization of values that travel along graph edges.

A port carries one of a small, closed set of runtime values: a domain model, an artifact reference,
a list of objects, or a plain mapping. This module turns each of those into a tagged JSON form and
back, which is what lets the engine persist a node's output for resume and for the node cache and
then keep executing with the *typed* value rather than a dictionary.

Unknown tags are refused rather than guessed: silently coercing an unexpected payload back into the
wrong class is exactly how a pipeline starts producing plausible-looking wrong data.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from vidliner.core.errors import ErrorCode, InfrastructureFailure
from vidliner.core.results import ArtifactRef, PortValue
from vidliner.domain.annotations import AnnotationBundle
from vidliner.domain.instances import ObjectInstance, ObjectItem, ObjectSelection
from vidliner.domain.masks import MaskRef
from vidliner.domain.media import MediaAsset
from vidliner.domain.quality import AcceptanceDecision, GateOutcome, QualityReport
from vidliner.domain.replacement import ReplacementCandidate, ReplacementPlan
from vidliner.domain.scene import SceneContext
from vidliner.domain.shapes import BoundingBox

__all__ = ["TAG_MAP", "decode_value", "encode_value"]

#: Runtime classes that may cross a graph edge, keyed by the tag stored in the JSON form.
TAG_MAP: dict[str, type[BaseModel]] = {
    "annotation-bundle": AnnotationBundle,
    "artifact-ref": ArtifactRef,
    "bbox": BoundingBox,
    "candidate": ReplacementCandidate,
    "decision": AcceptanceDecision,
    "gate-outcome": GateOutcome,
    "object-instance": ObjectInstance,
    "object-item": ObjectItem,
    "object-selection": ObjectSelection,
    "mask-ref": MaskRef,
    "media-asset": MediaAsset,
    "plan": ReplacementPlan,
    "port-value": PortValue,
    "quality-report": QualityReport,
    "scene-context": SceneContext,
}

_OBJECT_LIST_TAGS: dict[str, type[BaseModel]] = {
    "instances": ObjectInstance,
    "object-items": ObjectItem,
    "plan-list": ReplacementPlan,
    "candidate-list": ReplacementCandidate,
    "annotation-list": AnnotationBundle,
}


def encode_value(value: Any) -> dict[str, Any]:
    """Encode a port value into a tagged, JSON-serializable mapping.

    Args:
        value: a domain model, artifact reference, list of domain models, or plain mapping.

    Raises:
        InfrastructureFailure: when the value is not part of the closed port vocabulary, because
            silently persisting an unknown payload would make resume unreliable.
    """
    if value is None:
        return {"tag": "none"}
    if isinstance(value, BaseModel):
        tag = _tag_for(type(value))
        return {"tag": tag, "payload": value.model_dump(mode="json")}
    if isinstance(value, (list, tuple)):
        items = list(value)
        if not items:
            return {"tag": "list", "payload": []}
        if all(isinstance(item, BaseModel) for item in items):
            tag = _list_tag_for(type(items[0]))
            return {"tag": tag, "payload": [item.model_dump(mode="json") for item in items]}
        return {"tag": "list", "payload": [_encode_plain(item) for item in items]}
    if isinstance(value, dict):
        return {"tag": "mapping", "payload": {str(key): _encode_plain(item) for key, item in value.items()}}
    if isinstance(value, (str, int, float, bool)):
        return {"tag": "scalar", "payload": value}
    raise InfrastructureFailure(
        f"port value of type {type(value).__name__} is not part of the serializable port vocabulary",
        code=ErrorCode.PORT_TYPE_MISMATCH,
        detail={"type": type(value).__name__},
    )


def decode_value(encoded: dict[str, Any]) -> Any:
    """Decode a tagged mapping back into its runtime value.

    Raises:
        InfrastructureFailure: on an unknown tag or a payload that does not validate. An unknown tag
            is never silently tolerated: a decoded-away value would look like a missing upstream port
            and would quietly skip a whole branch.
    """
    tag = encoded.get("tag")
    payload = encoded.get("payload")
    if tag == "none":
        return None
    if tag == "scalar":
        return payload
    if tag == "list":
        return list(payload or [])
    if tag == "mapping":
        return dict(payload or {})
    if tag in TAG_MAP:
        return TAG_MAP[tag].model_validate(payload)
    if tag in _OBJECT_LIST_TAGS:
        model = _OBJECT_LIST_TAGS[tag]
        return tuple(model.model_validate(item) for item in payload or [])
    raise InfrastructureFailure(
        f"cannot decode persisted port value with tag {tag!r}",
        code=ErrorCode.STATE_STORE_FAILED,
        detail={"tag": tag},
    )


def _encode_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_encode_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode_plain(item) for key, item in value.items()}
    return value


def _tag_for(cls: type[BaseModel]) -> str:
    for tag, candidate in TAG_MAP.items():
        if candidate is cls:
            return tag
    raise InfrastructureFailure(
        f"type {cls.__name__} has no port serialization tag",
        code=ErrorCode.PORT_TYPE_MISMATCH,
        detail={"type": cls.__name__},
    )


def _list_tag_for(cls: type[BaseModel]) -> str:
    for tag, candidate in _OBJECT_LIST_TAGS.items():
        if candidate is cls:
            return tag
    raise InfrastructureFailure(
        f"type {cls.__name__} has no list port serialization tag",
        code=ErrorCode.PORT_TYPE_MISMATCH,
        detail={"type": cls.__name__},
    )
