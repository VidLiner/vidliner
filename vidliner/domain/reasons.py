"""Machine-readable reason codes.

A decision without a reason code is not usable evidence: an operator cannot tell whether to loosen
a threshold, fix a backend, or fix the source data. Every code below is emitted by exactly one rule
in the quality or dataset-integrity layers, and codes are part of the public contract — they appear
in manifests, in the HTML review report, and in dataset reports, so they are never renamed without
a migration note.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = ["REASON_CATALOG", "ReasonCategory", "ReasonCode", "describe_reason", "reasons_for_category"]


class ReasonCategory(StrEnum):
    """Which part of the system owns a reason code."""

    QUALITY = "quality"
    ANNOTATION = "annotation"
    DATA = "data"
    EXECUTION = "execution"


class ReasonCode(StrEnum):
    """Closed vocabulary of acceptance and integrity reasons."""

    # --- semantic -------------------------------------------------------------
    SEMANTIC_MISMATCH = "SEMANTIC_MISMATCH"
    WRONG_CLASS = "WRONG_CLASS"
    OBJECT_NOT_FOUND = "OBJECT_NOT_FOUND"

    # --- background -----------------------------------------------------------
    BACKGROUND_CHANGED = "BACKGROUND_CHANGED"
    BACKGROUND_NOISE = "BACKGROUND_NOISE"

    # --- mask -----------------------------------------------------------------
    MASK_INVALID = "MASK_INVALID"
    MASK_EMPTY = "MASK_EMPTY"
    MASK_BOUNDARY_ARTIFACT = "MASK_BOUNDARY_ARTIFACT"

    # --- geometry -------------------------------------------------------------
    GEOMETRY_VIOLATION = "GEOMETRY_VIOLATION"
    GEOMETRY_CENTROID_SHIFT = "GEOMETRY_CENTROID_SHIFT"
    GEOMETRY_SCALE_CHANGE = "GEOMETRY_SCALE_CHANGE"
    GEOMETRY_ASPECT_CHANGE = "GEOMETRY_ASPECT_CHANGE"
    GEOMETRY_GROUND_CONTACT = "GEOMETRY_GROUND_CONTACT"

    # --- artifacts in the generated image -------------------------------------
    ARTIFACT_DETECTED = "ARTIFACT_DETECTED"
    ARTIFACT_BOUNDARY_BREAK = "ARTIFACT_BOUNDARY_BREAK"
    ARTIFACT_DUPLICATE_OBJECT = "ARTIFACT_DUPLICATE_OBJECT"
    ARTIFACT_FLOATING_OBJECT = "ARTIFACT_FLOATING_OBJECT"
    ARTIFACT_WATERMARK = "ARTIFACT_WATERMARK"
    ARTIFACT_TEXT = "ARTIFACT_TEXT"
    OBJECT_DISAPPEARED = "OBJECT_DISAPPEARED"
    DEFORMATION = "DEFORMATION"

    # --- annotation -----------------------------------------------------------
    ANNOTATION_OUT_OF_BOUNDS = "ANNOTATION_OUT_OF_BOUNDS"
    ANNOTATION_TOO_SMALL = "ANNOTATION_TOO_SMALL"
    ANNOTATION_MISMATCH = "ANNOTATION_MISMATCH"
    ANNOTATION_EMPTY = "ANNOTATION_EMPTY"

    # --- temporal (video) -----------------------------------------------------
    TEMPORAL_FLICKER = "TEMPORAL_FLICKER"
    TEMPORAL_IDENTITY_DRIFT = "TEMPORAL_IDENTITY_DRIFT"
    TEMPORAL_DISCONTINUITY = "TEMPORAL_DISCONTINUITY"
    TEMPORAL_TRACK_BREAK = "TEMPORAL_TRACK_BREAK"

    # --- decisions ------------------------------------------------------------
    BELOW_MINIMUM_OVERALL = "BELOW_MINIMUM_OVERALL"
    REVIEW_BORDERLINE = "REVIEW_BORDERLINE"

    # --- dataset integrity ----------------------------------------------------
    DEMO_BACKEND_NOT_ALLOWED = "DEMO_BACKEND_NOT_ALLOWED"
    DUPLICATE_NEAR = "DUPLICATE_NEAR"
    DUPLICATE_EXACT = "DUPLICATE_EXACT"
    SPLIT_LEAKAGE = "SPLIT_LEAKAGE"
    CREATIVE_MODE_NOT_ALLOWED = "CREATIVE_MODE_NOT_ALLOWED"
    SOURCE_SPLIT_NOT_ALLOWED = "SOURCE_SPLIT_NOT_ALLOWED"

    # --- execution ------------------------------------------------------------
    EVALUATOR_FAILED = "EVALUATOR_FAILED"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"


REASON_CATALOG: Final[dict[ReasonCode, tuple[ReasonCategory, str]]] = {
    ReasonCode.SEMANTIC_MISMATCH: (
        ReasonCategory.QUALITY,
        "generated object does not match the requested category",
    ),
    ReasonCode.WRONG_CLASS: (
        ReasonCategory.QUALITY,
        "re-detected object has a different class than requested",
    ),
    ReasonCode.OBJECT_NOT_FOUND: (ReasonCategory.QUALITY, "no object was found at the replaced location"),
    ReasonCode.BACKGROUND_CHANGED: (ReasonCategory.QUALITY, "non-target pixels changed beyond tolerance"),
    ReasonCode.BACKGROUND_NOISE: (
        ReasonCategory.QUALITY,
        "non-target pixels changed slightly but pervasively",
    ),
    ReasonCode.MASK_INVALID: (ReasonCategory.QUALITY, "mask is malformed or has an implausible area"),
    ReasonCode.MASK_EMPTY: (ReasonCategory.QUALITY, "mask selected no pixels"),
    ReasonCode.MASK_BOUNDARY_ARTIFACT: (
        ReasonCategory.QUALITY,
        "mask edge shows halo, clipping, or cut-through",
    ),
    ReasonCode.GEOMETRY_VIOLATION: (
        ReasonCategory.QUALITY,
        "geometry differs from the source beyond the budget",
    ),
    ReasonCode.GEOMETRY_CENTROID_SHIFT: (ReasonCategory.QUALITY, "object centroid moved beyond the budget"),
    ReasonCode.GEOMETRY_SCALE_CHANGE: (ReasonCategory.QUALITY, "object area changed beyond the budget"),
    ReasonCode.GEOMETRY_ASPECT_CHANGE: (
        ReasonCategory.QUALITY,
        "object aspect ratio changed beyond the budget",
    ),
    ReasonCode.GEOMETRY_GROUND_CONTACT: (
        ReasonCategory.QUALITY,
        "ground contact row moved beyond the budget",
    ),
    ReasonCode.ARTIFACT_DETECTED: (ReasonCategory.QUALITY, "the generated image contains a visible artifact"),
    ReasonCode.ARTIFACT_BOUNDARY_BREAK: (
        ReasonCategory.QUALITY,
        "object boundary is broken or discontinuous",
    ),
    ReasonCode.ARTIFACT_DUPLICATE_OBJECT: (
        ReasonCategory.QUALITY,
        "the replacement was duplicated in the frame",
    ),
    ReasonCode.ARTIFACT_FLOATING_OBJECT: (
        ReasonCategory.QUALITY,
        "the object does not touch its support surface",
    ),
    ReasonCode.ARTIFACT_WATERMARK: (ReasonCategory.QUALITY, "a watermark or logo was introduced"),
    ReasonCode.ARTIFACT_TEXT: (ReasonCategory.QUALITY, "unexpected text was introduced"),
    ReasonCode.OBJECT_DISAPPEARED: (ReasonCategory.QUALITY, "an unrelated object disappeared from the frame"),
    ReasonCode.DEFORMATION: (ReasonCategory.QUALITY, "the object is severely deformed"),
    ReasonCode.ANNOTATION_OUT_OF_BOUNDS: (ReasonCategory.ANNOTATION, "annotation extends outside the image"),
    ReasonCode.ANNOTATION_TOO_SMALL: (ReasonCategory.ANNOTATION, "annotation area is below the minimum"),
    ReasonCode.ANNOTATION_MISMATCH: (
        ReasonCategory.ANNOTATION,
        "annotation does not match the visible object",
    ),
    ReasonCode.ANNOTATION_EMPTY: (
        ReasonCategory.ANNOTATION,
        "no annotation was produced for the replaced object",
    ),
    ReasonCode.TEMPORAL_FLICKER: (ReasonCategory.QUALITY, "appearance flickers between frames"),
    ReasonCode.TEMPORAL_IDENTITY_DRIFT: (ReasonCategory.QUALITY, "object identity drifts across frames"),
    ReasonCode.TEMPORAL_DISCONTINUITY: (ReasonCategory.QUALITY, "object position is discontinuous"),
    ReasonCode.TEMPORAL_TRACK_BREAK: (ReasonCategory.QUALITY, "object track breaks or changes identity"),
    ReasonCode.BELOW_MINIMUM_OVERALL: (ReasonCategory.QUALITY, "overall score is below the recipe minimum"),
    ReasonCode.REVIEW_BORDERLINE: (
        ReasonCategory.QUALITY,
        "candidate is within the review band of a hard gate",
    ),
    ReasonCode.DEMO_BACKEND_NOT_ALLOWED: (
        ReasonCategory.DATA,
        "the dataset would be produced with demonstration backends",
    ),
    ReasonCode.DUPLICATE_NEAR: (ReasonCategory.DATA, "candidate is a near-duplicate of another sample"),
    ReasonCode.DUPLICATE_EXACT: (ReasonCategory.DATA, "candidate is an exact duplicate of another sample"),
    ReasonCode.SPLIT_LEAKAGE: (ReasonCategory.DATA, "augmentation descendant would cross a split boundary"),
    ReasonCode.CREATIVE_MODE_NOT_ALLOWED: (
        ReasonCategory.DATA,
        "creative replacements are not permitted in the dataset",
    ),
    ReasonCode.SOURCE_SPLIT_NOT_ALLOWED: (
        ReasonCategory.DATA,
        "the source sample's split is not augmentable",
    ),
    ReasonCode.EVALUATOR_FAILED: (
        ReasonCategory.EXECUTION,
        "a quality evaluator could not produce a measurement",
    ),
    ReasonCode.BACKEND_UNAVAILABLE: (ReasonCategory.EXECUTION, "the bound backend is not available"),
}


def describe_reason(code: ReasonCode | str) -> tuple[ReasonCategory, str]:
    """Return the category and human-readable description of a reason code.

    Unknown codes raise: an undocumented code in a manifest would make the decision unreviewable.
    """
    try:
        return REASON_CATALOG[ReasonCode(code)]
    except (KeyError, ValueError) as exc:
        raise KeyError(f"unknown reason code {code!r}") from exc


def reasons_for_category(category: ReasonCategory) -> tuple[ReasonCode, ...]:
    """Every reason code owned by ``category``, in declaration order."""
    return tuple(code for code, (owner, _) in REASON_CATALOG.items() if owner is category)
