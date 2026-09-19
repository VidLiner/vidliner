"""The closed capability vocabulary.

A *capability* is what the pipeline needs; a *backend* is what satisfies it. Capability names are
versioned strings so that a new, incompatible contract can be introduced without silently
reinterpreting an old one: an operator that needs ``vision.object_detection.v1`` will not accept a
backend that only advertises ``vision.object_detection.v2``.

The constants here are the single source of truth. Every operator declares its needs by referencing
them, and the runtime resolver matches on the exact string.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "BUILTIN_CAPABILITIES",
    "CAPABILITY_GROUPS",
    "CAP_DEPTH_ESTIMATION",
    "CAP_IMAGE_COMPOSITING",
    "CAP_IMAGE_HARMONIZATION",
    "CAP_INSTANCE_SEGMENTATION",
    "CAP_MASK_REFINEMENT",
    "CAP_OBJECT_DETECTION",
    "CAP_OBJECT_REPLACEMENT",
    "CAP_OBJECT_TRACKING",
    "CAP_PLANNING",
    "CAP_QUALITY_ARTIFACT",
    "CAP_QUALITY_BACKGROUND",
    "CAP_QUALITY_EMBEDDING",
    "CAP_QUALITY_SEMANTIC",
    "CAP_QUALITY_VLM",
    "CAP_SCENE_ANALYSIS",
    "CAP_TRACKING",
    "CAP_VIDEO_REPLACEMENT",
    "KNOWN_CAPABILITIES",
    "PRODUCTION_CAPABILITIES",
    "capability_for_refinement",
    "describe_capability",
    "is_builtin_capability",
    "is_known_capability",
    "requires_production_backend",
]

# --- perception -------------------------------------------------------------
CAP_OBJECT_DETECTION: Final = "vision.object_detection.v1"
CAP_INSTANCE_SEGMENTATION: Final = "vision.instance_segmentation.v1"
CAP_OBJECT_TRACKING: Final = "vision.object_tracking.v1"
CAP_TRACKING: Final = CAP_OBJECT_TRACKING
CAP_DEPTH_ESTIMATION: Final = "vision.depth_estimation.v1"
CAP_SCENE_ANALYSIS: Final = "vision.scene_analysis.v1"

# --- planning ---------------------------------------------------------------
CAP_PLANNING: Final = "planning.replacement.v1"

# --- generation and refinement ---------------------------------------------
CAP_OBJECT_REPLACEMENT: Final = "generation.object_replacement.v1"
CAP_IMAGE_COMPOSITING: Final = "generation.image_compositing.v1"
CAP_IMAGE_HARMONIZATION: Final = "generation.image_harmonization.v1"
CAP_MASK_REFINEMENT: Final = "generation.mask_refinement.v1"
CAP_VIDEO_REPLACEMENT: Final = "generation.video_replacement.v1"

# --- quality ----------------------------------------------------------------
CAP_QUALITY_SEMANTIC: Final = "quality.semantic_match.v1"
CAP_QUALITY_BACKGROUND: Final = "quality.background_preservation.v1"
CAP_QUALITY_ARTIFACT: Final = "quality.artifact_detection.v1"
CAP_QUALITY_EMBEDDING: Final = "quality.embedding.v1"
CAP_QUALITY_VLM: Final = "quality.vision_evaluation.v1"

CAPABILITY_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "perception": (
        CAP_OBJECT_DETECTION,
        CAP_INSTANCE_SEGMENTATION,
        CAP_OBJECT_TRACKING,
        CAP_DEPTH_ESTIMATION,
        CAP_SCENE_ANALYSIS,
    ),
    "planning": (CAP_PLANNING,),
    "generation": (
        CAP_OBJECT_REPLACEMENT,
        CAP_IMAGE_COMPOSITING,
        CAP_IMAGE_HARMONIZATION,
        CAP_MASK_REFINEMENT,
        CAP_VIDEO_REPLACEMENT,
    ),
    "quality": (
        CAP_QUALITY_SEMANTIC,
        CAP_QUALITY_BACKGROUND,
        CAP_QUALITY_ARTIFACT,
        CAP_QUALITY_EMBEDDING,
        CAP_QUALITY_VLM,
    ),
}

KNOWN_CAPABILITIES: Final[tuple[str, ...]] = tuple(
    capability for group in CAPABILITY_GROUPS.values() for capability in group
)

#: Capabilities that need no bound backend, because the pipeline measures them itself in
#: :mod:`vidliner.quality.metrics`. They remain in the vocabulary because recipes reference them in
#: acceptance gates and the dataset report is keyed by them, but a runtime profile must not bind
#: them and ``vidliner backend check`` must not report them as uncovered.
BUILTIN_CAPABILITIES: Final[tuple[str, ...]] = (CAP_QUALITY_BACKGROUND,)

#: Capabilities that must be served by a production implementation before a dataset may be exported.
#:
#: These four are the ones whose output *becomes* the training data: what is detected is what is
#: replaced, what is segmented is what is labelled, what is generated is what is trained on, and what
#: is judged is what is allowed through. A demonstration stand-in for any of them can still run a
#: complete job — that is what makes the demo and the test suite possible — but the dataset it
#: produces is not training data, and saying so is the difference between an honest pipeline and a
#: plausible-looking one.
PRODUCTION_CAPABILITIES: Final[tuple[str, ...]] = (
    CAP_OBJECT_DETECTION,
    CAP_INSTANCE_SEGMENTATION,
    CAP_OBJECT_REPLACEMENT,
    CAP_QUALITY_SEMANTIC,
)


def is_builtin_capability(name: str) -> bool:
    """Whether the pipeline satisfies ``name`` without a bound backend."""
    return name in BUILTIN_CAPABILITIES


def requires_production_backend(name: str) -> bool:
    """Whether ``name`` must be served by a production backend before a dataset may be exported."""
    return name in PRODUCTION_CAPABILITIES


_DESCRIPTIONS: Final[dict[str, str]] = {
    CAP_OBJECT_DETECTION: "locate objects and report class, confidence, and bounding box",
    CAP_INSTANCE_SEGMENTATION: "produce a pixel-level mask for a prompted object",
    CAP_OBJECT_TRACKING: "associate one object across frames (video)",
    CAP_DEPTH_ESTIMATION: "estimate per-pixel relative depth (video/3D-aware planning)",
    CAP_SCENE_ANALYSIS: "measure lighting, orientation, scale, occlusion, and ground contact",
    CAP_PLANNING: "turn a recipe and a scene context into an executable replacement plan",
    CAP_OBJECT_REPLACEMENT: "generate a replacement object inside a masked region of an image",
    CAP_IMAGE_COMPOSITING: "blend a generated region into an image without changing the rest",
    CAP_IMAGE_HARMONIZATION: "match colour and illumination between a generated region and its surroundings",
    CAP_MASK_REFINEMENT: "clean and feather a mask edge",
    CAP_VIDEO_REPLACEMENT: "generate a temporally consistent replacement across frames",
    CAP_QUALITY_SEMANTIC: "judge whether a generated object matches the requested category",
    CAP_QUALITY_BACKGROUND: "measure whether non-target pixels were preserved",
    CAP_QUALITY_ARTIFACT: "detect visible artifacts in a generated image",
    CAP_QUALITY_EMBEDDING: "produce an embedding for similarity comparison",
    CAP_QUALITY_VLM: "answer a structured visual question about a generated image",
}

_REFINEMENT_CAPABILITIES: Final[dict[str, str]] = {
    "refine.mask_edges": CAP_MASK_REFINEMENT,
    "refine.alpha_blend": CAP_IMAGE_COMPOSITING,
    "refine.harmonize": CAP_IMAGE_HARMONIZATION,
}


def is_known_capability(name: str) -> bool:
    """Whether ``name`` is part of the published capability vocabulary."""
    return name in KNOWN_CAPABILITIES


def describe_capability(name: str) -> str:
    """One-line description of a capability.

    Raises:
        KeyError: when the capability is not part of the vocabulary.
    """
    if name not in _DESCRIPTIONS:
        raise KeyError(f"unknown capability {name!r}")
    return _DESCRIPTIONS[name]


def capability_for_refinement(operator: str) -> str:
    """Map a refinement operator name to the capability it requires.

    Raises:
        KeyError: when the operator is not a known refinement step.
    """
    try:
        return _REFINEMENT_CAPABILITIES[operator]
    except KeyError as exc:
        raise KeyError(f"unknown refinement operator {operator!r}") from exc
