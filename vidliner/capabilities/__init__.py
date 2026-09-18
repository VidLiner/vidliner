"""Capability vocabulary and backend contracts.

An operator declares *what it needs* using the names in :mod:`vidliner.capabilities.names`; the
runtime profile decides *which backend* provides it. Nothing in this package names a vendor.
"""

from __future__ import annotations

from vidliner.capabilities.backend import (
    Backend,
    BackendHealth,
    BackendProbe,
    DetectionRequest,
    DetectionResult,
    GenerationOutcome,
    GenerationRequest,
    ImageCompositingBackend,
    ImageHarmonizationBackend,
    InstanceSegmentationBackend,
    MaskRefinementBackend,
    ObjectDetectorBackend,
    ObjectReplacementBackend,
    PipelineContext,
    QualityEvaluatorBackend,
    ReplacementPlannerBackend,
    SceneAnalysisBackend,
    SegmentationRequest,
    SegmentationResult,
    TrackerBackend,
    VisionEvaluationBackend,
)
from vidliner.capabilities.names import (
    KNOWN_CAPABILITIES,
    capability_for_refinement,
    describe_capability,
    is_known_capability,
)

__all__ = [
    "KNOWN_CAPABILITIES",
    "Backend",
    "BackendHealth",
    "BackendProbe",
    "DetectionRequest",
    "DetectionResult",
    "GenerationOutcome",
    "GenerationRequest",
    "ImageCompositingBackend",
    "ImageHarmonizationBackend",
    "InstanceSegmentationBackend",
    "MaskRefinementBackend",
    "ObjectDetectorBackend",
    "ObjectReplacementBackend",
    "PipelineContext",
    "QualityEvaluatorBackend",
    "ReplacementPlannerBackend",
    "SceneAnalysisBackend",
    "SegmentationRequest",
    "SegmentationResult",
    "TrackerBackend",
    "VisionEvaluationBackend",
    "capability_for_refinement",
    "describe_capability",
    "is_known_capability",
]
