"""Quality evaluation: measure a candidate and decide whether it may become training data.

This is the stage the whole product rests on. The operator produces one :class:`MetricOutcome` per
metric, each carrying its measured value, its gate, and the machine-readable reason codes it
contributed. Decisions come from the gate engine, never from the generator's own opinion of its
output — ``raw_metadata`` from a backend is recorded for audit and is never an input to a gate.

Two masks arrive here and they mean different things. Confusing them is how an open loop passes its
own checks:

* the **input mask** is the region handed to the generator — the only area where change was
  *authorised*. It defines what ``background_preservation`` is allowed to ignore;
* the **regenerated mask** is the object that was actually found in the generated image by
  ``verify.redetect``. It defines ``target_presence``, ``mask_boundary``, ``geometry``,
  ``annotation_consistency`` and the exported label.

Measured here:

* ``semantic_match`` — the bound evaluator's judgement that the new object matches the request;
* ``target_presence`` — a new object actually occupies the target region (requires ``found``);
* ``background_preservation`` — SSIM, changed-pixel ratio, and perceptual delta outside the *input*
  mask, with a hard ceiling on large-area background change;
* ``geometry`` — centroid, area ratio, aspect ratio, and ground contact of the **regenerated** object
  against the source object;
* ``mask_boundary`` — whether the regenerated mask edge sits on real image structure or shows a seam;
* ``artifact_free`` — fragmenting, floating, duplicated, or text-like evidence, from a local
  heuristic and, when one is bound, a vision-language evaluator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vidliner.capabilities.backend import SemanticAssessRequest, VLMRequest
from vidliner.capabilities.names import (
    CAP_QUALITY_ARTIFACT,
    CAP_QUALITY_SEMANTIC,
    CAP_QUALITY_VLM,
)
from vidliner.core.errors import ErrorCode, OperatorFailure, VidlinerError
from vidliner.core.results import ArtifactRef
from vidliner.domain.enums import (
    Comparison,
    Determinism,
    GateStatus,
    MetricName,
    PortType,
    Severity,
    StageName,
)
from vidliner.domain.quality import MetricOutcome, QualityReport
from vidliner.domain.reasons import ReasonCode
from vidliner.operators.backend_context import backend_context
from vidliner.operators.base import ExecutionContext, InputSpec, Operator, OperatorSpec, OutputSpec
from vidliner.operators.shared import backend_handle
from vidliner.quality.metrics import (
    annotation_consistency_score,
    artifact_measurements,
    background_measurements,
    boundary_quality,
    geometry_measurements,
)

__all__ = ["EvaluateCandidateOperator", "EvaluateConfig", "register_all"]


class EvaluateConfig(BaseModel):
    """Configuration of the quality-evaluation operator."""

    model_config = ConfigDict(extra="forbid")

    minimum_target_presence: float = Field(default=0.60, ge=0.0, le=1.0)
    background_change_ceiling: float = Field(default=0.06, ge=0.0, le=1.0)
    change_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    feather_px: int = Field(default=2, ge=0, le=32)
    min_annotation_area_px: int = Field(default=4, ge=0)
    artifact_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    use_vision_evaluator: bool = True
    hard_gates: dict[str, float] = Field(default_factory=dict)
    warn_gates: dict[str, float] = Field(default_factory=dict)
    geometry: dict[str, float] = Field(default_factory=dict)


class EvaluateCandidateOperator(Operator):
    """Measure a refined candidate across every quality metric."""

    @property
    def spec(self) -> OperatorSpec:
        """Declared contract of the evaluation operator."""
        return OperatorSpec(
            name="evaluate.candidate",
            version="1.0.0",
            stage=StageName.EVALUATE,
            summary="measure semantic match, presence, background, geometry, boundary, and artifacts",
            inputs={
                "source_image": InputSpec(PortType.IMAGE_REF, "the untouched source frame"),
                "image": InputSpec(PortType.IMAGE_REF, "the refined candidate"),
                "source_mask": InputSpec(PortType.MASK_REF, "the source object's mask"),
                "input_mask": InputSpec(
                    PortType.MASK_REF, "the mask handed to the generator: the authorised change region"
                ),
                "mask": InputSpec(PortType.MASK_REF, "the mask of the object found in the output"),
                "item": InputSpec(PortType.INSTANCES, "the object found in the output"),
                "found": InputSpec(PortType.FLAG, "whether the object was found at all", required=False),
                "candidate": InputSpec(PortType.CANDIDATE_REF, "the generated candidate payload"),
                "plan": InputSpec(PortType.PLAN, "the replacement plan"),
            },
            outputs={"report": OutputSpec(PortType.QUALITY_REPORT, "per-metric quality evidence")},
            config_model=EvaluateConfig,
            capabilities=(CAP_QUALITY_SEMANTIC, CAP_QUALITY_ARTIFACT),
            determinism=Determinism.DETERMINISTIC,
            timeout_s=300.0,
            max_parallelism=2,
        )

    async def run(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        """Evaluate the candidate and return its quality report."""
        config = EvaluateConfig.model_validate(context.config)
        source_image = _artifact(inputs.get("source_image"), port="source_image")
        candidate_image = _artifact(inputs.get("image"), port="image")
        source_mask_ref = _mask(inputs.get("source_mask"))
        input_mask_ref = _mask(inputs.get("input_mask")) or _mask(inputs.get("mask"))
        measured_mask_ref = _mask(inputs.get("mask"))
        if source_mask_ref is None or input_mask_ref is None or measured_mask_ref is None:
            raise OperatorFailure(
                "quality evaluation requires the source mask, the authorised change region, and the "
                "regenerated mask",
                code=ErrorCode.MASK_EMPTY,
                detail={
                    "source": source_mask_ref is not None,
                    "input": input_mask_ref is not None,
                    "regenerated": measured_mask_ref is not None,
                },
            )
        found = bool(inputs.get("found", True))
        item, candidate, plan = _context_objects(inputs)

        source = _array(context, source_image)
        produced = _array(context, candidate_image)
        if source.shape != produced.shape:
            raise OperatorFailure(
                "the refined candidate has a different size than the source frame",
                code=ErrorCode.MEDIA_CORRUPT,
                detail={"source": list(source.shape), "candidate": list(produced.shape)},
            )
        source_mask = context.io.load_mask(source_mask_ref.artifact)
        # The authorised change region, and the object that is actually in the output.
        input_mask = context.io.load_mask(input_mask_ref.artifact)
        found_mask = context.io.load_mask(measured_mask_ref.artifact)
        measured_mask = found_mask if found_mask.any() else input_mask

        outcomes: list[MetricOutcome] = []
        evaluators: list[str] = []

        semantic, evaluator_id = await self._semantic(
            context, candidate_image, measured_mask_ref, candidate, plan, config
        )
        outcomes.append(semantic)
        if evaluator_id:
            evaluators.append(evaluator_id)

        outcomes.append(_presence(found_mask, input_mask_ref, config, found=found))

        background = background_measurements(
            source,
            produced,
            input_mask,
            feather_px=config.feather_px,
            change_threshold=config.change_threshold,
        )
        outcomes.append(_background(background, config))

        boundaries = boundary_quality(produced, measured_mask)
        outcomes.append(_boundary(boundaries))

        geometry = geometry_measurements(source_mask, measured_mask)
        outcomes.append(_geometry(geometry, config))

        artifact_outcome, artifact_evaluator = await self._artifacts(
            context, candidate_image, measured_mask_ref, produced, measured_mask, config
        )
        outcomes.append(artifact_outcome)
        if artifact_evaluator:
            evaluators.append(artifact_evaluator)

        outcomes.append(_annotation(measured_mask, candidate, plan, config, item, found=found))

        report = QualityReport(
            candidate_id=str(candidate.get("candidate_key", "")),
            sample_id=str(candidate.get("sample_id", "")),
            metrics=tuple(outcomes),
            evaluators=tuple(dict.fromkeys(evaluators)),
            auxiliary={
                "source_digest": source_image.digest,
                "candidate_digest": candidate_image.digest,
                "source_mask_digest": source_mask_ref.artifact.digest,
                "input_mask_digest": input_mask_ref.artifact.digest,
                "regenerated_mask_digest": measured_mask_ref.artifact.digest,
                "found": found,
                "category": candidate.get("category"),
                "object_id": item.object_id if item is not None else None,
            },
        )
        context.publish(
            "evaluate.completed",
            metrics={
                outcome.metric.value: outcome.value for outcome in outcomes if outcome.value is not None
            },
            evaluators=list(report.evaluators),
        )
        return {"report": report}

    # -- individual metrics ------------------------------------------------ #

    async def _semantic(
        self,
        context: ExecutionContext,
        candidate_image: ArtifactRef,
        mask_ref: Any,
        candidate: dict[str, Any],
        plan: Any,
        config: EvaluateConfig,
    ) -> tuple[MetricOutcome, str]:
        gate = _gate(config, MetricName.SEMANTIC_MATCH)
        expected = str(candidate.get("category") or (plan.expected_category if plan else ""))
        description = str(candidate.get("description") or "")
        try:
            handle = await backend_handle(context, CAP_QUALITY_SEMANTIC)
            assessment = await handle.call(
                "assess_semantics",
                SemanticAssessRequest(
                    image=candidate_image,
                    mask=mask_ref.artifact,
                    bbox=_bbox_tuple(context, mask_ref),
                    expected_category=expected,
                    description=description,
                ),
                backend_context(context),
            )
        except VidlinerError as exc:
            return (
                MetricOutcome(
                    metric=MetricName.SEMANTIC_MATCH,
                    threshold=gate,
                    comparison=Comparison.AT_LEAST,
                    severity=Severity.HARD,
                    status=GateStatus.ERROR,
                    reason_codes=(ReasonCode.EVALUATOR_FAILED.value,),
                    detail={"error": exc.message, "code": exc.code.value},
                ),
                "",
            )
        passed = assessment.score >= gate
        return (
            MetricOutcome(
                metric=MetricName.SEMANTIC_MATCH,
                value=round(float(assessment.score), 6),
                threshold=gate,
                comparison=Comparison.AT_LEAST,
                severity=Severity.HARD,
                status=GateStatus.PASSED if passed else GateStatus.FAILED,
                evaluator_id=assessment.evaluator_id,
                reason_codes=()
                if passed
                else (assessment.reason_codes or (ReasonCode.SEMANTIC_MISMATCH.value,)),
                detail={
                    "expected_category": expected,
                    "matched_label": assessment.matched_label,
                    "top_labels": dict(list(assessment.labels.items())[:5]),
                    **{key: value for key, value in assessment.detail.items() if key != "appearance"},
                },
            ),
            assessment.evaluator_id,
        )

    async def _artifacts(
        self,
        context: ExecutionContext,
        candidate_image: ArtifactRef,
        mask_ref: Any,
        produced: np.ndarray,
        candidate_mask: np.ndarray,
        config: EvaluateConfig,
    ) -> tuple[MetricOutcome, str]:
        gate = config.artifact_threshold
        measurements = artifact_measurements(produced, candidate_mask)
        reason_codes: list[str] = []
        detail: dict[str, Any] = dict(measurements.as_detail())
        score = measurements.freedom_score
        evaluator_id = ""
        if config.use_vision_evaluator:
            try:
                handle = await backend_handle(context, CAP_QUALITY_VLM)
                response = await handle.call(
                    "evaluate",
                    VLMRequest(
                        image=candidate_image,
                        mask=mask_ref.artifact,
                        question="Does the generated region contain artifacts?",
                        choices=(
                            "duplicate_object",
                            "floating_object",
                            "boundary_break",
                            "unexpected_text",
                            "unexpected_disappearance",
                        ),
                        schema_name="artifact_report",
                    ),
                    backend_context(context),
                )
                evaluator_id = response.evaluator_id
                fields = response.fields
                detail["vision_evaluator"] = {
                    key: value for key, value in fields.items() if key != "reason_codes"
                }
                reported = fields.get("reason_codes")
                if isinstance(reported, list):
                    reason_codes.extend(str(item) for item in reported)
                value = fields.get("artifact_score")
                if isinstance(value, (int, float)):
                    # Blend the learned judgement with the local heuristic; a VLM is not infallible
                    # either, and the local measurement is the one that is reproducible offline.
                    score = float(np.clip(0.7 * (1.0 - float(value)) + 0.3 * score, 0.0, 1.0))
            except VidlinerError as exc:
                detail["vision_evaluator_error"] = {"code": exc.code.value, "message": exc.message}
        passed = score >= 1.0 - gate
        if not passed and not reason_codes:
            reason_codes.append(ReasonCode.ARTIFACT_DETECTED.value)
        return (
            MetricOutcome(
                metric=MetricName.ARTIFACT_FREE,
                value=round(float(score), 6),
                threshold=round(1.0 - gate, 6),
                comparison=Comparison.AT_LEAST,
                severity=Severity.WARN,
                status=GateStatus.PASSED if passed else GateStatus.WARNED,
                evaluator_id=evaluator_id or None,
                reason_codes=tuple(dict.fromkeys(reason_codes)) if not passed else (),
                detail=detail,
            ),
            evaluator_id,
        )


def _presence(
    found_mask: np.ndarray,
    input_mask_ref: Any,
    config: EvaluateConfig,
    *,
    found: bool,
) -> MetricOutcome:
    """Target presence: does an object the detector can see occupy the replaced region?

    The check is driven by the re-detection result, not by the mask the generator was handed. That is
    the whole point: with the input mask, a generator that returns the image untouched would score
    1.0 here, because the mask is the thing it was given.
    """
    present = found and bool(found_mask.any())
    coverage = float(found_mask.sum()) / max(1, found_mask.size)
    expected_ratio = input_mask_ref.area_px / max(1, input_mask_ref.shape.pixel_count)
    score = 0.0
    if present and expected_ratio > 0:
        # An object that is far smaller than the region it replaced is a partial replacement: the
        # rest of the old object, or bare background, is still there.
        score = float(np.clip(min(1.0, coverage / max(expected_ratio, 1e-6)), 0.0, 1.0))
    passed = present and score >= config.minimum_target_presence
    return MetricOutcome(
        metric=MetricName.TARGET_PRESENCE,
        value=round(score, 6),
        threshold=config.minimum_target_presence,
        comparison=Comparison.AT_LEAST,
        severity=Severity.HARD,
        status=GateStatus.PASSED if passed else GateStatus.FAILED,
        reason_codes=() if passed else (ReasonCode.OBJECT_NOT_FOUND.value,),
        detail={
            "found": present,
            "re_detected": found,
            "coverage": round(coverage, 6),
            "expected_coverage": round(expected_ratio, 6),
        },
    )


def _background(background: Any, config: EvaluateConfig) -> MetricOutcome:
    """Background preservation, with a hard ceiling on large-area change."""
    score = background.score
    ceiling_exceeded = background.changed_ratio > config.background_change_ceiling
    passed = not ceiling_exceeded
    reason_codes: tuple[str, ...] = ()
    if ceiling_exceeded:
        reason_codes = (ReasonCode.BACKGROUND_CHANGED.value,)
    elif score < 0.93:
        reason_codes = (ReasonCode.BACKGROUND_NOISE.value,)
    return MetricOutcome(
        metric=MetricName.BACKGROUND_PRESERVATION,
        value=round(float(score), 6),
        threshold=0.93,
        comparison=Comparison.AT_LEAST,
        severity=Severity.HARD,
        status=GateStatus.PASSED if passed else GateStatus.FAILED,
        reason_codes=reason_codes if not passed else (),
        detail={
            **background.as_detail(),
            "change_ceiling": config.background_change_ceiling,
            "ceiling_exceeded": ceiling_exceeded,
        },
    )


def _boundary(boundaries: Any) -> MetricOutcome:
    score = boundaries.score
    passed = score >= 0.6
    return MetricOutcome(
        metric=MetricName.MASK_BOUNDARY,
        value=round(float(score), 6),
        threshold=0.6,
        comparison=Comparison.AT_LEAST,
        severity=Severity.WARN,
        status=GateStatus.PASSED if passed else GateStatus.WARNED,
        reason_codes=() if passed else (ReasonCode.MASK_BOUNDARY_ARTIFACT.value,),
        detail=boundaries.as_detail(),
    )


def _geometry(geometry: Any, config: EvaluateConfig) -> MetricOutcome:
    """Geometry agreement, scored as the worst relative violation across four measurements."""
    limits = {
        "centroid_shift_max": config.geometry.get("centroid_shift_max", 0.08),
        "area_ratio_min": config.geometry.get("area_ratio_min", 0.70),
        "area_ratio_max": config.geometry.get("area_ratio_max", 1.45),
        "aspect_ratio_delta_max": config.geometry.get("aspect_ratio_delta_max", 0.30),
        "ground_contact_max_px": config.geometry.get("ground_contact_max_px", 0.05),
    }
    reason_codes: list[str] = []
    violations = {
        "centroid": geometry.centroid_shift / max(limits["centroid_shift_max"], 1e-6),
    }
    if geometry.centroid_shift > limits["centroid_shift_max"]:
        reason_codes.append(ReasonCode.GEOMETRY_CENTROID_SHIFT.value)
    area_low = limits["area_ratio_min"]
    area_high = limits["area_ratio_max"]
    if geometry.area_ratio < area_low:
        violations["area"] = area_low / max(geometry.area_ratio, 1e-6)
        reason_codes.append(ReasonCode.GEOMETRY_SCALE_CHANGE.value)
    elif geometry.area_ratio > area_high:
        violations["area"] = geometry.area_ratio / max(area_high, 1e-6)
        reason_codes.append(ReasonCode.GEOMETRY_SCALE_CHANGE.value)
    else:
        violations["area"] = 1.0
    violations["aspect"] = geometry.aspect_delta / max(limits["aspect_ratio_delta_max"], 1e-6)
    if geometry.aspect_delta > limits["aspect_ratio_delta_max"]:
        reason_codes.append(ReasonCode.GEOMETRY_ASPECT_CHANGE.value)
    violations["ground"] = geometry.contact_shift / max(limits["ground_contact_max_px"], 1e-6)
    if geometry.contact_shift > limits["ground_contact_max_px"]:
        reason_codes.append(ReasonCode.GEOMETRY_GROUND_CONTACT.value)
    worst = max(violations.values())
    score = float(np.clip(1.0 - max(0.0, worst - 1.0), 0.0, 1.0))
    if len(reason_codes) > 1:
        reason_codes.insert(0, ReasonCode.GEOMETRY_VIOLATION.value)
    passed = not reason_codes
    return MetricOutcome(
        metric=MetricName.GEOMETRY,
        value=round(score, 6),
        threshold=1.0,
        comparison=Comparison.AT_LEAST,
        severity=Severity.HARD,
        status=GateStatus.PASSED if passed else GateStatus.FAILED,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        detail={
            **geometry.as_detail(),
            "violations": {key: round(value, 4) for key, value in violations.items()},
            "limits": limits,
        },
    )


def _annotation(
    candidate_mask: np.ndarray,
    candidate: dict[str, Any],
    plan: Any,
    config: EvaluateConfig,
    item: Any,
    *,
    found: bool,
) -> MetricOutcome:
    """Annotation usability: non-empty, inside the frame, above the minimum area, and class-declared.

    This metric judges whether the label the pipeline is about to write is *structurally* usable.
    Whether the replacement actually belongs to the requested category is a judgement about
    appearance, and it is made by the semantic evaluator — a detector's fixed label set cannot
    express a sub-category such as ``sedan``, so inferring category agreement from it would reject
    correct replacements and accept wrong ones.
    """
    expected = str(candidate.get("category") or (plan.expected_category if plan else ""))
    geometry = _bbox_of(candidate_mask)
    matched = bool(expected) and found
    # The polygon that will actually be exported is the one segmentation produced for this object;
    # measuring the annotation the harness is about to write is the whole point of this metric.
    polygon = getattr(getattr(item, "instance", None), "polygon", None)
    polygon_points = polygon.point_count if polygon is not None else 0
    score, problems = annotation_consistency_score(
        shape=candidate_mask.shape,
        bbox=geometry,
        mask_area=int(candidate_mask.sum()),
        polygon_points=polygon_points,
        min_area_px=config.min_annotation_area_px,
        expected_category_matched=matched,
    )
    passed = score >= 0.95
    return MetricOutcome(
        metric=MetricName.ANNOTATION_CONSISTENCY,
        value=round(float(score), 6),
        threshold=0.95,
        comparison=Comparison.AT_LEAST,
        severity=Severity.HARD,
        status=GateStatus.PASSED if passed else GateStatus.FAILED,
        reason_codes=tuple(problems) if not passed else (),
        detail={
            "expected_category": expected,
            "mask_area_px": int(candidate_mask.sum()),
            "min_area_px": config.min_annotation_area_px,
            "bbox": list(geometry) if geometry else None,
        },
    )


def _bbox_of(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return (
        float(columns.min()),
        float(rows.min()),
        float(columns.max() + 1),
        float(rows.max() + 1),
    )


def _bbox_tuple(context: ExecutionContext, mask_ref: Any) -> tuple[float, float, float, float]:
    """The replaced object's bounding box, read from its regenerated mask."""
    array = context.io.load_mask(mask_ref.artifact)
    box = _bbox_of(array)
    if box is None:
        raise OperatorFailure(
            "the candidate mask is empty, so no bounding box can be derived",
            code=ErrorCode.MASK_EMPTY,
            detail={"object_id": None},
        )
    return box


def _gate(config: EvaluateConfig, metric: MetricName) -> float:
    if metric.value in config.hard_gates:
        return config.hard_gates[metric.value]
    if metric.value in config.warn_gates:
        return config.warn_gates[metric.value]
    return 0.90 if metric is MetricName.SEMANTIC_MATCH else 0.5


def _array(context: ExecutionContext, artifact: ArtifactRef) -> np.ndarray:
    from vidliner.backends.imageops import load_image_array

    return load_image_array(context.io, artifact)


def _artifact(value: Any, *, port: str) -> ArtifactRef:
    if isinstance(value, ArtifactRef):
        return value
    if isinstance(value, dict):
        return ArtifactRef.model_validate(value)
    raise OperatorFailure(
        f"evaluation needs an image artifact on port {port!r}, received {type(value).__name__}",
        code=ErrorCode.PORT_TYPE_MISMATCH,
        detail={"port": port},
    )


def _mask(value: Any) -> Any:
    from vidliner.domain.masks import MaskRef

    if value is None:
        return None
    if isinstance(value, MaskRef):
        return value
    if isinstance(value, dict):
        return MaskRef.model_validate(value)
    return None


def _context_objects(inputs: dict[str, Any]) -> tuple[Any, dict[str, Any], Any]:
    """Unpack the target item, candidate payload, and plan from the node's inputs."""
    from vidliner.domain.instances import ObjectItem
    from vidliner.domain.replacement import ReplacementPlan

    item_value = inputs.get("item")
    item = item_value if isinstance(item_value, ObjectItem) else ObjectItem.model_validate(item_value)
    candidate_value = inputs.get("candidate")
    if not isinstance(candidate_value, dict):
        raise OperatorFailure(
            "evaluation needs the candidate payload produced by generation",
            code=ErrorCode.PORT_TYPE_MISMATCH,
        )
    plan_value = inputs.get("plan")
    plan = (
        plan_value if isinstance(plan_value, ReplacementPlan) else ReplacementPlan.model_validate(plan_value)
    )
    return item, candidate_value, plan


def register_all(registry: Any) -> None:
    """Register this module's operators in ``registry``."""
    registry.register(EvaluateCandidateOperator)
