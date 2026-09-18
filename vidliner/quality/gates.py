"""Gate evaluation: turning measurements into an acceptance decision.

The rule set is small and explicit:

1. **Hard gates first.** Any hard gate that fails rejects the candidate, whatever the overall score
   is, and every failing hard gate is listed.
2. **Then the overall minimum.**
3. **Then the review band.** A candidate that misses a hard gate by no more than ``review_band``
   *and* meets the overall minimum is marked ``NEEDS_REVIEW`` instead of rejected, because a
   borderline sample is worth a human's ten seconds.
4. **Missing measurements.** When an evaluator produced nothing for a metric that has a gate, the
   gate is ``SKIPPED``. By default that is tolerated (the overall score is computed over what was
   measured); with ``allow_missing_metrics: false`` it rejects, which is the right setting for a
   production run where an evaluator going silent must not look like a pass.
"""

from __future__ import annotations

from collections.abc import Iterable

from vidliner.domain.enums import Comparison, DecisionState, GateStatus, MetricName, Severity
from vidliner.domain.quality import (
    AcceptancePolicy,
    GateOutcome,
    GateSpec,
    MetricOutcome,
    QualityReport,
)
from vidliner.domain.reasons import ReasonCode

__all__ = [
    "METRIC_REASON_CODES",
    "build_outcome",
    "evaluate_gates",
    "metric_for_reason",
    "metric_specs",
]

#: Default reason codes a metric contributes when it fails, when the measurement itself did not
#: supply something more specific.
METRIC_REASON_CODES: dict[MetricName, str] = {
    MetricName.SEMANTIC_MATCH: ReasonCode.SEMANTIC_MISMATCH.value,
    MetricName.TARGET_PRESENCE: ReasonCode.OBJECT_NOT_FOUND.value,
    MetricName.BACKGROUND_PRESERVATION: ReasonCode.BACKGROUND_CHANGED.value,
    MetricName.MASK_BOUNDARY: ReasonCode.MASK_BOUNDARY_ARTIFACT.value,
    MetricName.GEOMETRY: ReasonCode.GEOMETRY_VIOLATION.value,
    MetricName.ARTIFACT_FREE: ReasonCode.ARTIFACT_DETECTED.value,
    MetricName.ANNOTATION_CONSISTENCY: ReasonCode.ANNOTATION_MISMATCH.value,
    MetricName.TEMPORAL_CONSISTENCY: ReasonCode.TEMPORAL_FLICKER.value,
}

_METRIC_REASON_LOOKUP = {code: metric for metric, code in METRIC_REASON_CODES.items()}


def metric_specs() -> dict[MetricName, Comparison]:
    """Direction of every metric (all ``at_least`` today; see ``metric_direction``)."""
    return dict.fromkeys(MetricName, Comparison.AT_LEAST)


def metric_for_reason(code: str) -> MetricName | None:
    """Which metric a reason code belongs to, when it is a metric-level code."""
    try:
        return _METRIC_REASON_LOOKUP[ReasonCode(code)]
    except (KeyError, ValueError):
        return None


def evaluate_gates(policy: AcceptancePolicy, outcomes: Iterable[MetricOutcome]) -> GateOutcome:
    """Evaluate a policy against measured outcomes and produce the decision evidence.

    Args:
        policy: the acceptance policy.
        outcomes: one outcome per metric that was evaluated; metrics with no measurement are
            expected to be present with ``value=None`` and ``status=SKIPPED``.

    Returns:
        A :class:`GateOutcome` carrying the decision, the overall score, the failing gates, and the
        machine-readable reason codes.
    """
    measurements = list(outcomes)
    by_metric = {outcome.metric: outcome for outcome in measurements}
    overall = policy.overall_from(measurements)

    failed_hard: list[str] = []
    warned: list[str] = []
    absent: list[MetricName] = []
    reason_codes: list[str] = []
    borderline: list[str] = []

    for gate in policy.gates:
        outcome = by_metric.get(gate.metric)
        value = outcome.value if outcome is not None else None
        if outcome is None or value is None:
            absent.append(gate.metric)
            continue
        if gate.passes(value):
            continue
        codes = _reason_codes_for(gate, outcome)
        if gate.severity is Severity.HARD:
            if _within_review_band(gate, value, policy.review_band) and overall >= policy.minimum_overall:
                borderline.append(gate.metric.value)
                reason_codes.append(ReasonCode.REVIEW_BORDERLINE.value)
                reason_codes.extend(codes)
            else:
                failed_hard.append(gate.metric.value)
                reason_codes.extend(codes)
        elif gate.severity is Severity.WARN:
            warned.append(gate.metric.value)
            reason_codes.extend(codes)

    if not failed_hard and overall < policy.minimum_overall:
        failed_hard.append("overall")
        reason_codes.append(ReasonCode.BELOW_MINIMUM_OVERALL.value)

    if absent and not policy.allow_missing_metrics:
        for metric in absent:
            failed_hard.append(f"{metric.value}:missing")
            reason_codes.append(ReasonCode.EVALUATOR_FAILED.value)

    if failed_hard:
        state = DecisionState.REJECTED
    elif borderline:
        state = DecisionState.NEEDS_REVIEW
    elif warned and overall < policy.minimum_overall + policy.review_band:
        state = DecisionState.NEEDS_REVIEW
        reason_codes.append(ReasonCode.REVIEW_BORDERLINE.value)
    else:
        state = DecisionState.ACCEPTED

    return GateOutcome(
        state=state,
        overall_score=overall,
        minimum_overall=policy.minimum_overall,
        failed_hard_gates=tuple(dict.fromkeys(failed_hard)),
        warned_gates=tuple(dict.fromkeys(warned)),
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        policy_hash=policy.policy_hash,
        evaluated_metrics=tuple(
            outcome.metric.value for outcome in measurements if outcome.value is not None
        ),
        missing_metrics=tuple(metric.value for metric in absent),
    )


def _reason_codes_for(gate: GateSpec, outcome: MetricOutcome) -> list[str]:
    if outcome.reason_codes:
        return list(outcome.reason_codes)
    fallback = METRIC_REASON_CODES.get(gate.metric)
    return [fallback] if fallback else []


def _within_review_band(gate: GateSpec, value: float, band: float) -> bool:
    if band <= 0.0:
        return False
    return gate.margin(value) >= -band


def build_outcome(
    policy: AcceptancePolicy,
    report: QualityReport,
) -> QualityReport:
    """Attach a gate outcome to a quality report.

    Kept separate from :func:`evaluate_gates` so that ``vidliner qa`` can re-evaluate stored metric
    evidence under a new policy without recomputing any metric.
    """
    outcome = evaluate_gates(policy, report.metrics)
    return report.model_copy(update={"outcome": outcome})


def gate_status_for(
    metric: MetricName,
    policy: AcceptancePolicy,
    *,
    passed: bool,
    severity: Severity | None = None,
) -> GateStatus:
    """Status for a measured metric, given the policy and whether the gate passed."""
    gate = policy.gate_for(metric)
    effective = severity or (gate.severity if gate else Severity.INFO)
    if passed:
        return GateStatus.PASSED
    if effective is Severity.HARD:
        return GateStatus.FAILED
    if effective is Severity.WARN:
        return GateStatus.WARNED
    return GateStatus.PASSED
