"""Acceptance policy construction and gate application.

The policy is data, not code: ``vidliner qa <job>`` rebuilds it from the recipe and re-evaluates the
stored metric evidence, which is what makes a threshold change a cheap, auditable operation instead
of a full regeneration.
"""

from __future__ import annotations

from vidliner.domain.enums import Comparison, MetricName, Severity
from vidliner.domain.quality import (
    AcceptanceDecision,
    AcceptancePolicy,
    GateOutcome,
    GateSpec,
    QualityReport,
)
from vidliner.domain.recipe import Recipe

__all__ = ["decide_candidate", "policy_from_recipe", "reapply_policy"]


def policy_from_recipe(recipe: Recipe) -> AcceptancePolicy:
    """Build the acceptance policy a recipe declares.

    Inverted metrics are handled here, once: ``maximum_artifact_score`` is a *ceiling* on artifact
    evidence, so it becomes an ``AT_LEAST`` gate on ``artifact_free`` with threshold
    ``1 - maximum_artifact_score``. Every other gate keeps the recipe's direction.
    """
    quality = recipe.quality
    gates: list[GateSpec] = []
    for name, threshold in quality.hard_gates.items():
        metric = MetricName(name)
        gates.append(GateSpec(metric=metric, threshold=threshold, severity=Severity.HARD))
    for name, threshold in quality.warn_gates.items():
        metric = MetricName(name)
        gates.append(GateSpec(metric=metric, threshold=threshold, severity=Severity.WARN))
    if not any(gate.metric is MetricName.ARTIFACT_FREE for gate in gates):
        gates.append(
            GateSpec(
                metric=MetricName.ARTIFACT_FREE,
                threshold=round(1.0 - quality.maximum_artifact_score, 6),
                severity=Severity.WARN,
            )
        )
    if not any(gate.metric is MetricName.TARGET_PRESENCE for gate in gates):
        gates.append(
            GateSpec(
                metric=MetricName.TARGET_PRESENCE,
                threshold=quality.minimum_target_presence,
                severity=Severity.HARD,
            )
        )
    return AcceptancePolicy(
        minimum_overall=quality.minimum_overall,
        gates=tuple(gates),
        review_band=quality.review_band,
        weights=dict(quality.weights),
        background_change_ceiling=quality.background_change_ceiling,
        allow_missing_metrics=quality.allow_missing_metrics,
    )


def decide_candidate(
    report: QualityReport,
    policy: AcceptancePolicy,
    *,
    candidate_id: str,
    job_id: str,
) -> tuple[AcceptanceDecision, QualityReport]:
    """Apply a policy to a quality report.

    Returns:
        ``(decision, updated report)`` where the report now carries its :class:`GateOutcome`.
    """
    from vidliner.quality.gates import evaluate_gates

    outcome = evaluate_gates(policy, report.metrics)
    evaluated = report.model_copy(update={"outcome": outcome})
    decision = AcceptanceDecision(
        candidate_id=candidate_id,
        job_id=job_id,
        state=outcome.state,
        reason_codes=outcome.reason_codes,
        failed_hard_gates=outcome.failed_hard_gates,
        overall_score=outcome.overall_score,
        policy_hash=outcome.policy_hash,
    )
    return decision, evaluated


def reapply_policy(
    report: QualityReport, policy: AcceptancePolicy, *, candidate_id: str, job_id: str
) -> AcceptanceDecision:
    """Re-evaluate an existing report under a (possibly changed) policy, recomputing nothing."""
    decision, _ = decide_candidate(report, policy, candidate_id=candidate_id, job_id=job_id)
    return decision


def metrics_without_measurement(outcome: GateOutcome) -> tuple[str, ...]:
    """Metrics a policy required but no evaluator measured."""
    return outcome.missing_metrics


def comparison_for(metric: MetricName) -> Comparison:
    """Direction of a metric, exported so callers do not have to import the gate module."""
    from vidliner.domain.quality import metric_direction

    return metric_direction(metric)
