"""Quality evidence, acceptance policy, and decisions.

The design rule this module encodes: **a score is not a decision**. A decision comes from gates,
each gate names the metric it constrains, the threshold, and the severity of failing it. Hard gates
reject regardless of how good the overall score looks, because one corrupt label is worse than one
lost sample.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidliner.core.canonical import digest_json
from vidliner.core.results import utc_now
from vidliner.domain.enums import Comparison, DecisionState, GateStatus, MetricName, Severity
from vidliner.domain.reasons import ReasonCode

__all__ = [
    "AcceptanceDecision",
    "AcceptancePolicy",
    "GateOutcome",
    "GateSpec",
    "MetricOutcome",
    "MetricSpec",
    "QualityReport",
    "default_metric_weights",
    "metric_direction",
]


def metric_direction(metric: MetricName) -> Comparison:
    """Whether a higher or lower value of ``metric`` is better.

    Every metric VidLiner defines is expressed so that **higher is better**, including
    ``artifact_free``, which is the *freedom from artifacts* (so ``1 - artifact_score``). Keeping a
    single direction for the whole vocabulary removes an entire class of sign errors from gate
    arithmetic; if a future metric genuinely wants the opposite direction it must declare it on its
    :class:`MetricSpec` and the gate may then override the comparison explicitly.
    """
    del metric  # the direction is uniform today; the parameter documents the extension point
    return Comparison.AT_LEAST


def default_metric_weights() -> dict[str, float]:
    """Weights used to combine metrics into the overall score.

    Weights are renormalised over the metrics actually evaluated, so a missing evaluator lowers
    confidence in the score rather than silently counting as zero.
    """
    return {
        MetricName.SEMANTIC_MATCH.value: 0.24,
        MetricName.TARGET_PRESENCE.value: 0.16,
        MetricName.BACKGROUND_PRESERVATION.value: 0.22,
        MetricName.MASK_BOUNDARY.value: 0.10,
        MetricName.GEOMETRY.value: 0.12,
        MetricName.ARTIFACT_FREE.value: 0.10,
        MetricName.ANNOTATION_CONSISTENCY.value: 0.06,
    }


class MetricSpec(BaseModel):
    """Declares what a metric means and in which direction it is good."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: MetricName
    comparison: Comparison
    description: str
    display_name: str = ""

    @property
    def label(self) -> str:
        """Human-readable metric name."""
        return self.display_name or self.name.value.replace("_", " ")


class GateSpec(BaseModel):
    """One acceptance threshold on one metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: MetricName
    threshold: float = Field(ge=0.0, le=1.0)
    severity: Severity = Severity.HARD
    comparison: Comparison | None = None

    @property
    def direction(self) -> Comparison:
        """Effective comparison for this gate."""
        return self.comparison or metric_direction(self.metric)

    def passes(self, value: float) -> bool:
        """Whether ``value`` satisfies this gate."""
        if self.direction is Comparison.AT_LEAST:
            return value >= self.threshold
        return value <= self.threshold

    def margin(self, value: float) -> float:
        """How far ``value`` is above the threshold, in the metric's own direction."""
        if self.direction is Comparison.AT_LEAST:
            return value - self.threshold
        return self.threshold - value


class MetricOutcome(BaseModel):
    """One metric's measured value and how it compared to its gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: MetricName
    value: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    comparison: Comparison = Comparison.AT_LEAST
    severity: Severity = Severity.HARD
    status: GateStatus = GateStatus.SKIPPED
    evaluator_id: str | None = None
    reason_codes: tuple[str, ...] = ()
    detail: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int | None = Field(default=None, ge=0)

    @property
    def measured(self) -> bool:
        """True when an actual measurement exists."""
        return self.value is not None

    @property
    def failed(self) -> bool:
        """True when the gate failed."""
        return self.status is GateStatus.FAILED

    @property
    def warned(self) -> bool:
        """True when the gate produced a warning."""
        return self.status is GateStatus.WARNED

    def normalised(self, *, higher_is_better: bool = True) -> float | None:
        """Value oriented so that larger is better, for the overall score."""
        if self.value is None:
            return None
        if self.comparison is Comparison.AT_LEAST:
            return self.value
        return 1.0 - self.value if higher_is_better else self.value


class GateOutcome(BaseModel):
    """The aggregate result of evaluating a policy against a set of metric outcomes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: DecisionState
    overall_score: float = Field(ge=0.0, le=1.0)
    minimum_overall: float = Field(default=0.0, ge=0.0, le=1.0)
    failed_hard_gates: tuple[str, ...] = ()
    warned_gates: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    policy_hash: str = ""
    evaluated_metrics: tuple[str, ...] = ()
    missing_metrics: tuple[str, ...] = ()

    @property
    def is_accepted(self) -> bool:
        """True when the gate outcome permits export."""
        return self.state is DecisionState.ACCEPTED

    @property
    def needs_review(self) -> bool:
        """True when a human should look at this candidate."""
        return self.state is DecisionState.NEEDS_REVIEW

    @property
    def reason_code_set(self) -> set[ReasonCode]:
        """Failed-gate reason codes parsed into the closed vocabulary."""
        return {ReasonCode(code) for code in self.reason_codes}


class AcceptanceDecision(BaseModel):
    """The persisted decision for one candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    job_id: str
    state: DecisionState
    reason_codes: tuple[str, ...] = ()
    failed_hard_gates: tuple[str, ...] = ()
    overall_score: float = Field(ge=0.0, le=1.0)
    policy_hash: str
    decided_at: datetime = Field(default_factory=utc_now)

    @property
    def is_accepted(self) -> bool:
        """True when the candidate may be exported."""
        return self.state is DecisionState.ACCEPTED


class AcceptancePolicy(BaseModel):
    """The full acceptance policy derived from a recipe's ``quality`` block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum_overall: float = Field(default=0.75, ge=0.0, le=1.0)
    gates: tuple[GateSpec, ...] = ()
    review_band: float = Field(default=0.0, ge=0.0, le=0.5)
    weights: dict[str, float] = Field(default_factory=dict)
    required_metrics: tuple[MetricName, ...] = ()
    background_change_ceiling: float = Field(default=0.06, ge=0.0, le=1.0)
    allow_missing_metrics: bool = True
    """When false, a metric that produced no measurement rejects the candidate."""

    @model_validator(mode="after")
    def _check_gates(self) -> Self:
        seen: set[MetricName] = set()
        for gate in self.gates:
            if gate.metric in seen:
                raise ValueError(f"acceptance policy declares metric {gate.metric.value} twice")
            seen.add(gate.metric)
        return self

    @property
    def policy_hash(self) -> str:
        """Stable digest of the policy, recorded with every decision."""
        return digest_json(self.model_dump(mode="json"))

    def hard_gates(self) -> tuple[GateSpec, ...]:
        """Gates whose failure rejects regardless of the overall score."""
        return tuple(gate for gate in self.gates if gate.severity is Severity.HARD)

    def warn_gates(self) -> tuple[GateSpec, ...]:
        """Gates whose failure only warns."""
        return tuple(gate for gate in self.gates if gate.severity is Severity.WARN)

    def gate_for(self, metric: MetricName) -> GateSpec | None:
        """Return the gate constraining ``metric``, if any."""
        for gate in self.gates:
            if gate.metric is metric:
                return gate
        return None

    def effective_weights(self, metrics: tuple[str, ...]) -> dict[str, float]:
        """Weights restricted to ``metrics`` and renormalised so they sum to 1."""
        base = {**default_metric_weights(), **self.weights}
        selected = {name: max(0.0, base.get(name, 0.0)) for name in metrics}
        total = sum(selected.values())
        if total <= 0:
            return {name: 1.0 / len(metrics) for name in metrics} if metrics else {}
        return {name: weight / total for name, weight in selected.items()}

    def overall_from(self, outcomes: list[MetricOutcome]) -> float:
        """Compute the overall score from measured metric outcomes."""
        measured = [outcome for outcome in outcomes if outcome.value is not None]
        if not measured:
            return 0.0
        weights = self.effective_weights(tuple(outcome.metric.value for outcome in measured))
        total = 0.0
        for outcome in measured:
            normalised = outcome.normalised()
            assert normalised is not None  # guaranteed by the filter above
            total += weights[outcome.metric.value] * normalised
        return min(max(total, 0.0), 1.0)


class QualityReport(BaseModel):
    """Complete quality evidence for one candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    sample_id: str
    metrics: tuple[MetricOutcome, ...] = ()
    outcome: GateOutcome | None = None
    evaluators: tuple[str, ...] = ()
    evaluated_at: datetime = Field(default_factory=utc_now)
    duration_ms: int | None = Field(default=None, ge=0)
    auxiliary: dict[str, Any] = Field(default_factory=dict)

    @property
    def overall_score(self) -> float | None:
        """Overall score, or ``None`` when no gate has been applied yet."""
        return self.outcome.overall_score if self.outcome else None

    @property
    def decision(self) -> DecisionState | None:
        """Decision state, or ``None`` before the gate stage runs."""
        return self.outcome.state if self.outcome else None

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """Reason codes contributed by the gates."""
        return self.outcome.reason_codes if self.outcome else ()

    @property
    def failed_hard_gates(self) -> tuple[str, ...]:
        """Names of hard gates that failed."""
        return self.outcome.failed_hard_gates if self.outcome else ()

    @property
    def reason_code_set(self) -> set[ReasonCode]:
        """Reason codes parsed into the closed vocabulary."""
        return {ReasonCode(code) for code in self.reason_codes}

    def metric(self, name: MetricName) -> MetricOutcome | None:
        """Return the outcome for one metric, if measured."""
        for outcome in self.metrics:
            if outcome.metric is name:
                return outcome
        return None

    def summary(self) -> dict[str, float]:
        """Compact metric→value mapping for reports and the dataset distribution table."""
        return {outcome.metric.value: outcome.value for outcome in self.metrics if outcome.value is not None}
