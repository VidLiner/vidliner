"""Quality measurement, gate, and policy tests.

These are the tests that decide whether the product's central promise holds: a generated sample is
rejected when it should be, and the reason is machine-readable.
"""

from __future__ import annotations

import numpy as np
import pytest

from vidliner.core.errors import QualityReject
from vidliner.domain.enums import Comparison, DecisionState, GateStatus, MetricName, Severity
from vidliner.domain.quality import (
    AcceptancePolicy,
    GateOutcome,
    GateSpec,
    MetricOutcome,
    QualityReport,
)
from vidliner.domain.reasons import REASON_CATALOG, ReasonCategory, ReasonCode, describe_reason
from vidliner.quality.fingerprint import (
    Fingerprint,
    hamming_distance,
    perceptual_hash,
    similarity_from_distance,
)
from vidliner.quality.gates import evaluate_gates, metric_for_reason
from vidliner.quality.metrics import (
    artifact_measurements,
    background_measurements,
    boundary_quality,
    connected_components,
    difference_map,
    geometry_measurements,
    image_similarity,
    mask_iou,
)


def _scene(width: int = 64, height: int = 48, *, seed: int = 0, flat: bool = False) -> np.ndarray:
    """A structured test scene.

    The default scene contains a gradient and a bright block so that structural and perceptual
    metrics have real structure to measure; ``flat=True`` produces the degenerate constant image that
    some tests need in order to check behaviour on a featureless input.
    """
    rng = np.random.default_rng(seed)
    if flat:
        base = np.full((height, width, 3), 0.5, dtype=np.float32)
    else:
        row = np.linspace(0.1, 0.9, height, dtype=np.float32)[:, None]
        column = np.linspace(0.9, 0.1, width, dtype=np.float32)[None, :]
        base = np.stack(
            [
                np.broadcast_to(row, (height, width)),
                np.broadcast_to(column, (height, width)),
                np.full((height, width), 0.5, dtype=np.float32),
            ],
            axis=2,
        )
        base[height // 4 : height // 2, width // 4 : width // 2] = 0.95
        base[height // 2 :, :] *= 0.6
    base = base.astype(np.float32)
    base += rng.normal(0.0, 0.01, size=base.shape).astype(np.float32)
    return np.clip(base, 0.0, 1.0)


def _mask(
    width: int = 64, height: int = 48, *, box: tuple[int, int, int, int] = (16, 12, 40, 32)
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return mask


# --------------------------------------------------------------------------- #
# Pixel measurements
# --------------------------------------------------------------------------- #


def test_identical_images_have_perfect_similarity() -> None:
    image = _scene()
    assert image_similarity(image, image) == pytest.approx(1.0)


def test_similarity_drops_when_the_image_changes() -> None:
    image = _scene()
    changed = image.copy()
    changed[:, :, :] = 0.95
    assert image_similarity(image, changed) < 0.9


def test_similarity_region_restricts_the_comparison() -> None:
    image = _scene()
    changed = image.copy()
    changed[0:5, 0:5] = 1.0
    region = np.ones(image.shape[:2], dtype=bool)
    region[0:10, 0:10] = False
    assert image_similarity(image, changed, region=region) > image_similarity(image, changed)


def test_difference_map_shape_mismatch_is_refused() -> None:
    with pytest.raises(ValueError):
        difference_map(np.zeros((4, 4)), np.zeros((5, 5)))


def test_background_measurements_are_perfect_for_an_untouched_image() -> None:
    image = _scene()
    measurements = background_measurements(image, image, _mask())
    assert measurements.changed_ratio == pytest.approx(0.0)
    assert measurements.score == pytest.approx(1.0)


def test_background_measurements_detect_a_recoloured_background() -> None:
    image = _scene()
    changed = image.copy()
    changed[:, :, :] = 0.9
    measurements = background_measurements(image, changed, _mask())
    assert measurements.changed_ratio > 0.5
    assert measurements.score < 0.7


def test_background_measurements_ignore_change_inside_the_mask() -> None:
    image = _scene()
    changed = image.copy()
    mask = _mask()
    changed[mask] = 0.0
    measurements = background_measurements(image, changed, mask)
    assert measurements.changed_ratio == pytest.approx(0.0)


def test_connected_components_counts_regions() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:6, 2:6] = True
    mask[10:14, 10:14] = True
    assert connected_components(mask) == [16, 16]
    assert connected_components(np.zeros((4, 4), dtype=bool)) == []


def test_boundary_quality_reports_empty_masks() -> None:
    measurements = boundary_quality(_scene(), np.zeros((48, 64), dtype=bool))
    assert measurements.is_empty
    assert measurements.score == 0.0


def test_boundary_quality_scores_a_clean_mask() -> None:
    image = _scene()
    mask = _mask()
    # Put structure on the boundary so the mask edge sits on real image content.
    image[mask] = 0.85
    measurements = boundary_quality(image, mask)
    assert 0.0 <= measurements.score <= 1.0
    assert "seam" in measurements.as_detail()


def test_artifact_measurements_flag_a_split_mask() -> None:
    mask = np.zeros((48, 64), dtype=bool)
    mask[5:15, 5:15] = True
    mask[30:40, 30:40] = True
    measurements = artifact_measurements(_scene(), mask)
    assert measurements.component_ratio < 0.7
    assert measurements.floating >= 0.0


def test_artifact_free_score_is_one_minus_the_artifact_score() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    measurements = artifact_measurements(_scene(32, 32), mask)
    assert measurements.freedom_score == pytest.approx(1.0 - measurements.artifact_score)


def test_geometry_measurements_are_identity_for_the_same_mask() -> None:
    mask = _mask()
    measurements = geometry_measurements(mask, mask)
    assert measurements.centroid_shift == pytest.approx(0.0)
    assert measurements.area_ratio == pytest.approx(1.0)
    assert measurements.aspect_delta == pytest.approx(0.0)


def test_geometry_measurements_detect_a_shift_and_resize() -> None:
    source = _mask(box=(10, 10, 30, 30))
    target = _mask(box=(20, 10, 30, 30))
    measurements = geometry_measurements(source, target)
    assert measurements.centroid_shift > 0
    assert measurements.area_ratio < 1.0


def test_geometry_measurements_handle_an_empty_mask() -> None:
    measurements = geometry_measurements(_mask(), np.zeros((48, 64), dtype=bool))
    assert measurements.area_ratio == 0.0
    assert measurements.centroid_shift == 1.0


def test_mask_iou() -> None:
    first = _mask(box=(0, 0, 10, 10))
    second = _mask(box=(5, 0, 15, 10))
    assert mask_iou(first, second) == pytest.approx(50 / 150)
    assert mask_iou(first, first) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #


def test_perceptual_hash_is_stable_and_short() -> None:
    image = _scene()
    first = perceptual_hash(image)
    assert first == perceptual_hash(image)
    assert len(first.hex) == 16


def test_perceptual_hash_survives_mild_noise() -> None:
    image = _scene()
    rng = np.random.default_rng(1)
    noisy = np.clip(image + rng.normal(0.0, 0.005, size=image.shape), 0.0, 1.0)
    assert perceptual_hash(image).distance(perceptual_hash(noisy)) <= 10


def test_perceptual_hash_separates_different_images() -> None:
    first = perceptual_hash(_scene(seed=1))
    second = perceptual_hash(_scene(seed=2))
    assert first.distance(second) > 0


def test_all_hash_algorithms_run() -> None:
    image = _scene()
    for algorithm in ("phash", "ahash", "dhash"):
        assert len(perceptual_hash(image, algorithm=algorithm).hex) == 16


def test_unknown_hash_algorithm_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown perceptual hash"):
        perceptual_hash(_scene(), algorithm="magic")


def test_hamming_distance_and_similarity() -> None:
    assert hamming_distance(0b1010, 0b1000) == 1
    assert similarity_from_distance(0) == pytest.approx(1.0)
    assert similarity_from_distance(64) == pytest.approx(0.0)


def test_fingerprint_parse_roundtrip() -> None:
    fingerprint = Fingerprint(0x0123456789ABCDEF)
    assert Fingerprint.parse(fingerprint.hex).value == fingerprint.value


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def _policy(**overrides: object) -> AcceptancePolicy:
    payload: dict[str, object] = {
        "minimum_overall": 0.5,
        "gates": (
            GateSpec(metric=MetricName.SEMANTIC_MATCH, threshold=0.9, severity=Severity.HARD),
            GateSpec(metric=MetricName.BACKGROUND_PRESERVATION, threshold=0.9, severity=Severity.HARD),
        ),
    }
    payload.update(overrides)
    return AcceptancePolicy.model_validate(payload)


def _outcomes(**values: float | None) -> list[MetricOutcome]:
    return [
        MetricOutcome(
            metric=MetricName(name),
            value=value,
            threshold=0.9,
            status=GateStatus.PASSED if value is not None else GateStatus.SKIPPED,
        )
        for name, value in values.items()
    ]


def test_gates_accept_a_strong_candidate() -> None:
    outcome = evaluate_gates(
        _policy(), _outcomes(semantic_match=0.95, background_preservation=0.99, geometry=1.0)
    )
    assert outcome.state is DecisionState.ACCEPTED
    assert outcome.reason_codes == ()
    assert outcome.overall_score > 0.5


def test_a_hard_gate_failure_rejects_even_with_a_high_overall_score() -> None:
    outcome = evaluate_gates(
        _policy(),
        _outcomes(semantic_match=0.99, background_preservation=0.10, geometry=1.0),
    )
    assert outcome.state is DecisionState.REJECTED
    assert "background_preservation" in outcome.failed_hard_gates
    assert ReasonCode.BACKGROUND_CHANGED.value in outcome.reason_codes


def test_missing_measurements_are_tolerated_by_default() -> None:
    outcome = evaluate_gates(_policy(), _outcomes(semantic_match=0.95, background_preservation=0.99))
    assert outcome.state is DecisionState.ACCEPTED
    assert outcome.missing_metrics == ()


def test_missing_measurements_reject_when_the_policy_forbids_them() -> None:
    policy = _policy(
        allow_missing_metrics=False, gates=(GateSpec(metric=MetricName.GEOMETRY, threshold=0.9),)
    )
    outcome = evaluate_gates(policy, [MetricOutcome(metric=MetricName.GEOMETRY, value=None)])
    assert outcome.state is DecisionState.REJECTED
    assert ReasonCode.EVALUATOR_FAILED.value in outcome.reason_codes


def test_overall_minimum_rejects_a_mediocre_candidate() -> None:
    policy = _policy(minimum_overall=0.99, gates=())
    outcome = evaluate_gates(policy, _outcomes(semantic_match=0.8))
    assert outcome.state is DecisionState.REJECTED
    assert "overall" in outcome.failed_hard_gates
    assert ReasonCode.BELOW_MINIMUM_OVERALL.value in outcome.reason_codes


def test_review_band_marks_a_borderline_candidate_for_review() -> None:
    policy = _policy(minimum_overall=0.5, review_band=0.05)
    outcome = evaluate_gates(
        policy,
        _outcomes(
            semantic_match=0.87, background_preservation=0.99, geometry=1.0, annotation_consistency=1.0
        ),
    )
    assert outcome.state is DecisionState.NEEDS_REVIEW
    assert ReasonCode.REVIEW_BORDERLINE.value in outcome.reason_codes


def test_warn_gate_does_not_reject() -> None:
    policy = _policy(
        minimum_overall=0.1,
        gates=(GateSpec(metric=MetricName.MASK_BOUNDARY, threshold=0.9, severity=Severity.WARN),),
    )
    outcome = evaluate_gates(policy, _outcomes(mask_boundary=0.4))
    assert outcome.state is not DecisionState.REJECTED
    assert "mask_boundary" in outcome.warned_gates


def test_policy_hash_is_recorded_on_the_outcome() -> None:
    policy = _policy()
    outcome = evaluate_gates(policy, _outcomes(semantic_match=0.95))
    assert outcome.policy_hash == policy.policy_hash


def test_quality_report_helpers() -> None:
    report = QualityReport(
        candidate_id="c_1",
        sample_id="s_1",
        metrics=(MetricOutcome(metric=MetricName.GEOMETRY, value=0.75),),
        outcome=GateOutcome(
            state=DecisionState.REJECTED,
            overall_score=0.6,
            reason_codes=(ReasonCode.GEOMETRY_VIOLATION.value,),
        ),
    )
    assert report.overall_score == 0.6
    assert report.decision is DecisionState.REJECTED
    assert report.metric(MetricName.GEOMETRY) is not None
    assert report.metric(MetricName.MASK_BOUNDARY) is None
    assert report.summary() == {"geometry": 0.75}
    assert report.reason_code_set == {ReasonCode.GEOMETRY_VIOLATION}


def test_quality_reject_is_a_sample_outcome_not_a_system_failure() -> None:
    from vidliner.core.errors import FailureClass

    rejection = QualityReject("too blurry", reason_codes=[ReasonCode.ARTIFACT_DETECTED.value])
    assert rejection.failure_class is FailureClass.QUALITY
    payload = rejection.to_dict()
    assert payload["code"] == "QUALITY_REJECT"


def test_metric_for_reason_maps_metric_codes() -> None:
    assert metric_for_reason(ReasonCode.SEMANTIC_MISMATCH.value) is MetricName.SEMANTIC_MATCH
    assert metric_for_reason(ReasonCode.SPLIT_LEAKAGE.value) is None


# --------------------------------------------------------------------------- #
# Reason catalogue
# --------------------------------------------------------------------------- #


def test_every_reason_code_is_documented_and_owned() -> None:
    for code in ReasonCode:
        category, description = describe_reason(code)
        assert description
        assert category in set(ReasonCategory)


def test_unknown_reason_code_is_refused() -> None:
    with pytest.raises(KeyError):
        describe_reason("NOT_A_REAL_CODE")


def test_reason_categories_have_members() -> None:
    for code, (category, description) in REASON_CATALOG.items():
        assert isinstance(code, ReasonCode)
        assert description
        assert category in set(ReasonCategory)


def test_comparison_and_severity_vocabularies() -> None:
    assert Comparison.AT_LEAST.value == "at_least"
    assert Severity.HARD.value == "hard"
