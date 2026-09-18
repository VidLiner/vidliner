"""End-to-end proof that the verification loop is closed.

A job is run against a replacement backend that *removes* the target object. The output is a valid,
plausible image whose background is byte-identical outside the mask — so every check that compares
the generated image with the mask the generator was handed reports success.

The test asserts the opposite: no candidate is accepted, each is rejected with ``OBJECT_NOT_FOUND``,
and no annotation is exported for an object that is not in the image. This is the property the
``verify.redetect`` stage exists for, and it is asserted here rather than only in a unit test so that
the whole pipeline is covered end to end.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from tests.conftest import make_recipe
from vidliner.domain.enums import CandidateState, JobState
from vidliner.domain.recipe import Recipe
from vidliner.pipeline.runner import JobOptions
from vidliner.pipeline.service import JobRequest, Session

ERASING_BACKEND = "tests.fixtures.erasing_backend:ErasingReplacementBackend"


@pytest.fixture(name="erasing_session")
def erasing_session_fixture(session: Session, workspace: Path, dataset_root: Path) -> Session:
    """A session whose replacement capability erases the object instead of replacing it.

    The already-open :func:`session` fixture is reconfigured rather than a second one opened: two
    live connections to the same state database is a situation no caller ever has, and a test that
    creates one is testing something that does not happen in production.
    """
    from vidliner.runtime.profile import default_profile

    profile = default_profile()
    replacement = profile.backends["builtin_replacement"].model_copy(update={"use": ERASING_BACKEND})
    profile = profile.model_copy(
        update={"backends": {**profile.backends, "builtin_replacement": replacement}}
    )
    (workspace / "runtime.yaml").write_text(
        yaml.safe_dump(profile.model_dump(mode="json", exclude={"source_path"}), sort_keys=False),
        encoding="utf-8",
    )
    session.reload_profile()
    return session


def _run(session: Session, recipe: Recipe, **options: object):
    return asyncio.run(
        session.run(JobRequest(recipe=recipe, options=JobOptions(**options)))  # type: ignore[arg-type]
    )


def _unique_recipe(seed: int, **overrides: object) -> Recipe:
    """A recipe whose job id cannot collide with another test's.

    ``job_id`` is derived from the recipe hash, the seed, and the sample set, so tests that share a
    dataset and a seed share a job — and the second one would resume the first one's finished job
    instead of running its own. Each test therefore gets its own seed.
    """
    document = make_recipe(
        "./data/source",
        candidates=1,
        minimum_overall=0.05,
        hard_gates={},
        seed=seed,
        **overrides,  # type: ignore[arg-type]
    )
    return Recipe.model_validate(document)


def test_erased_object_is_never_accepted(erasing_session: Session) -> None:
    """The generator removed the object; every candidate must be rejected."""
    # Permissive gates on purpose: the rejection must come from the *presence* check, not from a
    # threshold the test set to make rejection likely.
    recipe = _unique_recipe(1_001)
    outcome = _run(erasing_session, recipe, workers=2)

    assert outcome.manifest.state is JobState.SUCCEEDED
    assert outcome.manifest.counters.generated == 10
    assert outcome.manifest.counters.accepted == 0, "a candidate with no object must never be accepted"
    assert outcome.manifest.counters.rejected == 10
    # Rejection is a sample outcome, not a system failure.
    assert outcome.manifest.counters.failed == 0


def test_erased_object_is_rejected_for_the_right_reason(erasing_session: Session) -> None:
    """The reason must be OBJECT_NOT_FOUND, not a vague low score."""
    recipe = _unique_recipe(1_002)
    outcome = _run(erasing_session, recipe, workers=2)

    candidates = erasing_session.state.candidates(outcome.manifest.job_id)
    assert candidates
    for candidate in candidates:
        assert candidate.state is CandidateState.REJECTED
        assert "OBJECT_NOT_FOUND" in candidate.reason_codes, candidate.reason_codes


def test_erased_object_produces_no_exported_annotation(erasing_session: Session) -> None:
    """Nothing is exported, and no label is written for an object that is not there."""
    recipe = _unique_recipe(1_003)
    outcome = _run(erasing_session, recipe, workers=2)

    assert outcome.export is not None
    assert outcome.export.accepted == 0
    assert list((outcome.export.path / "images").glob("*.png")) == []
    assert list((outcome.export.path / "provenance").glob("*.json")) == []
    document = json.loads((outcome.export.path / "annotations" / "instances_train.json").read_text())
    assert document["images"] == []
    assert document["annotations"] == []


def test_presence_metric_records_that_nothing_was_found(erasing_session: Session) -> None:
    """The evidence must say *why*: the re-detector found no object in the replaced region."""
    recipe = _unique_recipe(1_004)
    outcome = _run(erasing_session, recipe, workers=2)

    candidate = erasing_session.state.candidates(outcome.manifest.job_id)[0]
    metrics = {row["metric"]: row for row in erasing_session.state.metrics_for(candidate.candidate_id)}
    presence = metrics["target_presence"]
    assert presence["status"] == "failed"
    assert presence["value"] == 0.0
    detail = presence["detail"]
    if isinstance(detail, str):
        detail = json.loads(detail)
    assert detail["found"] is False
    assert detail["re_detected"] is False
    # The background was untouched — which is exactly why an open loop would have accepted this.
    assert metrics["background_preservation"]["value"] > 0.9


def test_verification_stage_is_recorded_in_the_run_evidence(erasing_session: Session) -> None:
    """The stage runs, and its outcome is auditable from the event log."""
    recipe = _unique_recipe(1_005)
    outcome = _run(erasing_session, recipe, workers=2)

    events = [
        json.loads(line)
        for line in (Path(outcome.manifest.run_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    not_found = [event for event in events if event["event"] == "verify.not_found"]
    assert not_found, "the re-detection stage must record that it found nothing"
    assert not_found[0]["wanted_class"] == "car"
    assert not_found[0]["expected_category"] in {"sedan", "suv"}

    evidence = erasing_session.state.node_evidence(outcome.manifest.job_id)
    verify_nodes = [row for row in evidence if row["operator"] == "verify.redetect"]
    assert verify_nodes
    assert {row["stage"] for row in verify_nodes} == {"verify"}
    assert all(row["status"] in {"succeeded", "skipped"} for row in verify_nodes)


def test_a_working_generator_still_passes_the_presence_check(session: Session, dataset_root: Path) -> None:
    """The loop must not reject everything: the built-in generator still produces accepted samples."""
    recipe = _unique_recipe(1_006)
    outcome = _run(session, recipe, workers=2)
    assert outcome.manifest.counters.accepted > 0, "the verification loop must not be a blanket rejection"
    assert outcome.manifest.counters.failed == 0
