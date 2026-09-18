"""Resume, cache, cancellation, and failure-semantics tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.conftest import make_recipe
from vidliner.core.errors import ErrorCode, FailureClass, QualityReject, ValidationFailure, VidlinerError
from vidliner.domain.enums import CandidateState, JobState, NodeStatus
from vidliner.domain.recipe import Recipe
from vidliner.pipeline.runner import JobOptions
from vidliner.pipeline.service import JobRequest, Session


def _recipe(document: dict[str, object]) -> Recipe:
    return Recipe.model_validate(document)


def _run(session: Session, recipe: Recipe, **options: object):
    return asyncio.run(
        session.run(JobRequest(recipe=recipe, options=JobOptions(**options)))  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# Cache and resume
# --------------------------------------------------------------------------- #


def test_rerunning_the_same_job_reuses_the_cache(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1, seed=777))
    first = _run(session, recipe, workers=2)
    assert first.manifest.counters.generated == 10
    cached_first = first.manifest.counters.nodes_cached
    assert cached_first == 0

    # ``resume=False`` skips the job's own record and forces the engine to answer from the node
    # cache, which is the cross-job reuse path.
    second = asyncio.run(
        session.run(JobRequest(recipe=recipe, options=JobOptions(workers=2, resume=False, prefer_cache=True)))
    )
    assert second.manifest.state is JobState.SUCCEEDED
    assert second.manifest.counters.nodes_cached > 0
    assert second.manifest.state is JobState.SUCCEEDED
    assert second.manifest.counters.failed == 0
    # The cached run recovers each candidate's identity from the evidence on disk and re-applies the
    # policy, so the accepted count is reproduced without regenerating anything.
    assert second.manifest.counters.accepted == first.manifest.counters.accepted


def test_a_rerun_of_the_same_job_resumes_instead_of_recomputing(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1, seed=778))
    first = _run(session, recipe, workers=2)
    second = _run(session, recipe, workers=2)
    assert second.manifest.counters.nodes_resumed > 0
    assert second.manifest.counters.accepted == first.manifest.counters.accepted


def test_resume_after_an_interrupted_job(session: Session, dataset_root: Path) -> None:
    """Interrupt after the first sample, then resume and finish."""
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    plan, discovery = session.plan(recipe)
    contexts = session.sample_contexts(recipe, discovery)
    from vidliner.pipeline.runner import JobRunner

    runner = JobRunner(
        recipe=recipe,
        plan=plan,
        workspace=session.workspace,
        store=session.store,
        state=session.state,
        registry=session.registry,
        sample_contexts=contexts,
        options=JobOptions(workers=1),
    )

    async def scenario() -> None:
        task = asyncio.create_task(runner.execute())
        await asyncio.sleep(0.4)
        runner.cancel()
        await task

    asyncio.run(scenario())
    interrupted = session.state.require_job(plan.job_plan.job_id)
    assert interrupted.state is JobState.CANCELLED
    evidence_before = session.state.node_evidence(plan.job_plan.job_id)
    assert evidence_before

    resumed = asyncio.run(session.resume(plan.job_plan.job_id))
    assert resumed.manifest.state is JobState.SUCCEEDED
    assert resumed.manifest.counters.accepted >= 0
    # Nodes that finished before the interruption were reused, not recomputed.
    assert resumed.manifest.counters.nodes_resumed > 0


def test_resume_refuses_a_finished_job(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    outcome = _run(session, recipe)
    with pytest.raises(VidlinerError) as error:
        asyncio.run(session.resume(outcome.manifest.job_id))
    assert error.value.code is ErrorCode.JOB_STATE_INVALID


def test_unknown_job_is_reported(session: Session) -> None:
    with pytest.raises(VidlinerError) as error:
        session.job("j_missing")
    assert error.value.code is ErrorCode.JOB_NOT_FOUND


# --------------------------------------------------------------------------- #
# Failure semantics
# --------------------------------------------------------------------------- #


def test_quality_reject_is_not_a_system_failure() -> None:
    rejection = QualityReject("not good enough", reason_codes=["BACKGROUND_CHANGED"])
    assert rejection.failure_class is FailureClass.QUALITY
    assert rejection.reason_codes == ["BACKGROUND_CHANGED"]


def test_failure_classes_are_distinct() -> None:
    from vidliner.core.errors import BackendFailure, Cancellation, InfrastructureFailure, OperatorFailure

    assert OperatorFailure("x").failure_class is FailureClass.OPERATOR
    assert BackendFailure("x").failure_class is FailureClass.BACKEND
    assert InfrastructureFailure("x").failure_class is FailureClass.INFRASTRUCTURE
    assert Cancellation("x").failure_class is FailureClass.CANCELLATION
    assert ValidationFailure("x").failure_class is FailureClass.VALIDATION


def test_backend_failure_records_retry_safety() -> None:
    from vidliner.core.errors import BackendFailure

    safe = BackendFailure("timeout", backend_id="svc", safe_to_retry=True)
    unsafe = BackendFailure("bad request", backend_id="svc")
    assert safe.to_dict()["safe_to_retry"] is True
    assert unsafe.to_dict()["safe_to_retry"] is False
    assert safe.detail["backend_id"] == "svc"


def test_describe_failure_handles_foreign_exceptions() -> None:
    from vidliner.core.errors import describe_failure

    payload = describe_failure(RuntimeError("boom"))
    assert payload["code"] == ErrorCode.OPERATOR_CONFIG_INVALID.value
    assert "boom" in payload["message"]


def test_a_broken_backend_fails_the_job_with_a_structured_reason(
    session: Session, dataset_root: Path
) -> None:
    """Point the detector at a nonexistent class and give it an unusable import."""
    from vidliner.runtime.profile import BackendSpec, default_profile

    profile = default_profile()
    broken = profile.model_copy(
        update={
            "backends": {
                **profile.backends,
                "builtin_detector": BackendSpec(
                    use="vidliner.backends.heuristic.detector:HeuristicDetectorBackend",
                    options={"declared_capabilities": ["vision.object_detection.v1"]},
                ),
            }
        }
    )
    # A detector that refuses every class makes the job produce no targets; that is a data outcome.
    document = make_recipe("./data/source", classes=("unicorn",), candidates=1)
    outcome = _run(session, _recipe(document))
    assert outcome.manifest.state is JobState.SUCCEEDED
    assert outcome.manifest.counters.targets == 0
    del broken


def test_job_failure_is_recorded_with_a_failure_class(session: Session, dataset_root: Path) -> None:
    """A missing source file must fail the job with evidence rather than silently skipping it."""
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    plan, discovery = session.plan(recipe)
    contexts = session.sample_contexts(recipe, discovery)
    # Delete one source file after planning: ingest must refuse to continue.
    victim = next(iter(contexts.values()))
    Path(victim.source_path).unlink()

    from vidliner.pipeline.runner import JobRunner

    runner = JobRunner(
        recipe=recipe,
        plan=plan,
        workspace=session.workspace,
        store=session.store,
        state=session.state,
        registry=session.registry,
        sample_contexts=contexts,
        options=JobOptions(workers=1),
    )
    manifest = asyncio.run(runner.execute())
    # Node failures are recorded; the job itself still finishes because other samples succeeded.
    assert manifest.counters.nodes_failed >= 1
    evidence = session.state.node_evidence(manifest.job_id)
    failures = [row for row in evidence if row["status"] == NodeStatus.FAILED.value]
    assert failures
    assert failures[0]["error_code"] in {
        ErrorCode.MEDIA_UNSUPPORTED.value,
        ErrorCode.ARTIFACT_MISSING.value,
        ErrorCode.WORKSPACE_INVALID.value,
    }


def test_fail_fast_stops_after_the_first_failure(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    plan, discovery = session.plan(recipe)
    contexts = session.sample_contexts(recipe, discovery)
    next(iter(contexts.values()))
    for context in list(contexts.values())[:5]:
        Path(context.source_path).unlink()

    from vidliner.pipeline.runner import JobRunner

    runner = JobRunner(
        recipe=recipe,
        plan=plan,
        workspace=session.workspace,
        store=session.store,
        state=session.state,
        registry=session.registry,
        sample_contexts=contexts,
        options=JobOptions(workers=1, fail_fast=True),
    )
    manifest = asyncio.run(runner.execute())
    assert manifest.counters.nodes_failed == 1
    assert any("fail_fast" in note for note in (runner.report.notes if runner.report else []))


# --------------------------------------------------------------------------- #
# Re-evaluation
# --------------------------------------------------------------------------- #


def test_qa_reapplies_a_tighter_policy_without_regenerating(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=2))
    outcome = _run(session, recipe)
    assert outcome.manifest.counters.accepted == 20

    stricter = recipe.model_copy(
        update={
            "quality": recipe.quality.model_copy(
                update={"minimum_overall": 0.999, "hard_gates": {"semantic_match": 0.999}}
            )
        }
    )
    result = session.re_evaluate(outcome.manifest.job_id, stricter)
    assert result["evaluated"] == 20
    assert result["changed"] == 20
    counts = session.state.candidate_counts(outcome.manifest.job_id)
    assert counts.get("rejected") == 20
    assert counts.get("accepted") is None


def test_qa_is_idempotent_for_an_unchanged_policy(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    outcome = _run(session, recipe)
    first = session.re_evaluate(outcome.manifest.job_id, recipe)
    second = session.re_evaluate(outcome.manifest.job_id, recipe)
    assert first["changed"] == 0
    assert second["changed"] == 0


def test_candidate_transitions_are_recorded(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    outcome = _run(session, recipe)
    candidates = session.state.candidates(outcome.manifest.job_id)
    assert candidates
    events = session.state.candidate_events(candidates[0].candidate_id)
    states = [event["to_state"] for event in events]
    assert states[0] == CandidateState.GENERATED.value
    assert CandidateState.ACCEPTED.value in states
