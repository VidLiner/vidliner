"""Storage tests: the content-addressed artifact store and the SQLite state store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidliner.core.errors import ErrorCode, InfrastructureFailure, ValidationFailure
from vidliner.core.graph import OperationNode
from vidliner.core.results import ArtifactRef, JobCounters, NodeResult, PortValue, utc_now
from vidliner.domain.enums import (
    ArtifactKind,
    CandidateState,
    JobState,
    NodeStatus,
    PortType,
    SampleState,
    StageName,
)
from vidliner.domain.jobs import JobManifest
from vidliner.storage.state import EvidenceRecorder, InMemoryRunState, StateStore
from vidliner.storage.workspace import ArtifactStore, BackendIO, Workspace


@pytest.fixture(name="store")
def store_fixture(tmp_path: Path) -> ArtifactStore:
    """An artifact store over a temporary workspace."""
    return ArtifactStore(Workspace(tmp_path / "ws").initialize())


@pytest.fixture(name="state")
def state_fixture(tmp_path: Path) -> StateStore:
    """An initialised state store."""
    return StateStore(tmp_path / "state.db").initialize()


# --------------------------------------------------------------------------- #
# Artifact store
# --------------------------------------------------------------------------- #


def test_workspace_initialises_the_expected_layout(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "ws").initialize()
    assert workspace.is_initialized()
    assert workspace.artifacts_dir.is_dir()
    assert workspace.runs_dir.is_dir()
    assert workspace.cache_dir.is_dir()


def test_workspace_refuses_paths_outside_its_boundary(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "ws").initialize()
    with pytest.raises(ValidationFailure) as error:
        workspace.resolve_inside(tmp_path / "elsewhere" / "file.txt")
    assert error.value.code is ErrorCode.PATH_OUTSIDE_WORKSPACE


def test_workspace_resolves_relative_paths_inside(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "ws").initialize()
    resolved = workspace.resolve_inside("data/source")
    assert resolved == workspace.root / "data" / "source"
    assert workspace.relative(resolved) == "data/source"


def test_workspace_scratch_is_removed_on_exit(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "ws").initialize()
    with workspace.scratch() as scratch:
        marker = scratch / "file.txt"
        marker.write_text("x")
        assert marker.exists()
    assert not scratch.exists()


def test_artifact_write_is_content_addressed_and_idempotent(store: ArtifactStore) -> None:
    first = store.write(b"payload", kind=ArtifactKind.REPORT, media_type="application/json")
    second = store.write(b"payload", kind=ArtifactKind.REPORT, media_type="application/json")
    assert first.digest == second.digest
    assert store.exists(first.digest)
    files = list(store.workspace.artifacts_dir.rglob("*.json"))
    assert len(files) == 1


def test_artifact_read_verifies_the_digest(store: ArtifactStore) -> None:
    reference = store.write(b"payload", kind=ArtifactKind.REPORT, media_type="application/json")
    path = store.path_of(reference.digest)
    path.write_bytes(b"tampered")
    with pytest.raises(ValidationFailure) as error:
        store.read(reference.digest)
    assert error.value.code is ErrorCode.ARTIFACT_DIGEST_MISMATCH


def test_artifact_read_without_verification_returns_raw_bytes(store: ArtifactStore) -> None:
    reference = store.write(b"payload", kind=ArtifactKind.REPORT, media_type="application/json")
    assert store.read(reference.digest, verify=False) == b"payload"


def test_artifact_missing_raises(store: ArtifactStore) -> None:
    with pytest.raises(ValidationFailure) as error:
        store.read("a" * 64)
    assert error.value.code is ErrorCode.ARTIFACT_MISSING


def test_artifact_size_cap_is_enforced(tmp_path: Path) -> None:
    store = ArtifactStore(Workspace(tmp_path / "ws").initialize(), max_artifact_bytes=4)
    with pytest.raises(ValidationFailure) as error:
        store.write(b"too large", kind=ArtifactKind.REPORT, media_type="application/json")
    assert error.value.code is ErrorCode.MEDIA_TOO_LARGE


def test_artifact_dimension_guard_is_enforced(tmp_path: Path) -> None:
    store = ArtifactStore(Workspace(tmp_path / "ws").initialize(), max_dimension=64)
    with pytest.raises(ValidationFailure) as error:
        store.write(b"x", kind=ArtifactKind.IMAGE, media_type="image/png", width=1000, height=10)
    assert error.value.code is ErrorCode.MEDIA_TOO_LARGE


def test_artifact_json_roundtrip(store: ArtifactStore) -> None:
    reference = store.write_json({"b": 2, "a": 1}, kind=ArtifactKind.REPORT)
    assert store.read_json(reference.digest) == {"b": 2, "a": 1}


def test_artifact_register_imports_an_existing_file(store: ArtifactStore, tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    from PIL import Image

    Image.new("RGB", (12, 9), (1, 2, 3)).save(source)
    reference = store.register(source, kind=ArtifactKind.IMAGE, media_type="image/png")
    assert (reference.width, reference.height) == (12, 9)
    assert store.exists(reference.digest)
    assert source.exists()  # a copy was made, the source is untouched


def test_artifact_register_can_move(store: ArtifactStore, tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"data")
    store.register(source, kind=ArtifactKind.REPORT, media_type="application/octet-stream", move=True)
    assert not source.exists()


def test_artifact_register_refuses_a_directory(store: ArtifactStore, tmp_path: Path) -> None:
    with pytest.raises(ValidationFailure):
        store.register(tmp_path, kind=ArtifactKind.REPORT, media_type="application/json")


def test_backend_io_roundtrips_images_and_masks(store: ArtifactStore) -> None:
    import numpy as np
    from PIL import Image

    io = BackendIO(store)
    image = Image.new("RGB", (16, 12), (10, 20, 30))
    reference = io.save_image(image)
    loaded = io.image(reference)
    assert loaded.size == (16, 12)

    mask = np.zeros((12, 16), dtype=bool)
    mask[2:6, 3:9] = True
    mask_reference = io.save_mask(mask)
    restored = io.load_mask(mask_reference)
    assert restored.shape == mask.shape
    assert int(restored.sum()) == int(mask.sum())


def test_artifact_stats(store: ArtifactStore) -> None:
    store.write_json({"a": 1}, kind=ArtifactKind.REPORT)
    store.write_json({"b": 2}, kind=ArtifactKind.REPORT)
    stats = store.stats()
    assert stats["artifacts"] == 2
    assert stats["bytes"] > 0


# --------------------------------------------------------------------------- #
# State store
# --------------------------------------------------------------------------- #


def _manifest(job_id: str = "j_test", *, seed: int = 5) -> JobManifest:
    return JobManifest(
        job_id=job_id,
        state=JobState.CREATED,
        recipe_name="test",
        recipe_hash="a" * 64,
        recipe_snapshot={"recipe": "test"},
        runtime_snapshot={"profile": "local"},
        seed=seed,
        workspace_root="/tmp/ws",
        run_dir="/tmp/ws/runs/j_test",
    )


def _node(node_id: str = "n_a") -> OperationNode:
    return OperationNode(
        node_id=node_id,
        operator="test.op",
        operator_version="1.0.0",
        stage=StageName.DETECT,
        outputs={"out": PortType.ANY},
    )


def _artifact_ref() -> ArtifactRef:
    return ArtifactRef(
        kind=ArtifactKind.IMAGE, digest="f" * 64, media_type="image/png", size_bytes=8, suffix="png"
    )


def _result(node_id: str = "n_a") -> NodeResult:
    now = utc_now()
    artifact = _artifact_ref()
    return NodeResult(
        node_id=node_id,
        operator="test.op",
        operator_version="1.0.0",
        status=NodeStatus.SUCCEEDED,
        outputs={"out": PortValue.of_artifact(artifact)},
        artifacts=[artifact],
        payload={"value": 1},
        started_at=now,
        finished_at=now,
        duration_ms=2,
    )


def test_state_store_creates_the_schema(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db").initialize()
    assert store.schema_version() == 1
    assert store.counts()["jobs"] == 0
    store.close()


def test_state_store_initialize_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    StateStore(path).initialize().close()
    StateStore(path).initialize().close()


def test_job_roundtrip_and_state_transitions(state: StateStore) -> None:
    state.create_job(_manifest())
    stored = state.get_job("j_test")
    assert stored is not None
    assert stored.state is JobState.CREATED
    state.transition_job("j_test", JobState.RUNNING)
    assert state.require_job("j_test").state is JobState.RUNNING
    state.transition_job("j_test", JobState.SUCCEEDED)
    manifest = state.require_job("j_test")
    assert manifest.state is JobState.SUCCEEDED
    assert manifest.finished_at is not None


def test_transition_of_missing_job_fails(state: StateStore) -> None:
    with pytest.raises(InfrastructureFailure) as error:
        state.transition_job("missing", JobState.RUNNING)
    assert error.value.code is ErrorCode.JOB_NOT_FOUND


def test_list_and_latest_jobs(state: StateStore) -> None:
    state.create_job(_manifest("j_one"))
    state.create_job(_manifest("j_two"))
    assert {manifest.job_id for manifest in state.list_jobs()} == {"j_one", "j_two"}
    assert state.list_jobs(states=(JobState.RUNNING,)) == []
    state.transition_job("j_one", JobState.RUNNING)
    assert [manifest.job_id for manifest in state.list_jobs(states=(JobState.RUNNING,))] == ["j_one"]
    assert state.latest_job() is not None


def test_summary_and_reports_roundtrip(state: StateStore) -> None:
    from vidliner.core.results import JobSummary

    state.create_job(_manifest())
    state.set_summary(
        "j_test",
        JobSummary(
            job_id="j_test",
            state=JobState.SUCCEEDED,
            created_at=utc_now(),
            counters=JobCounters(samples=2, accepted=1),
        ),
    )
    assert state.require_job("j_test").counters.samples == 2
    state.record_report("j_test", "review", "runs/j_test/review/index.html")
    assert state.reports_for("j_test") == {"review": "runs/j_test/review/index.html"}


def test_node_result_roundtrip_supports_resume(state: StateStore) -> None:
    state.create_job(_manifest())
    result = _result()
    state.record_node_result(
        "j_test", result, stage="detect", lineage="sample:s_1", config_hash="c" * 64, seed=3, input_digests=[]
    )
    restored = state.node_result("j_test", "n_a")
    assert restored is not None
    assert restored.payload["value"] == 1
    assert restored.outputs["out"].artifact is not None
    assert state.node_result("j_test", "n_missing") is None


def test_node_evidence_is_queryable(state: StateStore) -> None:
    from vidliner.core.results import OperationEvidence

    state.create_job(_manifest())
    state.record_node_evidence(
        OperationEvidence(
            job_id="j_test",
            node_id="n_a",
            operator="test.op",
            operator_version="1.0.0",
            stage="detect",
            lineage="sample:s_1",
            status=NodeStatus.SUCCEEDED,
            config_hash="c" * 64,
            seed=1,
            duration_ms=4,
        )
    )
    rows = state.node_evidence("j_test")
    assert len(rows) == 1
    assert rows[0]["operator"] == "test.op"
    assert state.stage_totals("j_test") == {"detect": {"succeeded": 1}}


def test_node_cache_roundtrip_and_hit_counting(state: StateStore) -> None:
    result = _result()
    state.cache_put(
        "key-1", result, operator="test.op", operator_version="1", implementation="x", config_hash="c"
    )
    first = state.cache_get("key-1")
    assert first is not None
    assert state.cache_get("key-1") is not None
    assert state.cache_stats()["hits"] == 2
    assert state.clear_cache() == 1
    assert state.cache_get("key-1") is None


def test_artifact_catalogue_tracks_kinds(state: StateStore) -> None:
    reference = ArtifactRef(
        kind=ArtifactKind.MASK, digest="d" * 64, media_type="image/png", size_bytes=10, suffix="png"
    )
    state.record_artifact(reference, rel_path="artifacts/dd/x.png", job_id="j_test")
    state.record_artifact(reference, rel_path="artifacts/dd/x.png", job_id="j_test")
    assert state.artifact_kinds("d" * 64) == ("mask",)


def test_candidate_lifecycle_and_events(state: StateStore) -> None:
    state.create_job(_manifest())
    state.upsert_candidate(
        candidate_id="c_1",
        job_id="j_test",
        sample_id="s_1",
        target_object_id="o_1",
        candidate_key="sedan-1",
        category="sedan",
        state=CandidateState.GENERATED,
        seed=7,
        source_digest="a" * 64,
        output_digest=None,
        overall_score=None,
        policy_hash="p",
        plan_json={},
        intent_json={},
    )
    state.append_candidate_event("c_1", to_state=CandidateState.GENERATED)
    state.set_candidate_state(
        "c_1", CandidateState.ACCEPTED, reason_codes=(), output_digest="b" * 64, overall_score=0.9
    )
    stored = state.get_candidate("c_1")
    assert stored is not None
    assert stored.state is CandidateState.ACCEPTED
    assert stored.overall_score == 0.9
    events = state.candidate_events("c_1")
    assert [event["to_state"] for event in events] == ["generated", "accepted"]
    assert state.candidate_counts("j_test") == {"accepted": 1}
    assert len(state.candidates("j_test", states=(CandidateState.ACCEPTED,))) == 1


def test_set_candidate_state_requires_an_existing_row(state: StateStore) -> None:
    with pytest.raises(InfrastructureFailure):
        state.set_candidate_state("missing", CandidateState.ACCEPTED)


def test_quality_metrics_roundtrip_and_averages(state: StateStore) -> None:
    state.create_job(_manifest())
    state.upsert_candidate(
        candidate_id="c_1",
        job_id="j_test",
        sample_id="s_1",
        target_object_id="o_1",
        candidate_key="k",
        category="sedan",
        state=CandidateState.EVALUATING,
        seed=1,
        source_digest="a" * 64,
        output_digest=None,
        overall_score=None,
        policy_hash="p",
        plan_json={},
        intent_json={},
    )
    state.record_metrics(
        "c_1",
        [
            {"metric": "semantic_match", "value": 0.9, "threshold": 0.8, "status": "passed"},
            {"metric": "geometry", "value": 1.0, "threshold": 1.0, "status": "passed"},
        ],
    )
    rows = state.metrics_for("c_1")
    assert {row["metric"] for row in rows} == {"semantic_match", "geometry"}
    averages = state.metric_averages("j_test")
    assert averages["semantic_match"] == pytest.approx(0.9)


def test_sample_registration_lineage_and_leakage(state: StateStore) -> None:
    state.register_sample(
        sample_id="s_source",
        job_id=None,
        root_digest="r" * 64,
        digest="d" * 64,
        rel_path="train/a.png",
        split="train",
        state=SampleState.SOURCE,
    )
    state.create_job(_manifest())
    state.register_sample(
        sample_id="s_child",
        job_id="j_test",
        root_digest="r" * 64,
        digest="e" * 64,
        rel_path="out/a.png",
        split="val",
        state=SampleState.ACCEPTED,
        source_sample_id="s_source",
        augmentation_depth=1,
    )
    state.link_lineage("s_child", "s_source")
    assert state.ancestors("s_child") == ("s_source",)
    leakage = state.find_leakage()
    assert leakage == [("s_child", "s_source", "val")]


def test_perceptual_hashes_and_duplicates(state: StateStore) -> None:
    state.register_sample(
        sample_id="s_1",
        job_id=None,
        root_digest="r" * 64,
        digest="d" * 64,
        rel_path="a.png",
        split="train",
        state=SampleState.SOURCE,
        perceptual_hash="0123456789abcdef",
    )
    assert state.perceptual_hashes() == {"s_1": "0123456789abcdef"}
    state.record_duplicate("j_test", "c_1", "s_1", kind="near", distance=2.0)
    assert state.duplicates("j_test")[0]["duplicate_of"] == "s_1"


# --------------------------------------------------------------------------- #
# Evidence recorder
# --------------------------------------------------------------------------- #


def test_evidence_recorder_roundtrips_results_and_verifies_artifacts(state: StateStore) -> None:
    state.create_job(_manifest())
    recorder = EvidenceRecorder(state, job_id="j_test", artifact_exists=lambda artifact: True)
    node = _node()
    result = _result()
    recorder.record_node(result, node, seed=1, config_hash="c" * 64)
    assert recorder.completed_node("n_a") is not None
    assert len(state.node_evidence("j_test")) == 1

    recorder.store_cache("key", result, node)
    assert recorder.cached_result("key") is not None

    missing = EvidenceRecorder(state, job_id="j_test", artifact_exists=lambda artifact: False)
    assert missing.cached_result("key") is None


def test_evidence_recorder_ignores_failed_nodes_for_resume(state: StateStore) -> None:
    state.create_job(_manifest())
    recorder = EvidenceRecorder(state, job_id="j_test")
    failed = _result().model_copy(update={"status": NodeStatus.FAILED})
    recorder.record_node(failed, _node(), seed=1, config_hash="c" * 64)
    assert recorder.completed_node("n_a") is None


def test_in_memory_run_state_behaves_like_the_store() -> None:
    state = InMemoryRunState()
    node = _node()
    result = _result()
    state.record_node(result, node, seed=1, config_hash="c" * 64)
    assert state.completed_node("n_a") is not None
    state.store_cache("key", result, node)
    assert state.cached_result("key") is not None
    assert state.artifacts_present(result)
    strict = InMemoryRunState(artifact_exists=lambda artifact: False)
    strict.store_cache("key", result, _node())
    assert strict.cached_result("key") is None


def test_in_memory_run_state_drops_cache_when_artifacts_vanish() -> None:
    state = InMemoryRunState(artifact_exists=lambda artifact: False)
    node = _node()
    result = _result()
    state.store_cache("key", result, node)
    assert state.cached_result("key") is None


def test_database_file_is_created_where_expected(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "ws").initialize()
    StateStore(workspace.state_path).initialize().close()
    assert workspace.state_path.is_file()
    assert json.loads("{}") == {}
