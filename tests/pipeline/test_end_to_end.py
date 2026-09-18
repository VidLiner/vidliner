"""End-to-end pipeline tests over the synthetic 10-image dataset.

These are the tests that correspond to the product's Definition of Done: read a dataset, find
objects, segment them, generate candidates, evaluate them, rebuild annotations, export COCO and YOLO,
resume an interrupted job, and reuse the cache on a re-run — all with no paid API and no downloaded
model.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tests.conftest import make_recipe
from vidliner.domain.enums import CandidateState, DecisionState, JobState, NodeStatus
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
# The whole pipeline
# --------------------------------------------------------------------------- #


def test_end_to_end_run_over_ten_images(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=3, values=("sedan", "suv", "pickup")))
    outcome = _run(session, recipe, workers=4)

    assert outcome.manifest.state is JobState.SUCCEEDED
    counters = outcome.manifest.counters
    assert counters.samples == 10
    assert counters.targets == 10
    assert counters.generated == 30
    assert counters.accepted == 30
    assert counters.failed == 0
    assert counters.acceptance_rate == pytest.approx(1.0)

    # Every sample produced evidence on disk.
    run_dir = Path(outcome.manifest.run_dir)
    assert (run_dir / "manifest.json").exists() is False  # the manifest lives in the database
    assert (run_dir / "job-summary.json").is_file()
    assert (run_dir / "events.jsonl").is_file()
    sample_dirs = [path for path in (run_dir / "samples").iterdir() if path.is_dir()]
    assert len(sample_dirs) == 10
    for sample_dir in sample_dirs:
        assert (sample_dir / "source.png").is_file()
        # Evidence is per candidate: a sample with three candidates keeps three of each.
        assert list(sample_dir.glob("refined-*.png"))
        assert list(sample_dir.glob("annotation-*.json"))
        assert list(sample_dir.glob("quality-*.json"))
        assert list(sample_dir.glob("candidate-*.json"))


def test_run_exports_a_coco_dataset(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=2))
    outcome = _run(session, recipe)
    assert outcome.export is not None
    root = outcome.export.path
    assert (root / "annotations" / "instances_train.json").is_file()
    assert (root / "manifest.json").is_file()
    assert (root / "dataset-report.json").is_file()
    assert (root / "provenance").is_dir()
    images = list((root / "images").glob("*.png"))
    assert len(images) == outcome.export.accepted

    document = json.loads((root / "annotations" / "instances_train.json").read_text())
    assert len(document["images"]) == len(images)
    assert len(document["annotations"]) == len(images)
    for annotation in document["annotations"]:
        assert annotation["bbox"][2] > 0
        assert annotation["bbox"][3] > 0
        assert annotation["area"] > 0
        assert annotation["segmentation"]


def test_run_exports_yolo(session: Session, dataset_root: Path) -> None:
    yolo_recipe = make_recipe("./data/source", candidates=1, output="./data/yolo")
    yolo_recipe["export"] = {**yolo_recipe["export"], "format": "yolo-segmentation"}  # type: ignore[dict-item]
    recipe = _recipe(yolo_recipe)
    outcome = _run(session, recipe)
    assert outcome.export is not None
    root = outcome.export.path
    assert (root / "classes.txt").is_file()
    assert (root / "data.yaml").is_file()
    labels = list((root / "labels").glob("*.txt"))
    assert labels
    fields = labels[0].read_text().split()
    assert len(fields) >= 7  # class index plus at least three polygon points


def test_rejected_candidates_never_enter_the_dataset(session: Session, dataset_root: Path) -> None:
    """A policy that no candidate can satisfy must produce a dataset with no samples."""
    recipe = _recipe(
        make_recipe(
            "./data/source",
            candidates=2,
            minimum_overall=0.999,
            hard_gates={
                "semantic_match": 0.999,
                "background_preservation": 0.999,
                "annotation_consistency": 0.999,
            },
        )
    )
    outcome = _run(session, recipe)
    assert outcome.manifest.state is JobState.SUCCEEDED
    assert outcome.manifest.counters.accepted == 0
    assert outcome.manifest.counters.rejected == outcome.manifest.counters.generated
    assert outcome.export is not None
    assert outcome.export.accepted == 0
    assert list((outcome.export.path / "images").glob("*.png")) == []


def test_quality_rejection_is_not_a_job_failure(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(
        make_recipe(
            "./data/source",
            candidates=1,
            minimum_overall=0.999,
            hard_gates={"semantic_match": 0.999},
        )
    )
    outcome = _run(session, recipe)
    assert outcome.manifest.state is JobState.SUCCEEDED
    assert outcome.manifest.counters.failed == 0
    assert outcome.manifest.counters.rejected == outcome.manifest.counters.generated
    candidates = session.state.candidates(outcome.manifest.job_id)
    assert all(candidate.state is CandidateState.REJECTED for candidate in candidates)
    assert any(candidate.reason_codes for candidate in candidates)


def test_provenance_records_everything_the_requirement_lists(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1, seed=1234))
    outcome = _run(session, recipe)
    assert outcome.export is not None
    provenance_files = sorted((outcome.export.path / "provenance").glob("*.json"))
    assert provenance_files
    record = json.loads(provenance_files[0].read_text())
    for key in (
        "source_sample_id",
        "source_digest",
        "recipe_hash",
        "job_id",
        "target_object_id",
        "replacement_category",
        "seeds",
        "operator_versions",
        "backend_ids",
        "generation_parameters",
        "quality_scores",
        "decision",
        "output_digest",
        "created_at",
    ):
        assert key in record, key
    assert record["seeds"]["job_seed"] == 1234
    assert record["decision"] == DecisionState.ACCEPTED.value


def test_no_credential_material_reaches_the_manifest_or_dataset(
    session: Session, dataset_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIDLINER_FAKE_KEY", "super-secret-value")
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    outcome = _run(session, recipe)
    manifest_text = json.dumps(outcome.manifest.model_dump(mode="json"))
    assert "super-secret-value" not in manifest_text
    assert outcome.export is not None
    dataset_text = (outcome.export.path / "manifest.json").read_text()
    assert "super-secret-value" not in dataset_text


def test_annotations_are_regenerated_not_copied(session: Session, dataset_root: Path) -> None:
    """The rebuilt annotation must carry the replacement category, not the source class."""
    recipe = _recipe(make_recipe("./data/source", candidates=1, values=("sedan",)))
    outcome = _run(session, recipe)
    assert outcome.export is not None
    document = json.loads((outcome.export.path / "annotations" / "instances_train.json").read_text())
    names = {category["name"] for category in document["categories"]}
    assert "sedan" in names
    assert "car" not in names


def test_seed_tree_is_recorded_per_sample(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1, seed=99))
    outcome = _run(session, recipe)
    assert outcome.manifest.seed == 99
    assert len(outcome.manifest.seed_tree) == 10
    assert len(set(outcome.manifest.seed_tree.values())) == 10


def test_job_id_is_stable_for_the_same_recipe_and_dataset(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    plan, _ = session.plan(recipe)
    again, _ = session.plan(recipe)
    assert plan.job_plan.job_id == again.job_plan.job_id
    other = _recipe(make_recipe("./data/source", candidates=1, seed=500))
    third, _ = session.plan(other)
    assert third.job_plan.job_id != plan.job_plan.job_id


def test_plan_is_free_and_reports_no_unmet_capabilities(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=2))
    plan, discovery = session.plan(recipe)
    assert plan.is_runnable
    assert plan.job_plan.node_count > 0
    assert discovery.count == 10
    estimate = plan.job_plan.estimate
    assert estimate.samples == 10
    assert estimate.candidates == len(plan.assembly.nodes("candidate"))
    assert estimate.nodes == plan.job_plan.node_count
    assert estimate.stages


def test_plan_document_can_be_written_and_read(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    outcome = _run(session, recipe, dry_run=True)
    assert outcome.manifest.state is JobState.SUCCEEDED
    assert outcome.export is None
    document = session.plan_document(outcome.manifest.job_id)
    assert document is not None
    assert document["nodes"]
    assert document["discovery"]["samples"] == 10


def test_dry_run_does_not_call_the_generator(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=2))
    outcome = _run(session, recipe, dry_run=True)
    assert outcome.manifest.counters.generated == 0
    candidates = session.state.candidates(outcome.manifest.job_id)
    assert candidates == []


# --------------------------------------------------------------------------- #
# Multi-target samples
# --------------------------------------------------------------------------- #


def test_multiple_targets_per_sample_fan_out_and_skip_absent_branches(
    session: Session, dataset_root: Path
) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1, max_per_sample=3))
    outcome = _run(session, recipe, workers=4)
    assert outcome.manifest.state is JobState.SUCCEEDED
    # Each synthetic image has exactly one object, so one branch is used and two are skipped.
    assert outcome.manifest.counters.targets == 10
    assert outcome.manifest.counters.generated == 10
    assert outcome.manifest.counters.failed == 0
    assert outcome.manifest.counters.nodes_succeeded > 0


def test_class_filter_produces_no_targets_and_still_succeeds(session: Session, dataset_root: Path) -> None:
    """A class the detector cannot report yields no targets, and that is a success, not a failure."""
    recipe = _recipe(make_recipe("./data/source", classes=("spaceship",), candidates=1))
    outcome = _run(session, recipe)
    assert outcome.manifest.state is JobState.SUCCEEDED
    assert outcome.manifest.counters.targets == 0
    assert outcome.manifest.counters.generated == 0
    assert outcome.manifest.counters.failed == 0


# --------------------------------------------------------------------------- #
# Split safety
# --------------------------------------------------------------------------- #


def test_directory_splits_are_discovered_and_only_train_is_augmented(
    session: Session, dataset_root: Path
) -> None:
    import shutil

    from vidliner.fixtures import build_dataset

    shutil.rmtree(dataset_root)
    build_dataset(dataset_root, count=10, annotations=False, split_mode="directory")
    recipe = _recipe(
        make_recipe(
            "./data/source",
            candidates=1,
            split_mode="directory",
            augment_splits=("train",),
            source_format=None,
        )
    )
    _plan, discovery = session.plan(recipe)
    # The fixture writes eight train images and two validation images, and only train is augmentable.
    assert discovery.split_counts.get("train") == 8
    assert discovery.split_counts.get("validation") == 2
    contexts = session.sample_contexts(recipe, discovery)
    assert len(contexts) == 8
    assert all(context.split == "train" for context in contexts.values())


def test_recipe_refuses_to_augment_the_test_split() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="refused by default"):
        _recipe(
            make_recipe(
                "./source",
                candidates=1,
                split_mode="directory",
                augment_splits=("train", "test"),
            )
        )


def test_dataset_format_round_trips_through_the_pipeline(session: Session, dataset_root: Path) -> None:
    """A YOLO-annotated input dataset must produce the same pipeline behaviour as a COCO one."""
    import shutil

    from vidliner.fixtures import SceneObject, SyntheticDataset, build_dataset, write_yolo

    shutil.rmtree(dataset_root)
    build_dataset(dataset_root, count=6, annotations=False)
    dataset = SyntheticDataset(root=dataset_root, images=[], objects={}, splits={}, categories=["car"])
    for path in sorted(dataset_root.glob("*.png")):
        dataset.images.append(path)
        dataset.objects[path.name] = [SceneObject("car", (20, 20, 80, 80), (200, 40, 40), label=0)]
        dataset.splits[path.name] = "train"
    write_yolo(dataset_root, dataset, list(dataset.splits), segmentation=False)

    recipe = _recipe(
        make_recipe("./data/source", candidates=1, fmt="yolo-detection", source_format="yolo-detection")
    )
    outcome = _run(session, recipe)
    assert outcome.manifest.state is JobState.SUCCEEDED
    assert outcome.manifest.counters.generated == 6


def test_engine_report_is_available_and_consistent(session: Session, dataset_root: Path) -> None:
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
        options=JobOptions(workers=2),
    )
    manifest = asyncio.run(runner.execute())
    report = runner.report
    assert report is not None
    assert report.succeeded()
    assert report.count(NodeStatus.SUCCEEDED) > 0
    assert manifest.counters.nodes_succeeded == report.count(NodeStatus.SUCCEEDED)
    assert runner.summary().job_id == manifest.job_id


def test_sample_outcomes_record_reason_histograms(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(
        make_recipe(
            "./data/source",
            candidates=1,
            minimum_overall=0.999,
            hard_gates={"semantic_match": 0.999},
        )
    )
    outcome = _run(session, recipe)
    total_reasons = sum(sum(item.reason_histogram.values()) for item in outcome.outcomes.values())
    assert total_reasons >= 10
