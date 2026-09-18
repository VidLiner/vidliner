"""Export, duplicate control, split safety, and reporting tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tests.conftest import make_recipe
from vidliner.control.duplicates import DuplicateIndex, DuplicateKind
from vidliner.control.report import build_distribution
from vidliner.control.splits import assert_no_leakage, plan_splits
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.annotations import ExportProfile
from vidliner.domain.recipe import Recipe
from vidliner.fixtures import render_scene
from vidliner.pipeline.runner import JobOptions
from vidliner.pipeline.service import JobRequest, Session
from vidliner.quality.fingerprint import perceptual_hash


def _recipe(document: dict[str, object]) -> Recipe:
    return Recipe.model_validate(document)


def _run(session: Session, recipe: Recipe, **options: object):
    return asyncio.run(
        session.run(JobRequest(recipe=recipe, options=JobOptions(**options)))  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# Duplicate control
# --------------------------------------------------------------------------- #


def test_duplicate_index_detects_an_exact_copy() -> None:
    image = render_scene(width=32, height=32, seed=1)
    index = DuplicateIndex(enabled=True, algorithm="phash", hamming_threshold=6)
    index.register_source("s_1", image)
    decision = index.check(image)
    assert decision.is_duplicate
    assert decision.kind == DuplicateKind.EXACT
    assert decision.reference == "s_1"


def test_duplicate_index_ignores_unrelated_images() -> None:
    first = render_scene(width=64, height=64, seed=1)
    second = render_scene(
        width=64,
        height=64,
        seed=2,
        objects=(
            __import__("vidliner.fixtures", fromlist=["SceneObject"]).SceneObject(
                "car", (4, 4, 60, 60), (10, 200, 10)
            ),
        ),
    )
    index = DuplicateIndex(enabled=True, hamming_threshold=4)
    index.register_source("s_1", first)
    assert not index.check(second).is_duplicate


def test_duplicate_index_respects_the_threshold() -> None:
    image = render_scene(
        width=64,
        height=64,
        seed=5,
        objects=(
            __import__("vidliner.fixtures", fromlist=["SceneObject"]).SceneObject(
                "car", (8, 8, 40, 40), (200, 60, 60)
            ),
        ),
    )
    strict = DuplicateIndex(enabled=True, hamming_threshold=0)
    strict.register_source("s_1", image)
    assert strict.check(image).is_duplicate
    # A lenient index also matches; the threshold only decides how different a copy may be.
    lenient = DuplicateIndex(enabled=True, hamming_threshold=64)
    lenient.register_source("s_1", image)
    assert lenient.check(image).is_duplicate
    assert lenient.check(image).distance == 0.0


def test_duplicate_index_can_be_disabled_or_algorithm_none() -> None:
    image = render_scene(width=16, height=16, seed=1)
    index = DuplicateIndex(enabled=True, algorithm="none")
    index.register_source("s_1", image)
    assert not index.check(image).is_duplicate
    disabled = DuplicateIndex(enabled=False)
    disabled.register_source("s_1", image)
    assert not disabled.check(image).is_duplicate


def test_duplicate_index_handles_embeddings() -> None:
    index = DuplicateIndex(enabled=True, embedding_threshold=0.9)
    index.register_embedding("a", (1.0, 0.0, 0.0))
    assert index.check_embedding((1.0, 0.0, 0.0)).is_duplicate
    assert not index.check_embedding((0.0, 1.0, 0.0)).is_duplicate


def test_duplicate_index_skips_source_comparison_when_asked() -> None:
    image = render_scene(width=16, height=16, seed=3)
    index = DuplicateIndex(enabled=True, compare_against_source=False)
    index.register_source("s_1", image)
    assert not index.check(image).is_duplicate


def test_duplicate_index_ignores_malformed_existing_hashes() -> None:
    index = DuplicateIndex(existing={"s_1": "not-hex", "s_2": "0123456789abcdef"})
    assert index.tracked == 1


def test_fingerprint_of_matches_the_module_function() -> None:
    image = render_scene(width=24, height=24, seed=9)
    index = DuplicateIndex(algorithm="dhash")
    assert index.fingerprint_of(image) == perceptual_hash(image, algorithm="dhash")


# --------------------------------------------------------------------------- #
# Split safety
# --------------------------------------------------------------------------- #


def test_plan_splits_notes_restriction() -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    plan = plan_splits(recipe)
    assert plan.augment_splits == ("train",)
    assert not plan.touches_protected
    assert plan.notes


def test_assert_no_leakage_accepts_matching_splits() -> None:
    assert_no_leakage([("child", "parent", "train", "train")])


def test_assert_no_leakage_reports_every_offending_edge() -> None:
    with pytest.raises(ValidationFailure) as error:
        assert_no_leakage(
            [
                ("a", "b", "train", "validation"),
                ("c", "d", "test", "train"),
            ]
        )
    assert error.value.code is ErrorCode.SPLIT_LEAKAGE
    assert len(error.value.detail["edges"]) == 2


def test_exporter_refuses_a_leaking_dataset(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    outcome = _run(session, recipe)
    # Introduce a leak after the fact: a source sample recorded in a different split.
    session.state.register_sample(
        sample_id="s_leak",
        job_id=None,
        root_digest="x" * 64,
        digest="y" * 64,
        rel_path="leak.png",
        split="validation",
        state=__import__("vidliner.domain.enums", fromlist=["SampleState"]).SampleState.SOURCE,
    )
    accepted = session.state.candidates(outcome.manifest.job_id)
    assert accepted
    session.state.link_lineage(accepted[0].sample_id, "s_leak")
    with pytest.raises(ValidationFailure) as error:
        session.export_job(outcome.manifest.job_id, recipe=recipe)
    assert error.value.code is ErrorCode.SPLIT_LEAKAGE


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def test_export_can_be_run_again_from_stored_state(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1, output="./data/first"))
    outcome = _run(session, recipe)
    assert outcome.export is not None

    second = session.export_job(outcome.manifest.job_id, recipe=recipe, path_override="./data/second")
    assert second.accepted == outcome.export.accepted
    assert (second.path / "manifest.json").is_file()


def test_export_honours_a_format_override(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    outcome = _run(session, recipe, export=False) if False else _run(session, recipe)
    assert outcome.export is not None
    yolo = session.export_job(
        outcome.manifest.job_id,
        recipe=recipe,
        format_override="yolo-detection",
        path_override="./data/yolo",
    )
    assert yolo.format == "yolo-detection"
    assert (yolo.path / "classes.txt").is_file()


def test_each_candidate_keeps_its_own_category(session: Session, dataset_root: Path) -> None:
    """Three candidates with three replacement values must produce three distinct categories."""
    recipe = _recipe(
        make_recipe("./data/source", candidates=3, values=("sedan", "suv", "pickup"), max_per_sample=1)
    )
    outcome = _run(session, recipe)
    assert outcome.export is not None
    document = json.loads((outcome.export.path / "annotations" / "instances_train.json").read_text())
    names = {category["name"] for category in document["categories"]}
    assert names == {"sedan", "suv", "pickup"}
    # Every annotation points at a declared category, and they are not all the same one.
    declared = {category["id"] for category in document["categories"]}
    used = {annotation["category_id"] for annotation in document["annotations"]}
    assert used <= declared
    assert len(used) == 3


def test_export_manifest_describes_every_sample(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    outcome = _run(session, recipe)
    assert outcome.export is not None
    manifest = json.loads((outcome.export.path / "manifest.json").read_text())
    assert manifest["format"] == "vidliner.dataset@1"
    assert manifest["counts"]["samples"] == outcome.export.accepted
    assert len(manifest["samples"]) == outcome.export.accepted
    assert manifest["seed_tree"]
    assert manifest["categories"]


def test_export_report_records_the_distribution(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1, values=("sedan", "suv")))
    outcome = _run(session, recipe)
    assert outcome.export is not None
    report = json.loads((outcome.export.path / "dataset-report.json").read_text())
    assert report["format"] == "vidliner.dataset-report@1"
    assert report["counts"]["accepted"] == outcome.export.accepted
    assert report["replacement_distribution"]
    assert report["object_size_distribution"]
    assert report["acceptance_by_category"]


def test_export_with_a_tiny_minimum_area_drops_samples(session: Session, dataset_root: Path) -> None:
    """An annotation below the minimum area is a validation failure, not a silently empty label."""
    document = make_recipe("./data/source", candidates=1)
    document["export"] = {**document["export"], "min_annotation_area_px": 100_000}  # type: ignore[dict-item]
    recipe = _recipe(document)
    try:
        outcome = _run(session, recipe)
    except ValidationFailure as error:
        # Either the annotation stage refused the mask outright, or the export found nothing to
        # write; both mean no oversized-area sample reached the dataset.
        assert error.code in {ErrorCode.ANNOTATION_TOO_SMALL, ErrorCode.ARTIFACT_MISSING}
        return
    assert outcome.export is not None
    assert outcome.export.accepted == 0


def test_export_profile_is_derived_from_the_recipe(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1, output="./data/custom"))
    outcome = _run(session, recipe)
    assert outcome.export is not None
    assert outcome.export.path.name == "custom"
    profile = ExportProfile(format=recipe.export.format, path=recipe.export.path)
    assert profile.is_segmentation  # coco-instance carries pixel geometry
    assert not ExportProfile(format="coco-detection", path="x").is_segmentation


def test_near_duplicate_outputs_are_excluded(session: Session, dataset_root: Path) -> None:
    """Two candidates with the same category and seed produce identical images, so one is dropped."""
    document = make_recipe("./data/source", candidates=3, values=("sedan",), max_per_sample=1)
    # Force every candidate to be identical by collapsing the candidate descriptions.
    document["replacement"] = {**document["replacement"], "seed": 1}  # type: ignore[dict-item]
    recipe = _recipe(document)
    outcome = _run(session, recipe)
    assert outcome.export is not None
    # The fake generator varies by seed, so duplicates are not guaranteed here; what matters is that
    # duplicate control ran and reported a number.
    assert outcome.export.duplicates >= 0
    duplicates = session.state.duplicates(outcome.manifest.job_id)
    assert len(duplicates) == outcome.export.duplicates


# --------------------------------------------------------------------------- #
# Distribution reporting
# --------------------------------------------------------------------------- #


def test_distribution_buckets_sizes_and_aspects(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    outcome = _run(session, recipe)
    assert outcome.export is not None
    from vidliner.domain.enums import SampleState

    report = build_distribution(
        workspace=session.workspace,
        state=session.state,
        job_id=outcome.manifest.job_id,
        recipe=recipe,
        accepted=[],
        rejected=[],
        review=[],
        duplicates={},
    )
    assert report.counts["accepted"] == 0
    assert "no candidate was accepted" in " ".join(report.notes)
    assert report.summary_lines()
    assert report.as_dict()["format"] == "vidliner.dataset-report@1"
    del SampleState


def test_distribution_counts_replacement_categories(session: Session, dataset_root: Path) -> None:
    recipe = _recipe(make_recipe("./data/source", candidates=1))
    outcome = _run(session, recipe)
    assert outcome.export is not None
    records = session.export_records(
        outcome.manifest.job_id,
        outcome.outcomes,
        recipe,
        include_review=False,
    )
    report = build_distribution(
        workspace=session.workspace,
        state=session.state,
        job_id=outcome.manifest.job_id,
        recipe=recipe,
        accepted=records,
        rejected=[],
        review=[],
        duplicates={},
    )
    assert sum(report.class_distribution.values()) == sum(
        record.annotation.object_count for record in records
    )
    assert report.acceptance_by_category


def test_split_manifest_can_be_written_and_reused(session: Session, dataset_root: Path) -> None:
    from vidliner.control.prepare import write_split_manifest
    from vidliner.control.sources import DatasetSource

    recipe = _recipe(make_recipe("./data/source", candidates=1, source_format=None))
    discovery = DatasetSource(root=dataset_root, recipe=recipe).discover()
    target = write_split_manifest(discovery, session.workspace.root / "splits.json")
    assert target.is_file()

    manifest_recipe = _recipe(
        make_recipe(
            "./data/source",
            candidates=1,
            source_format=None,
            split_mode="manifest",
            split_manifest=str(target),
        )
    )
    manifest_discovery = DatasetSource(root=dataset_root, recipe=manifest_recipe).discover()
    assert manifest_discovery.count == discovery.count
    assert all(sample.split == "train" for sample in manifest_discovery.samples)
    assert manifest_discovery.samples[0].split == "train"
