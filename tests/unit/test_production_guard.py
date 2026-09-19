"""Tests for the production guard and the shared evidence naming convention.

Two properties are asserted here:

* a demonstration stack may run a job but may not silently produce a dataset, and the refusal names
  every offending capability rather than one at a time;
* the per-candidate evidence file names are spelled the same way by the writer and every reader, so a
  resumed run cannot quietly find nothing.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tests.conftest import make_recipe
from vidliner.capabilities.names import PRODUCTION_CAPABILITIES, requires_production_backend
from vidliner.cli.main import app
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.enums import JobState
from vidliner.domain.recipe import Recipe
from vidliner.pipeline.export_names import evidence_path, file_key, kinds_for, matches_kind
from vidliner.pipeline.runner import JobOptions
from vidliner.pipeline.service import JobRequest, Session
from vidliner.runtime.production import (
    DemoUsage,
    ProductionVerdict,
    assert_production_ready,
    assess_production_readiness,
)
from vidliner.runtime.profile import BackendSpec, default_profile
from vidliner.runtime.registry import BackendRegistry

#: The four capabilities whose output *becomes* the training data, bound to backends that do not
#: declare themselves stand-ins. The classes are declarative markers (see the fixture module); the
#: guard reads one class attribute and never constructs them.
_PRODUCTION_BINDINGS = {
    "vision.object_detection.v1": (
        "ready_detector",
        "tests.fixtures.production_backends:ReadyDetectorBackend",
    ),
    "vision.instance_segmentation.v1": (
        "ready_segmenter",
        "tests.fixtures.production_backends:ReadySegmenterBackend",
    ),
    "generation.object_replacement.v1": (
        "ready_replacement",
        "tests.fixtures.production_backends:ReadyReplacementBackend",
    ),
    "quality.semantic_match.v1": (
        "ready_evaluator",
        "tests.fixtures.production_backends:ReadyEvaluatorBackend",
    ),
}


def _registry(**overrides: BackendSpec) -> BackendRegistry:
    profile = default_profile()
    return BackendRegistry(profile.model_copy(update={"backends": {**profile.backends, **overrides}}))


def _ready_specs() -> dict[str, BackendSpec]:
    """Profile entries for the non-demo backends that serve the production-critical capabilities."""
    return {name: BackendSpec(use=use) for name, use in _PRODUCTION_BINDINGS.values()}


def _production_registry() -> BackendRegistry:
    """A registry whose production-critical capabilities are served by non-demo backends.

    Every other capability is already production-grade in the default profile — a deterministic mask
    filter is exactly what it says it is — so only the four critical ones change.
    """
    return _registry(**_ready_specs())


def _candidate_bindings() -> dict[str, str]:
    """``capability → backend`` for the four production-critical capabilities."""
    return {capability: name for capability, (name, _) in _PRODUCTION_BINDINGS.items()}


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #


def test_production_capabilities_are_the_ones_whose_output_becomes_the_data() -> None:
    expected = {
        "vision.object_detection.v1",
        "vision.instance_segmentation.v1",
        "generation.object_replacement.v1",
        "quality.semantic_match.v1",
    }
    assert set(PRODUCTION_CAPABILITIES) == expected
    assert requires_production_backend("vision.object_detection.v1")
    assert not requires_production_backend("generation.mask_refinement.v1")


def test_the_default_profile_is_reported_as_a_demonstration_stack() -> None:
    """The profile the project ships must not be silently usable as a training-data producer."""
    registry = _registry()
    verdict = assess_production_readiness(registry, dict(registry.profile.bindings))
    assert not verdict.is_production_ready
    assert set(verdict.capabilities_involved) == set(PRODUCTION_CAPABILITIES)
    assert all(usage.backend.startswith("builtin_") for usage in verdict.demo_usage)


def test_a_production_stack_is_reported_ready() -> None:
    verdict = assess_production_readiness(_production_registry(), _candidate_bindings())
    assert verdict.is_production_ready
    assert verdict.explanation() == ()


def test_the_verdict_explains_every_offending_binding() -> None:
    registry = _registry()
    verdict = assess_production_readiness(registry, dict(registry.profile.bindings))
    explanation = verdict.explanation()
    assert len(explanation) == len(PRODUCTION_CAPABILITIES)
    assert all("demonstration stand-in" in line for line in explanation)


def test_the_refusal_lists_all_capabilities_and_a_remedy() -> None:
    registry = _registry()
    verdict = assess_production_readiness(registry, dict(registry.profile.bindings))
    with pytest.raises(ValidationFailure) as error:
        assert_production_ready(verdict)
    assert error.value.code is ErrorCode.DEMO_BACKEND_NOT_ALLOWED
    detail = error.value.detail
    assert set(detail["capabilities"]) == set(PRODUCTION_CAPABILITIES)
    assert "acceptance.allow_demo_backends" in detail["remedy"]


def test_a_ready_verdict_does_not_raise() -> None:
    assert_production_ready(ProductionVerdict())


def test_a_demo_declared_only_by_the_backend_class_is_still_caught() -> None:
    """The profile entry is the convenient place to declare it; the class is authoritative."""
    profile = default_profile()
    spec = profile.backends["builtin_detector"].model_copy(update={"demo_only": False})
    updated = profile.model_copy(update={"backends": {**profile.backends, "builtin_detector": spec}})
    verdict = assess_production_readiness(
        BackendRegistry(updated), {"vision.object_detection.v1": "builtin_detector"}
    )
    assert not verdict.is_production_ready
    assert verdict.demo_usage[0].reason == "declared by the backend itself"


def test_a_backend_that_cannot_be_imported_is_not_treated_as_demo() -> None:
    """An unusable import is a pre-flight problem, not a data-integrity one."""
    registry = _registry(broken=BackendSpec(use="vidliner.backends.nope:MissingBackend"))
    verdict = assess_production_readiness(registry, {"vision.object_detection.v1": "broken"})
    assert verdict.is_production_ready


def test_non_critical_capabilities_do_not_block_production() -> None:
    """A demonstration planner or refiner changes how a sample was made, not whether its label holds."""
    verdict = assess_production_readiness(
        _registry(),
        {
            "planning.replacement.v1": "builtin_planner",
            "vision.scene_analysis.v1": "builtin_scene",
            "generation.mask_refinement.v1": "builtin_refiner",
        },
    )
    assert verdict.is_production_ready


def test_a_capability_with_no_binding_is_not_judged() -> None:
    """An unbound capability is ``plan``'s business; reporting it as a demo binding would mislead."""
    verdict = assess_production_readiness(_registry(), {})
    assert verdict.is_production_ready
    assert verdict.capabilities_involved == ()


def test_demo_usage_reports_the_capability_description() -> None:
    usage = DemoUsage("vision.object_detection.v1", "builtin_detector", "test")
    assert "locate objects" in usage.description


# --------------------------------------------------------------------------- #
# The guard in a real job
# --------------------------------------------------------------------------- #


def _recipe(document: dict[str, object]) -> Recipe:
    return Recipe.model_validate(document)


def _run(session: Session, recipe: Recipe, **options: object):
    return asyncio.run(session.run(JobRequest(recipe=recipe, options=JobOptions(**options))))  # type: ignore[arg-type]


def test_a_job_on_demo_backends_is_refused_before_generating(session: Session, dataset_root: Path) -> None:
    """The refusal must happen in pre-flight, not after paying for candidates nobody may keep."""
    document = make_recipe("./data/source", candidates=2)
    document["acceptance"] = {"allow_demo_backends": False}
    with pytest.raises(ValidationFailure) as error:
        _run(session, _recipe(document))
    assert error.value.code is ErrorCode.DEMO_BACKEND_NOT_ALLOWED
    # Nothing was generated, and no job row was left behind claiming work happened.
    assert session.state.latest_job() is None


def test_a_recipe_may_acknowledge_a_demonstration_dataset(session: Session, dataset_root: Path) -> None:
    document = make_recipe("./data/source", candidates=1)
    document["acceptance"] = {"allow_demo_backends": True}
    outcome = _run(session, _recipe(document))
    assert outcome.manifest.state is JobState.SUCCEEDED
    assert outcome.manifest.counters.generated > 0
    # The manifest records what the dataset was made with, so it can be audited afterwards.
    assert outcome.manifest.demo_backends_allowed is True
    assert outcome.manifest.demo_backends
    assert set(outcome.manifest.demo_backends) <= set(PRODUCTION_CAPABILITIES)


def test_the_manifest_records_which_bindings_were_demonstrations(
    session: Session, dataset_root: Path
) -> None:
    document = make_recipe("./data/source", candidates=1)
    outcome = _run(session, _recipe(document))
    for capability in outcome.manifest.demo_backends:
        assert outcome.manifest.capability_bindings[capability].startswith("builtin_")


def test_export_refuses_a_demo_job_when_the_recipe_disagrees(session: Session, dataset_root: Path) -> None:
    """A profile or recipe edited after the run must not retroactively make an old run exportable."""
    document = make_recipe("./data/source", candidates=1)
    outcome = _run(session, _recipe(document))
    strict = _recipe({**document, "acceptance": {"allow_demo_backends": False}})
    with pytest.raises(ValidationFailure) as error:
        session.export_job(outcome.manifest.job_id, recipe=strict, path_override="./data/strict")
    assert error.value.code is ErrorCode.DEMO_BACKEND_NOT_ALLOWED


def test_plan_reports_a_demonstration_stack_before_anything_runs(
    session: Session, dataset_root: Path
) -> None:
    """`plan` must say a stack is a demonstration stack before anyone pays for candidates."""
    recipe_path = session.workspace.root / "guard-recipe.yaml"
    document = make_recipe("./data/source", candidates=1)
    document["acceptance"] = {"allow_demo_backends": False}
    recipe_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = CliRunner().invoke(
        app, ["plan", str(recipe_path), "--workspace", str(session.workspace.root), "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["production_ready"] is False
    assert payload["demo_backends_allowed"] is False
    assert payload["demo_backends"]
    assert set(payload["demo_backends"]) <= set(PRODUCTION_CAPABILITIES)
    assert set(payload["demo_backends"]) == {
        capability
        for capability, backend in payload["capability_bindings"].items()
        if backend.startswith("builtin_")
    } & set(PRODUCTION_CAPABILITIES)


def test_plan_reports_a_production_stack_as_ready(session: Session, dataset_root: Path) -> None:
    """The same command on a profile with real bindings must not warn."""
    profile = default_profile()
    real = profile.model_copy(
        update={
            "backends": {**profile.backends, **_ready_specs()},
            "bindings": {**profile.bindings, **_candidate_bindings()},
        }
    )
    runtime_path = session.workspace.root / "production-runtime.yaml"
    runtime_path.write_text(
        yaml.safe_dump(real.model_dump(mode="json", exclude={"source_path"}), sort_keys=False),
        encoding="utf-8",
    )
    recipe_path = session.workspace.root / "production-recipe.yaml"
    document = make_recipe("./data/source", candidates=1)
    document["acceptance"] = {"allow_demo_backends": False}
    recipe_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "plan",
            str(recipe_path),
            "--workspace",
            str(session.workspace.root),
            "--runtime",
            str(runtime_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["production_ready"] is True
    assert payload["demo_backends"] == {}


# --------------------------------------------------------------------------- #
# The evidence naming convention
# --------------------------------------------------------------------------- #


def test_file_key_is_filesystem_safe() -> None:
    assert file_key("candidate-0") == "candidate-0"
    for hostile in ("../escape", "sedan/../../etc", "a b", "x\\y"):
        key = file_key(hostile)
        assert "/" not in key and "\\" not in key and ".." not in key
    assert file_key("") == "candidate"
    assert file_key("   ") == "candidate"
    assert file_key("", fallback="o_1") == "o_1"


def test_the_writer_and_readers_agree_on_the_name(tmp_path: Path) -> None:
    """This is the defect the module exists for: two spellings of the same file."""
    written = evidence_path(tmp_path, "annotation", "candidate-0")
    assert written.name == "annotation-candidate-0.json"
    assert written.name in {f"annotation-{stem}.json" for stem in kinds_for("candidate-0")}


def test_matches_kind_recognises_keyed_and_unkeyed_evidence(tmp_path: Path) -> None:
    assert matches_kind(tmp_path / "annotation-candidate-0.json", "annotation")
    assert matches_kind(tmp_path / "annotation.json", "annotation")
    assert not matches_kind(tmp_path / "quality-candidate-0.json", "annotation")
    assert not matches_kind(tmp_path / "annotation-candidate-0.txt", "annotation")


def test_kinds_for_offers_the_keyed_name_first() -> None:
    stems = kinds_for("candidate-0")
    assert stems[0] == "candidate-0"
    assert file_key(stems[0]) in stems


def test_a_resumed_run_keeps_every_candidate(session: Session, dataset_root: Path) -> None:
    """A rerun reads the keyed evidence, so candidates are not silently lost."""
    recipe = _recipe(make_recipe("./data/source", candidates=2))
    first = _run(session, recipe, workers=2)
    assert first.manifest.counters.generated == 20

    second = _run(session, recipe, workers=2)
    assert second.manifest.state is JobState.SUCCEEDED
    assert second.manifest.counters.generated == first.manifest.counters.generated
    assert second.manifest.counters.accepted == first.manifest.counters.accepted
    assert second.manifest.counters.nodes_resumed > 0
