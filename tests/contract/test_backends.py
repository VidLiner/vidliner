"""Backend contract tests.

Every backend is exercised through its protocol, so these tests are the same whether the backend is a
built-in heuristic, the deterministic stand-in, or an adapter for a remote service.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vidliner.capabilities.backend import (
    Backend,
    BackendHealth,
    CompositingRequest,
    DetectionRequest,
    EmbeddingRequest,
    GenerationRequest,
    HarmonizationRequest,
    MaskRefinementRequest,
    PipelineContext,
    SemanticAssessRequest,
    VLMRequest,
)
from vidliner.capabilities.names import (
    CAP_IMAGE_COMPOSITING,
    CAP_IMAGE_HARMONIZATION,
    CAP_INSTANCE_SEGMENTATION,
    CAP_MASK_REFINEMENT,
    CAP_OBJECT_DETECTION,
    CAP_OBJECT_REPLACEMENT,
    CAP_PLANNING,
    CAP_QUALITY_ARTIFACT,
    CAP_QUALITY_BACKGROUND,
    CAP_QUALITY_EMBEDDING,
    CAP_QUALITY_SEMANTIC,
    CAP_SCENE_ANALYSIS,
    KNOWN_CAPABILITIES,
    describe_capability,
    is_known_capability,
)
from vidliner.core.errors import ErrorCode, VidlinerError
from vidliner.domain.enums import ArtifactKind, MediaKind
from vidliner.domain.instances import ObjectInstance
from vidliner.domain.replacement import (
    CandidateSpec,
    GeometryBudget,
    PreservationRules,
    ReplacementIntent,
    ReplacementPlan,
)
from vidliner.domain.scene import SceneContext
from vidliner.domain.shapes import BoundingBox, ImageShape
from vidliner.fixtures import render_scene
from vidliner.runtime.profile import BackendSpec, RuntimeProfile
from vidliner.runtime.registry import BackendRegistry, import_attribute
from vidliner.storage.workspace import ArtifactStore, BackendIO, Workspace


def _profile(**backends: BackendSpec) -> RuntimeProfile:
    from vidliner.runtime.profile import default_profile

    base = default_profile()
    return base.model_copy(update={"backends": {**base.backends, **backends}})


@pytest.fixture(name="io")
def io_fixture(tmp_path: Path) -> BackendIO:
    return BackendIO(ArtifactStore(Workspace(tmp_path / "ws").initialize()))


@pytest.fixture(name="context")
def context_fixture(io: BackendIO) -> PipelineContext:
    """A backend invocation context, carrying artifact access the way the runtime does."""
    return PipelineContext(job_id="j_test", node_id="n_test", seed=17, io=io)


def _image(
    io: BackendIO, *, objects: tuple[tuple[int, int, int, int], ...] = ((30, 20, 90, 80),)
) -> tuple[object, np.ndarray]:
    from vidliner.domain.enums import ArtifactKind

    scene_objects = tuple(
        __import__("vidliner.fixtures", fromlist=["SceneObject"]).SceneObject(
            class_name="car", bbox=box, colour=(200, 40, 40)
        )
        for box in objects
    )
    array = render_scene(width=160, height=120, objects=scene_objects, seed=3)
    from PIL import Image

    reference = io.save_image(Image.fromarray(array, mode="RGB"), kind=ArtifactKind.IMAGE)
    return reference, array


def _mask(io: BackendIO, box: tuple[int, int, int, int] = (30, 20, 90, 80)) -> tuple[object, np.ndarray]:
    mask = np.zeros((120, 160), dtype=bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return io.save_mask(mask), mask


# --------------------------------------------------------------------------- #
# Capability vocabulary
# --------------------------------------------------------------------------- #


def test_every_capability_is_described() -> None:
    for capability in KNOWN_CAPABILITIES:
        assert describe_capability(capability)
        assert is_known_capability(capability)


def test_unknown_capability_helpers() -> None:
    assert not is_known_capability("vision.telepathy.v1")
    with pytest.raises(KeyError):
        describe_capability("vision.telepathy.v1")


def test_builtin_capabilities_need_no_backend() -> None:
    """Background preservation is measured in-house, so no profile may bind it to a backend."""
    from vidliner.capabilities.names import BUILTIN_CAPABILITIES, is_builtin_capability
    from vidliner.runtime.profile import default_profile

    assert CAP_QUALITY_BACKGROUND in BUILTIN_CAPABILITIES
    assert is_builtin_capability(CAP_QUALITY_BACKGROUND)
    assert not is_builtin_capability(CAP_OBJECT_DETECTION)
    assert CAP_QUALITY_BACKGROUND not in default_profile().bindings


def test_default_profile_has_no_dangling_binding() -> None:
    """Every binding must name a backend that actually advertises the capability."""
    from vidliner.runtime.profile import default_profile

    profile = default_profile()
    registry = BackendRegistry(profile)
    for capability, backend in profile.bindings.items():
        declared = registry.declared_capabilities(backend)
        if declared:
            assert capability in declared, f"{backend} is bound to {capability} but does not advertise it"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_import_attribute_supports_both_forms() -> None:
    assert import_attribute("vidliner.capabilities.names:KNOWN_CAPABILITIES")
    assert import_attribute("vidliner.capabilities.names.KNOWN_CAPABILITIES")


def test_import_attribute_reports_a_missing_module() -> None:
    with pytest.raises(VidlinerError) as error:
        import_attribute("vidliner.nope:Thing")
    assert error.value.code is ErrorCode.BACKEND_NOT_FOUND


def test_import_attribute_reports_a_missing_attribute() -> None:
    with pytest.raises(VidlinerError):
        import_attribute("vidliner.capabilities.names:NotAThing")


def test_registry_resolves_the_default_profile() -> None:
    registry = BackendRegistry(_profile())
    report = registry.resolve(
        (
            CAP_OBJECT_DETECTION,
            CAP_INSTANCE_SEGMENTATION,
            CAP_SCENE_ANALYSIS,
            CAP_PLANNING,
            CAP_OBJECT_REPLACEMENT,
            CAP_QUALITY_SEMANTIC,
            CAP_QUALITY_ARTIFACT,
            CAP_QUALITY_EMBEDDING,
            CAP_MASK_REFINEMENT,
            CAP_IMAGE_COMPOSITING,
            CAP_IMAGE_HARMONIZATION,
        )
    )
    assert report.is_complete
    assert report.backend_for(CAP_OBJECT_DETECTION) == "builtin_detector"


def test_registry_reports_an_unbound_capability() -> None:
    profile = _profile().model_copy(update={"bindings": {}})
    registry = BackendRegistry(profile)
    report = registry.resolve(("vision.depth_estimation.v1",))
    assert report.unmet == ("vision.depth_estimation.v1",)
    assert not report.is_complete


def test_registry_refuses_an_ambiguous_capability() -> None:
    from vidliner.capabilities.names import CAP_QUALITY_SEMANTIC
    from vidliner.runtime.profile import BackendSpec

    profile = _profile(
        second_evaluator=BackendSpec(
            use="vidliner.backends.local.evaluator:LocalMetricEvaluatorBackend",
        )
    ).model_copy(update={"bindings": {}})
    registry = BackendRegistry(profile)
    with pytest.raises(VidlinerError) as error:
        registry.resolve((CAP_QUALITY_SEMANTIC,))
    assert error.value.code is ErrorCode.BINDING_AMBIGUOUS


def test_registry_refuses_an_unknown_capability_name() -> None:
    registry = BackendRegistry(_profile())
    with pytest.raises(VidlinerError):
        registry.resolve(("not.a.capability",))


def test_registry_reports_a_disabled_backend() -> None:
    registry = BackendRegistry(_profile())
    with pytest.raises(VidlinerError):
        registry.backend("missing_backend")


async def test_registry_probes_every_backend() -> None:
    registry = BackendRegistry(_profile())
    probes = await registry.probe_all()
    assert set(probes) == set(registry.profile.backend_names())
    assert all(probe.is_ready for probe in probes.values())


async def test_registry_probe_reports_a_broken_backend_import() -> None:
    profile = _profile(broken=BackendSpec(use="vidliner.backends.nope:MissingBackend"))
    probes = await BackendRegistry(profile).probe_all()
    assert probes["broken"].health == BackendHealth.UNAVAILABLE


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #


async def test_detector_finds_the_object(io: BackendIO, context: PipelineContext) -> None:
    registry = BackendRegistry(_profile())
    detector = registry.backend("builtin_detector")
    image, _ = _image(io)
    result = await detector.detect(DetectionRequest(image=image, classes=("car",)), context)
    assert result.instances
    best = max(result.instances, key=lambda instance: instance.bbox.area)
    assert best.class_name == "car"
    assert best.bbox.iou(BoundingBox(x_min=30, y_min=20, x_max=90, y_max=80)) > 0.3


async def test_detector_is_deterministic(io: BackendIO, context: PipelineContext) -> None:
    detector = BackendRegistry(_profile()).backend("builtin_detector")
    image, _ = _image(io)
    first = await detector.detect(DetectionRequest(image=image, classes=("car",)), context)
    second = await detector.detect(DetectionRequest(image=image, classes=("car",)), context)
    assert [instance.object_id for instance in first.instances] == [
        instance.object_id for instance in second.instances
    ]


async def test_detector_refuses_video_input(io: BackendIO, context: PipelineContext) -> None:
    detector = BackendRegistry(_profile()).backend("builtin_detector")
    image, _ = _image(io)
    with pytest.raises(VidlinerError) as error:
        await detector.detect(DetectionRequest(image=image, media_kind=MediaKind.VIDEO), context)
    assert error.value.code is ErrorCode.VIDEO_NOT_SUPPORTED


def test_detector_backend_declares_its_capability() -> None:
    backend = BackendRegistry(_profile()).backend("builtin_detector")
    assert CAP_OBJECT_DETECTION in backend.capabilities
    assert isinstance(backend, Backend)


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


async def test_segmentation_produces_a_mask_covering_the_object(
    io: BackendIO, context: PipelineContext
) -> None:
    segmenter = BackendRegistry(_profile()).backend("builtin_segmenter")
    image, _ = _image(io)
    result = await segmenter.segment(
        SegmentationRequestFactory.create(image, (30.0, 20.0, 90.0, 80.0)),
        context,
    )
    assert result.area_px > 100
    assert result.coverage > 0.02
    mask = io.load_mask(result.mask)
    assert mask[50, 60]
    assert not mask[2, 2]


async def test_segmentation_needs_a_prompt(context: PipelineContext) -> None:
    from vidliner.capabilities.backend import SegmentationRequest

    segmenter = BackendRegistry(_profile()).backend("builtin_segmenter")
    with pytest.raises(VidlinerError):
        await segmenter.segment(
            SegmentationRequest(
                image=SegmentationRequestFactory.artifact(),
                image_shape=ImageShape(width=8, height=8),
            ),
            context,
        )


async def test_segmentation_refuses_a_text_only_prompt(io: BackendIO, context: PipelineContext) -> None:
    from vidliner.capabilities.backend import SegmentationRequest

    segmenter = BackendRegistry(_profile()).backend("builtin_segmenter")
    image, _ = _image(io)
    with pytest.raises(VidlinerError) as error:
        await segmenter.segment(
            SegmentationRequest(image=image, image_shape=ImageShape(width=160, height=120), text="a car"),
            context,
        )
    assert error.value.code is ErrorCode.OPERATOR_NOT_IMPLEMENTED


class SegmentationRequestFactory:
    """Small helper so the segmentation tests read clearly."""

    @staticmethod
    def create(image: object, bbox: tuple[float, float, float, float]) -> object:
        from vidliner.capabilities.backend import SegmentationRequest

        return SegmentationRequest(
            image=image, image_shape=ImageShape(width=160, height=120), bbox=bbox, guidance="bbox"
        )

    @staticmethod
    def artifact() -> object:
        from vidliner.core.results import ArtifactRef

        return ArtifactRef(
            kind=ArtifactKind.IMAGE, digest="a" * 64, media_type="image/png", size_bytes=1, suffix="png"
        )


# --------------------------------------------------------------------------- #
# Scene analysis
# --------------------------------------------------------------------------- #


def _target(io: BackendIO, mask_box: tuple[int, int, int, int] = (30, 20, 90, 80)) -> ObjectInstance:
    from vidliner.core.identity import object_id

    mask_ref, _ = _mask(io, mask_box)
    box = BoundingBox(
        x_min=float(mask_box[0]), y_min=float(mask_box[1]), x_max=float(mask_box[2]), y_max=float(mask_box[3])
    )
    return ObjectInstance(
        object_id=object_id("b" * 64, 0, "car", box, 0.9),
        class_name="car",
        bbox=box,
        score=0.9,
        source_frame=f"{'b' * 64}#0",
    ).with_mask(
        __import__("vidliner.domain.masks", fromlist=["MaskRef"]).MaskRef(
            artifact=mask_ref,
            shape=ImageShape(width=160, height=120),
            area_px=int((mask_box[2] - mask_box[0]) * (mask_box[3] - mask_box[1])),
        )
    )


async def test_scene_analysis_measures_real_facts(io: BackendIO, context: PipelineContext) -> None:
    from vidliner.capabilities.backend import SceneAnalysisRequest

    analyser = BackendRegistry(_profile()).backend("builtin_scene")
    image, array = _image(io)
    target = _target(io)
    mask = io.load_mask(target.mask_ref.artifact)
    luminance = (array.astype(np.float32) / 255.0 * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(
        axis=2
    )
    scene = await analyser.analyse(
        SceneAnalysisRequest(
            image=image,
            image_shape=ImageShape(width=160, height=120),
            target=target,
            mask_array=mask,
            luminance=luminance,
        ),
        context,
    )
    assert isinstance(scene, SceneContext)
    assert scene.object_category == "car"
    assert scene.lighting.intensity is not None
    assert scene.background_summary
    assert scene.scale_class.value in {"tiny", "small", "medium", "large"}


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


def _intent(mode: str = "strict") -> ReplacementIntent:
    return ReplacementIntent(target_object="o_1", replacement_category="sedan", mode=mode)


def _scene(io: BackendIO) -> SceneContext:
    target = _target(io)
    return SceneContext(
        object_category="car",
        object_id=target.object_id,
        image_shape=ImageShape(width=160, height=120),
        bbox=target.bbox,
        mask=target.mask_ref,
        measured_by="test",
    )


async def test_planner_builds_one_spec_per_candidate(io: BackendIO, context: PipelineContext) -> None:
    from vidliner.capabilities.backend import PlanningRequest

    planner = BackendRegistry(_profile()).backend("builtin_planner")
    plan = await planner.plan(
        PlanningRequest(
            sample_id="s_1",
            intent=_intent(),
            scene=_scene(io),
            candidates_per_object=3,
            prompt_template="replace the {source} with {category}",
            negative_constraints=("no text",),
        ),
        context,
    )
    assert isinstance(plan, ReplacementPlan)
    assert plan.candidate_count == 3
    assert len({spec.candidate_key for spec in plan.candidate_specs}) == 3
    assert all(spec.seed for spec in plan.candidate_specs)
    assert "no text" in plan.negative_constraints


async def test_planner_seeds_are_stable_for_the_same_context(io: BackendIO, context: PipelineContext) -> None:
    from vidliner.capabilities.backend import PlanningRequest

    planner = BackendRegistry(_profile()).backend("builtin_planner")
    request = PlanningRequest(
        sample_id="s_1",
        intent=_intent(),
        scene=_scene(io),
        candidates_per_object=2,
        prompt_template="replace the {source} with {category}",
    )
    first = await planner.plan(request, context)
    second = await planner.plan(request, context)
    assert [spec.seed for spec in first.candidate_specs] == [spec.seed for spec in second.candidate_specs]


async def test_creative_plans_widen_the_geometry_budget(io: BackendIO, context: PipelineContext) -> None:
    from vidliner.capabilities.backend import PlanningRequest

    planner = BackendRegistry(_profile()).backend("builtin_planner")
    strict = await planner.plan(
        PlanningRequest(
            sample_id="s",
            intent=_intent("strict"),
            scene=_scene(io),
            candidates_per_object=1,
            prompt_template="{category}",
        ),
        context,
    )
    creative = await planner.plan(
        PlanningRequest(
            sample_id="s",
            intent=_intent("creative"),
            scene=_scene(io),
            candidates_per_object=1,
            prompt_template="{category}",
        ),
        context,
    )
    assert creative.geometry_budget.centroid_shift_max > strict.geometry_budget.centroid_shift_max


# --------------------------------------------------------------------------- #
# Replacement
# --------------------------------------------------------------------------- #


def _generation_request(io: BackendIO) -> GenerationRequest:
    image, _ = _image(io)
    mask, _ = _mask(io)
    return GenerationRequest(
        sample_id="s_1",
        target_object="o_1",
        candidate_key="sedan-1",
        source_image=image,
        target_mask=mask,
        mask_shape=ImageShape(width=160, height=120),
        target_bbox=BoundingBox(x_min=30, y_min=20, x_max=90, y_max=80),
        positive_prompt="replace the car with a sedan",
        negative_constraints=("no text",),
        expected_category="sedan",
        seed=42,
    )


async def test_fake_generator_repaints_the_region(io: BackendIO, context: PipelineContext) -> None:
    """The generator must replace the region's appearance, not leave the source pixels in place."""
    generator = BackendRegistry(_profile()).backend("builtin_replacement")
    request = _generation_request(io)
    outcome = await generator.replace(request, context)
    produced = np.asarray(io.image(outcome.image).convert("RGB"), dtype=np.float32)
    source = np.asarray(io.image(request.source_image).convert("RGB"), dtype=np.float32)
    mask = io.load_mask(request.target_mask)
    # The generated region is a single shaded colour, so its variance is far lower than the source
    # region's textured object.
    inside_variance = float(produced[mask].std())
    source_variance = float(source[mask].std())
    assert inside_variance <= source_variance


async def test_fake_generator_is_seed_deterministic(io: BackendIO, context: PipelineContext) -> None:
    generator = BackendRegistry(_profile()).backend("builtin_replacement")
    request = _generation_request(io)
    first = await generator.replace(request, context)
    second = await generator.replace(request, context)
    assert first.image.digest == second.image.digest


async def test_fake_generator_differs_across_seeds(io: BackendIO, context: PipelineContext) -> None:
    generator = BackendRegistry(_profile()).backend("builtin_replacement")
    request = _generation_request(io)
    first = await generator.replace(request, context)
    second = await generator.replace(request.model_copy(update={"seed": 43}), context)
    assert first.image.digest != second.image.digest


async def test_fake_generator_refuses_an_empty_mask(io: BackendIO, context: PipelineContext) -> None:
    generator = BackendRegistry(_profile()).backend("builtin_replacement")
    empty = np.zeros((120, 160), dtype=bool)
    from vidliner.domain.enums import ArtifactKind

    empty_ref = io.save_mask(empty).with_kind(ArtifactKind.MASK)
    request = _generation_request(io).model_copy(update={"target_mask": empty_ref})
    with pytest.raises(VidlinerError) as error:
        await generator.replace(request, context)
    assert error.value.code is ErrorCode.MASK_EMPTY


async def test_fake_generator_reports_what_it_ignored(io: BackendIO, context: PipelineContext) -> None:
    generator = BackendRegistry(_profile()).backend("builtin_replacement")
    image, _ = _image(io)
    request = _generation_request(io).model_copy(update={"reference_image": image})
    outcome = await generator.replace(request, context)
    assert "reference_image" in outcome.raw_metadata["ignored_parameters"]


# --------------------------------------------------------------------------- #
# Refinement
# --------------------------------------------------------------------------- #


async def test_refiner_cleans_and_blends(io: BackendIO, context: PipelineContext) -> None:
    refiner = BackendRegistry(_profile()).backend("builtin_refiner")
    mask, _ = _mask(io)
    refined = await refiner.refine_mask(
        MaskRefinementRequest(mask=mask, shape=ImageShape(width=160, height=120), feather_px=2, close_px=3),
        context,
    )
    assert refined.area_px > 0
    assert refined.mask.kind is ArtifactKind.MASK

    source, _ = _image(io)
    generated, _ = _image(io, objects=())
    result = await refiner.composite(
        CompositingRequest(
            source_image=source,
            candidate_image=generated,
            mask=refined.mask,
            shape=ImageShape(width=160, height=120),
            feather_px=2,
        ),
        context,
    )
    assert result.changed_ratio_outside_mask == 0.0


async def test_refiner_harmonises_colour(io: BackendIO, context: PipelineContext) -> None:
    refiner = BackendRegistry(_profile()).backend("builtin_refiner")
    image, _ = _image(io)
    mask, _ = _mask(io)
    result = await refiner.harmonize(
        HarmonizationRequest(image=image, mask=mask, shape=ImageShape(width=160, height=120), strength=0.5),
        context,
    )
    assert result.media_type == "image/png"


async def test_refiner_rejects_a_mismatched_mask(io: BackendIO, context: PipelineContext) -> None:
    from vidliner.capabilities.backend import MaskRefinementRequest as Request

    refiner = BackendRegistry(_profile()).backend("builtin_refiner")
    mask, _ = _mask(io)
    with pytest.raises(VidlinerError) as error:
        await refiner.refine_mask(
            Request(mask=mask, shape=ImageShape(width=64, height=64), feather_px=1), context
        )
    assert error.value.code is ErrorCode.MASK_SHAPE_MISMATCH


# --------------------------------------------------------------------------- #
# Evaluators
# --------------------------------------------------------------------------- #


async def test_semantic_evaluator_scores_a_region(io: BackendIO, context: PipelineContext) -> None:
    evaluator = BackendRegistry(_profile()).backend("builtin_metrics")
    image, _ = _image(io)
    mask, _ = _mask(io)
    assessment = await evaluator.assess_semantics(
        SemanticAssessRequest(
            image=image,
            mask=mask,
            bbox=(30.0, 20.0, 90.0, 80.0),
            expected_category="sedan",
        ),
        context,
    )
    assert 0.0 <= assessment.score <= 1.0
    assert assessment.evaluator_id
    assert assessment.detail["method"]


async def test_semantic_evaluator_flags_a_mismatch(io: BackendIO, context: PipelineContext) -> None:
    evaluator = BackendRegistry(_profile()).backend("builtin_metrics")
    image, _ = _image(io)
    mask, _ = _mask(io)
    assessment = await evaluator.assess_semantics(
        SemanticAssessRequest(
            image=image,
            mask=mask,
            bbox=(30.0, 20.0, 90.0, 80.0),
            expected_category="tree",
        ),
        context,
    )
    assert assessment.score < 0.9


async def test_artifact_evaluator_returns_a_structured_report(
    io: BackendIO, context: PipelineContext
) -> None:
    evaluator = BackendRegistry(_profile()).backend("builtin_metrics")
    image, _ = _image(io)
    mask, _ = _mask(io)
    response = await evaluator.evaluate(VLMRequest(image=image, mask=mask, question="artifacts?"), context)
    assert response.answer == "artifact_report"
    assert "artifact_score" in response.fields
    assert response.fields["duplicate_object"] is False


async def test_artifact_evaluator_handles_a_missing_mask(io: BackendIO, context: PipelineContext) -> None:
    evaluator = BackendRegistry(_profile()).backend("builtin_metrics")
    image, _ = _image(io)
    response = await evaluator.evaluate(VLMRequest(image=image, question="artifacts?"), context)
    assert response.fields["boundary_break"] is True


async def test_embedding_backend_produces_a_comparable_vector(
    io: BackendIO, context: PipelineContext
) -> None:
    backend = BackendRegistry(_profile()).backend("builtin_phash")
    image, _ = _image(io)
    first = await backend.embed(EmbeddingRequest(image=image), context)
    second = await backend.embed(EmbeddingRequest(image=image), context)
    assert first.dimension == 64
    assert first.cosine(second) == pytest.approx(1.0)


async def test_embedding_backend_hash_hex_matches_the_fingerprint(
    io: BackendIO, context: PipelineContext
) -> None:
    from vidliner.quality.fingerprint import perceptual_hash

    backend = BackendRegistry(_profile()).backend("builtin_phash")
    image, array = _image(io)
    reference = await backend.embed(EmbeddingRequest(image=image), context)
    bits = "".join("1" if value > 0.5 else "0" for value in reference.vector)
    assert f"{int(bits, 2):016x}" == perceptual_hash(array.astype(np.float32) / 255.0).hex


def test_embedding_cosine_refuses_a_dimension_mismatch() -> None:
    from vidliner.capabilities.backend import EmbeddingResult

    with pytest.raises(ValueError):
        EmbeddingResult(vector=(1.0, 2.0), dimension=2).cosine(EmbeddingResult(vector=(1.0,), dimension=1))


# --------------------------------------------------------------------------- #
# Profile and credentials
# --------------------------------------------------------------------------- #


def test_profile_rejects_an_unknown_capability_binding() -> None:
    from pydantic import ValidationError

    from vidliner.runtime.profile import RuntimeProfile

    with pytest.raises(ValidationError, match="unknown capability"):
        RuntimeProfile(profile="x", backends={}, bindings={"nope": "nothing"})


def test_profile_rejects_a_binding_to_an_unconfigured_backend() -> None:
    from pydantic import ValidationError

    from vidliner.runtime.profile import RuntimeProfile

    with pytest.raises(ValidationError, match="unconfigured backend"):
        RuntimeProfile(
            profile="x",
            backends={},
            bindings={"vision.object_detection.v1": "missing"},
        )


def test_profile_rejects_an_inline_credential_inside_a_repository(tmp_path: Path) -> None:
    from vidliner.core.errors import ValidationFailure
    from vidliner.runtime.profile import CredentialRef, profile_from_mapping

    (tmp_path / ".git").mkdir()
    document = {
        "profile": "x",
        "backends": {
            "svc": {
                "use": "vidliner.backends.http_replacement:HttpReplacementBackend",
                "options": {"endpoint": "http://localhost:1/edit"},
                "credentials": {"api_key": {"source": "value", "name": "secret"}},
            }
        },
        "bindings": {},
    }
    with pytest.raises(ValidationFailure):
        profile_from_mapping(document, source_path=tmp_path / "runtime.yaml")
    del CredentialRef


def test_credential_resolver_reads_env_and_redacts(monkeypatch: pytest.MonkeyPatch) -> None:
    from vidliner.core.errors import ValidationFailure
    from vidliner.runtime.profile import CredentialRef
    from vidliner.runtime.secrets import REDACTED, CredentialResolver

    monkeypatch.setenv("VIDLINER_TEST_KEY", "super-secret-value")
    resolver = CredentialResolver(environ={"VIDLINER_TEST_KEY": "super-secret-value"})
    isolated = CredentialResolver(environ={})
    reference = CredentialRef(source="env", name="VIDLINER_TEST_KEY")
    assert resolver.resolve(reference, backend="b", name="api_key") == "super-secret-value"
    assert resolver.redact("token super-secret-value here") == f"token {REDACTED} here"
    assert resolver.describe(reference)["name"] == "VIDLINER_TEST_KEY"

    missing = CredentialRef(source="env", name="VIDLINER_ABSENT_KEY")
    with pytest.raises(ValidationFailure) as error:
        isolated.resolve(missing, backend="b", name="api_key")
    assert error.value.code is ErrorCode.CREDENTIAL_MISSING
    assert (
        isolated.resolve(
            CredentialRef(source="env", name="VIDLINER_ABSENT_KEY", required=False), backend="b", name="k"
        )
        is None
    )


def test_redact_profile_removes_inline_values() -> None:
    from vidliner.runtime.profile import BackendSpec, CredentialRef, RuntimeProfile, redact_profile

    profile = RuntimeProfile(
        profile="x",
        backends={
            "svc": BackendSpec(
                use="vidliner.backends.http_replacement:HttpReplacementBackend",
                credentials={"api_key": CredentialRef(source="value", name="secret")},
            )
        },
    )
    snapshot = redact_profile(profile)
    assert snapshot["backends"]["svc"]["credentials"]["api_key"]["name"] == "<redacted>"
    assert "secret" not in str(snapshot)


def test_backend_versions_and_specs() -> None:
    spec = BackendSpec(use="vidliner.backends.base:LocalBackend", options={"a": 1})
    assert spec.enabled
    assert spec.capacity.limit == 2


def test_scene_context_and_plan_models_validate() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="duplicate candidate keys"):
        ReplacementPlan(
            plan_id="p",
            sample_id="s",
            intent=_intent(),
            expected_category="sedan",
            positive_prompt="x",
            candidate_specs=(
                CandidateSpec(candidate_key="a", category="sedan", description="x", seed=1),
                CandidateSpec(candidate_key="a", category="sedan", description="x", seed=2),
            ),
            plan_seed=1,
            planner_id="test",
        )
    assert PreservationRules().background == "strict"
    assert GeometryBudget().area_ratio_max == 1.45
