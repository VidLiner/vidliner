"""Domain model tests: shapes, media probing, masks, instances, annotations, recipes, provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vidliner.domain.annotations import AnnotationBundle, AnnotationObject, CategorySpec, ExportProfile
from vidliner.domain.enums import ArtifactKind, MediaKind, MetricName, Severity
from vidliner.domain.masks import BinaryMask, MaskRef, PolygonMask
from vidliner.domain.media import inspect_media, probe_image_shape
from vidliner.domain.provenance import find_secret_keys
from vidliner.domain.quality import AcceptancePolicy, GateSpec, MetricOutcome
from vidliner.domain.recipe import Recipe
from vidliner.domain.shapes import BoundingBox, ImageShape, Point
from vidliner.fixtures import build_dataset, render_scene

# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #


def test_bounding_box_rejects_inverted_edges() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x_min=10, y_min=10, x_max=5, y_max=20)


def test_bounding_box_geometry() -> None:
    box = BoundingBox(x_min=10, y_min=20, x_max=60, y_max=80)
    assert box.width == 50
    assert box.height == 60
    assert box.area == 3000
    assert box.centroid == Point(x=35, y=50)
    assert box.as_xywh() == (10, 20, 50, 60)


def test_bounding_box_iou_and_clipping() -> None:
    first = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
    second = BoundingBox(x_min=5, y_min=5, x_max=15, y_max=15)
    assert first.iou(second) == pytest.approx(25 / 175)
    clipped = BoundingBox(x_min=-5, y_min=-5, x_max=20, y_max=20).clipped(ImageShape(width=10, height=10))
    assert clipped.as_xyxy() == (0, 0, 10, 10)


def test_bounding_box_normalised_and_scaled() -> None:
    box = BoundingBox(x_min=5, y_min=5, x_max=15, y_max=15)
    assert box.normalized(ImageShape(width=20, height=20)) == (0.25, 0.25, 0.5, 0.5)
    grown = box.expanded(5)
    assert grown.as_xyxy() == (0, 0, 20, 20)


def test_image_shape_helpers() -> None:
    shape = ImageShape(width=100, height=50)
    assert shape.pixel_count == 5000
    assert shape.aspect_ratio == 2.0
    assert shape.diagonal > 111


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #


def test_probe_png_and_jpeg_dimensions(tmp_path: Path) -> None:
    from PIL import Image

    Image.fromarray(render_scene(width=64, height=48)).save(tmp_path / "a.png")
    Image.fromarray(render_scene(width=32, height=24)).save(tmp_path / "b.jpg")
    assert probe_image_shape(tmp_path / "a.png") == ImageShape(width=64, height=48)
    assert probe_image_shape(tmp_path / "b.jpg") == ImageShape(width=32, height=24)


def test_probe_webp_dimensions(tmp_path: Path) -> None:
    from PIL import Image

    Image.fromarray(render_scene(width=40, height=30)).save(tmp_path / "c.webp")
    assert probe_image_shape(tmp_path / "c.webp") == ImageShape(width=40, height=30)


def test_probe_rejects_unsupported_content(tmp_path: Path) -> None:
    from vidliner.core.errors import ValidationFailure

    path = tmp_path / "not-an-image.png"
    path.write_bytes(b"definitely not a png")
    with pytest.raises(ValidationFailure):
        probe_image_shape(path)


def test_inspect_media_builds_an_immutable_asset(tmp_path: Path) -> None:
    from PIL import Image

    Image.fromarray(render_scene(width=20, height=20)).save(tmp_path / "x.png")
    asset = inspect_media(tmp_path / "x.png", relative_to=tmp_path)
    assert asset.path == "x.png"
    assert asset.kind is MediaKind.IMAGE
    assert asset.media_type == "image/png"
    assert asset.shape == ImageShape(width=20, height=20)
    assert len(asset.digest) == 64


def test_inspect_media_refuses_unknown_extension(tmp_path: Path) -> None:
    from vidliner.core.errors import ValidationFailure

    (tmp_path / "notes.txt").write_text("hello")
    with pytest.raises(ValidationFailure):
        inspect_media(tmp_path / "notes.txt")


# --------------------------------------------------------------------------- #
# Masks and instances
# --------------------------------------------------------------------------- #


def _mask_ref(area: int = 100) -> MaskRef:
    from vidliner.core.results import ArtifactRef

    return MaskRef(
        artifact=ArtifactRef(
            kind=ArtifactKind.MASK,
            digest="a" * 64,
            media_type="image/png",
            size_bytes=10,
            width=100,
            height=100,
            suffix="png",
        ),
        shape=ImageShape(width=100, height=100),
        area_px=area,
    )


def test_mask_ref_computes_coverage() -> None:
    assert _mask_ref(2500).coverage == pytest.approx(0.25)
    assert _mask_ref(0).is_empty


def test_mask_ref_requires_a_mask_artifact() -> None:
    from vidliner.core.errors import ValidationFailure
    from vidliner.core.results import ArtifactRef

    with pytest.raises(ValidationError):
        MaskRef(
            artifact=ArtifactRef(
                kind=ArtifactKind.IMAGE,
                digest="a" * 64,
                media_type="image/png",
                size_bytes=1,
            ),
            shape=ImageShape(width=1, height=1),
            area_px=1,
        )
    del ValidationFailure


def test_binary_mask_rejects_impossible_area() -> None:
    with pytest.raises(ValidationError):
        BinaryMask(shape=ImageShape(width=10, height=10), area_px=101)


def test_polygon_roundtrips_through_coco_layout() -> None:
    polygon = PolygonMask(rings=[[Point(x=1, y=2), Point(x=5, y=2), Point(x=5, y=6), Point(x=1, y=6)]])
    flat = polygon.as_coco_flat()
    assert flat == [1.0, 2.0, 5.0, 2.0, 5.0, 6.0, 1.0, 6.0]
    assert PolygonMask.from_coco_flat(flat) == polygon


def test_polygon_requires_three_points() -> None:
    with pytest.raises(ValidationError):
        PolygonMask(rings=[[Point(x=0, y=0), Point(x=1, y=1)]])


def test_object_instance_identity_and_mask_attachment() -> None:
    from vidliner.domain.instances import ObjectInstance

    instance = ObjectInstance(
        object_id="o_1",
        class_name="car",
        bbox=BoundingBox(x_min=1, y_min=1, x_max=11, y_max=11),
        score=0.9,
        source_frame=f"{'a' * 64}#0",
    )
    assert instance.frame_index == 0
    assert instance.asset_digest == "a" * 64
    assert not instance.is_segmented
    masked = instance.with_mask(_mask_ref(50))
    assert masked.is_segmented
    assert masked.area_px == 50
    assert masked.coverage == pytest.approx(0.005)


def test_object_instance_rejects_degenerate_box() -> None:
    from vidliner.domain.instances import ObjectInstance

    with pytest.raises(ValidationError):
        ObjectInstance(
            object_id="o_1",
            class_name="car",
            bbox=BoundingBox(x_min=5, y_min=5, x_max=5, y_max=5),
            score=0.9,
        )


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #


def _bundle() -> AnnotationBundle:
    return AnnotationBundle(
        sample_id="s_1",
        image_shape=ImageShape(width=100, height=100),
        categories=(CategorySpec(category_id=0, name="car"),),
        objects=(
            AnnotationObject(
                object_id="o_1",
                category_id=0,
                bbox=BoundingBox(x_min=10, y_min=10, x_max=50, y_max=50),
            ),
        ),
    )


def test_annotation_bundle_rejects_undeclared_category() -> None:
    with pytest.raises(ValidationError):
        AnnotationBundle(
            sample_id="s",
            image_shape=ImageShape(width=10, height=10),
            categories=(CategorySpec(category_id=1, name="car"),),
            objects=(
                AnnotationObject(
                    object_id="o",
                    category_id=99,
                    bbox=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
                ),
            ),
        )


def test_annotation_bundle_category_lookup_and_replacement() -> None:
    bundle = _bundle()
    assert bundle.category_id("car") == 0
    assert bundle.category_name(0) == "car"
    assert bundle.object_count == 1
    replacement = AnnotationObject(
        object_id="o_1",
        category_id=0,
        bbox=BoundingBox(x_min=20, y_min=20, x_max=60, y_max=60),
    )
    updated = bundle.replace_object("o_1", replacement)
    assert updated.objects[0].bbox.x_min == 20


def test_annotation_bundle_ensure_categories_appends_ids() -> None:
    bundle = _bundle().ensure_categories(("sedan", "suv"))
    names = [category.name for category in bundle.categories]
    assert names == ["car", "sedan", "suv"]
    assert [category.category_id for category in bundle.categories] == [0, 1, 2]


def test_annotation_validation_reports_out_of_bounds() -> None:
    bundle = _bundle().with_objects(
        (
            AnnotationObject(
                object_id="o_1",
                category_id=0,
                bbox=BoundingBox(x_min=0, y_min=0, x_max=200, y_max=50),
            ),
        )
    )
    problems = bundle.validate_against_image()
    assert any("outside the image" in problem for problem in problems)


def test_export_profile_rejects_unknown_format() -> None:
    with pytest.raises(ValidationError):
        ExportProfile(format="pascal-voc")


# --------------------------------------------------------------------------- #
# Recipe
# --------------------------------------------------------------------------- #


def _recipe_document(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "recipe": "test",
        "dataset": {"input": "./data"},
        "target": {"classes": ["car"]},
        "replacement": {"values": ["sedan"]},
    }
    document.update(overrides)
    return document


def test_recipe_requires_replacement_values_for_category_strategy() -> None:
    with pytest.raises(ValidationError):
        Recipe.model_validate(_recipe_document(replacement={"strategy": "category", "values": []}))


def test_recipe_refuses_test_augmentation_without_opt_in() -> None:
    with pytest.raises(ValidationError, match="refused by default"):
        Recipe.model_validate(
            _recipe_document(
                dataset={"input": "./data", "augment_splits": ["test"], "splits": {"mode": "directory"}}
            )
        )


def test_recipe_refuses_test_augmentation_in_strict_mode_even_with_opt_in() -> None:
    with pytest.raises(ValidationError, match="strict replacement mode"):
        Recipe.model_validate(
            _recipe_document(
                dataset={
                    "input": "./data",
                    "augment_splits": ["test"],
                    "allow_test_augmentation": True,
                    "splits": {"mode": "directory"},
                }
            )
        )


def test_recipe_hash_ignores_estimates() -> None:
    first = Recipe.model_validate(_recipe_document())
    second = Recipe.model_validate(_recipe_document(estimates={"generation_unit_cost": 9.0}))
    assert first.recipe_hash() == second.recipe_hash()
    assert first.snapshot() != second.snapshot()


def test_recipe_hash_changes_with_intent() -> None:
    first = Recipe.model_validate(_recipe_document())
    second = Recipe.model_validate(_recipe_document(replacement={"values": ["suv"]}))
    assert first.recipe_hash() != second.recipe_hash()


def test_recipe_required_capabilities_include_refinement_steps() -> None:
    recipe = Recipe.model_validate(_recipe_document())
    capabilities = recipe.required_capabilities()
    assert "vision.object_detection.v1" in capabilities
    assert "generation.object_replacement.v1" in capabilities


def test_recipe_can_carry_a_video_variant_plan() -> None:
    recipe = Recipe.model_validate(
        _recipe_document(
            video={
                "id": "street",
                "seed": 17,
                "strategy": "paired",
                "operators": [
                    {
                        "name": "format.reframe",
                        "axis": "format",
                        "domains": {"aspect": ["1:1", "16:9"]},
                    }
                ],
            }
        )
    )
    assert recipe.video is not None
    assert "vision.object_tracking.v1" in recipe.required_capabilities()
    assert "generation.video_replacement.v1" not in recipe.required_capabilities()


def test_recipe_rejects_unknown_quality_metric() -> None:
    with pytest.raises(ValidationError, match="unknown quality metric"):
        Recipe.model_validate(_recipe_document(quality={"hard_gates": {"made_up": 0.5}}))


def test_recipe_rejects_metric_in_both_gate_kinds() -> None:
    with pytest.raises(ValidationError, match="both hard and warn"):
        Recipe.model_validate(
            _recipe_document(
                quality={"hard_gates": {"semantic_match": 0.9}, "warn_gates": {"semantic_match": 0.5}}
            )
        )


def test_recipe_deduplicates_and_sorts_classes() -> None:
    recipe = Recipe.model_validate(_recipe_document(target={"classes": ["van", "car", "car"]}))
    assert recipe.target.classes == ("car", "van")


def test_recipe_prompt_template_must_use_a_placeholder() -> None:
    with pytest.raises(ValidationError, match="prompt_template"):
        Recipe.model_validate(_recipe_document(replacement={"values": ["sedan"], "prompt_template": "fixed"}))


# --------------------------------------------------------------------------- #
# Quality policy models
# --------------------------------------------------------------------------- #


def test_policy_rejects_duplicate_gates() -> None:
    with pytest.raises(ValidationError):
        AcceptancePolicy(
            gates=(
                GateSpec(metric=MetricName.SEMANTIC_MATCH, threshold=0.9),
                GateSpec(metric=MetricName.SEMANTIC_MATCH, threshold=0.8, severity=Severity.WARN),
            )
        )


def test_policy_overall_renormalises_weights_over_measured_metrics() -> None:
    policy = AcceptancePolicy(minimum_overall=0.5)
    outcomes = [
        MetricOutcome(metric=MetricName.SEMANTIC_MATCH, value=0.8),
        MetricOutcome(metric=MetricName.GEOMETRY, value=1.0),
    ]
    expected = (0.24 * 0.8 + 0.12 * 1.0) / (0.24 + 0.12)
    assert policy.overall_from(outcomes) == pytest.approx(expected)


def test_policy_policy_hash_is_stable() -> None:
    first = AcceptancePolicy(minimum_overall=0.8)
    second = AcceptancePolicy(minimum_overall=0.8)
    assert first.policy_hash == second.policy_hash
    assert first.policy_hash != AcceptancePolicy(minimum_overall=0.9).policy_hash


def test_gate_margin_and_passes() -> None:
    gate = GateSpec(metric=MetricName.BACKGROUND_PRESERVATION, threshold=0.9)
    assert gate.passes(0.95)
    assert not gate.passes(0.85)
    assert gate.margin(0.92) == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# Provenance secret detection
# --------------------------------------------------------------------------- #


def test_find_secret_keys_flags_likely_credentials() -> None:
    payload = {"api_key": "secret-value", "nested": {"authorization": "Bearer x"}}
    found = find_secret_keys(payload)
    assert "api_key" in found
    assert "nested.authorization" in found


def test_find_secret_keys_allows_references_and_ordinary_keys() -> None:
    payload = {
        "api_key": "env:VIDLINER_KEY",
        "candidate_key": "sedan-1",
        "keys": ["a", "b"],
        "keyframe": 3,
        "credential_ref": "env:X",
    }
    assert find_secret_keys(payload) == set()


def test_find_secret_keys_ignores_null_values() -> None:
    assert find_secret_keys({"api_key": None}) == set()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def test_synthetic_dataset_writes_images_and_annotations(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path / "data", count=3, annotations=True)
    document = json.loads((tmp_path / "data" / "annotations" / "instances.json").read_text())
    assert dataset.count == 3
    assert len(document["images"]) == 3
    assert len(document["annotations"]) == 3
    assert document["categories"][0]["name"] == "car"
