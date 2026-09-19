"""Contracts shared by the video augmentation and verification layers."""

from __future__ import annotations

from vidliner.domain.shapes import BoundingBox
from vidliner.domain.video import (
    GeometryTransform,
    LabelTransform,
    TimeSpan,
    TimeTransform,
    VideoAugmentSpec,
    VideoOperatorSpec,
    WordStamp,
    build_contrast_pairs,
    compose_transforms,
    enumerate_video_variants,
    project_box,
    project_time_span,
    project_words,
)


def test_geometry_and_placement_project_normalised_boxes() -> None:
    transform = LabelTransform(
        geometry=GeometryTransform(a=2.0, d=1.0, tx=-0.25),
    )
    projected = project_box(transform, BoundingBox(x_min=0.25, y_min=0.2, x_max=0.5, y_max=0.4))
    assert projected.x_min == 0.25
    assert projected.x_max == 0.75
    assert projected.y_min == 0.2
    assert projected.y_max == 0.4


def test_time_and_word_projection_clip_to_variant_clock() -> None:
    transform = LabelTransform(time=TimeTransform(scale=1.0, offset=-1.0))
    words = project_words(
        transform,
        (
            WordStamp(word="before", t0=0.0, t1=0.5),
            WordStamp(word="inside", t0=1.5, t1=2.5),
        ),
        duration_s=1.0,
    )
    assert [word.word for word in words] == ["inside"]
    assert words[0].t0 == 0.5
    assert words[0].t1 == 1.0
    span = project_time_span(transform, TimeSpan(t0=2.0, t1=3.0))
    assert span.t0 == 1.0 and span.t1 == 2.0


def test_transform_composition_preserves_time_order_and_semantics() -> None:
    first = LabelTransform(time=TimeTransform(scale=2.0, offset=1.0))
    second = LabelTransform(time=TimeTransform(scale=0.5, offset=-2.0), semantics="edited")
    composed = compose_transforms(first, second)
    assert composed.time.scale == 1.0
    assert composed.time.offset == -1.5
    assert composed.semantics == "edited"


def test_paired_sampling_is_deterministic_and_has_identity_reference() -> None:
    spec = VideoAugmentSpec(
        id="street",
        seed=7,
        strategy="paired",
        operators=(
            VideoOperatorSpec(name="format.reframe", axis="format", domains={"aspect": ("1:1", "16:9")}),
            VideoOperatorSpec(name="appearance.grade", axis="style", domains={"saturation": (0.8, 1.2)}),
        ),
    )
    first = enumerate_video_variants(spec)
    second = enumerate_video_variants(spec)
    assert [variant.id for variant in first] == [variant.id for variant in second]
    assert first[0].ops == ()
    assert sum("format.reframe" in variant.fingerprint.get("format", "") for variant in first) == 2
    assert sum("appearance.grade" in variant.fingerprint.get("style", "") for variant in first) == 2
    pairs = build_contrast_pairs(first)
    assert len(pairs) == 6
    assert {pair.relation for pair in pairs} == {
        "identity-invariance",
        "format-invariance",
        "style-invariance",
    }
