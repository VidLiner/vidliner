"""Canonical serialization and identity tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from vidliner.core.canonical import canonical_json, digest_bytes, digest_file, digest_json, object_digest
from vidliner.core.identity import (
    candidate_id,
    job_id,
    lineage_key_candidate,
    node_id,
    object_id,
    sample_id,
)
from vidliner.core.seedtree import SeedTree, derive_seed
from vidliner.domain.enums import ArtifactKind
from vidliner.domain.shapes import BoundingBox


def test_canonical_json_sorts_keys_and_drops_none() -> None:
    assert canonical_json({"b": 1, "a": None, "c": {"z": 2, "y": 3}}) == '{"b":1,"c":{"y":3,"z":2}}'


def test_canonical_json_normalises_negative_zero_and_enums() -> None:
    from vidliner.domain.enums import MetricName

    assert canonical_json({"value": -0.0}) == '{"value":0.0}'
    assert canonical_json({"metric": MetricName.SEMANTIC_MATCH}) == '{"metric":"semantic_match"}'


def test_canonical_json_renders_utc_timestamps() -> None:
    stamp = datetime(2024, 5, 17, 12, 0, 0, tzinfo=UTC)
    assert canonical_json({"at": stamp}) == '{"at":"2024-05-17T12:00:00Z"}'


def test_canonical_json_refuses_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        canonical_json({"value": float("nan")})


def test_canonical_json_refuses_unsupported_types() -> None:
    with pytest.raises(TypeError, match="not canonically serializable"):
        canonical_json({"value": object()})


def test_canonical_json_accepts_paths_and_sets() -> None:
    assert canonical_json({"path": Path("a/b"), "set": {"b", "a"}}) == '{"path":"a/b","set":["a","b"]}'


def test_digest_json_is_stable_under_key_order() -> None:
    assert digest_json({"a": 1, "b": 2}) == digest_json({"b": 2, "a": 1})


def test_digest_file_matches_digest_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"vidliner")
    assert digest_file(path) == digest_bytes(b"vidliner")


def test_object_digest_prefixes_and_truncates() -> None:
    value = object_digest("x_", {"a": 1}, 8)
    assert value.startswith("x_")
    assert len(value) == 10


def test_object_digest_rejects_absurd_lengths() -> None:
    with pytest.raises(ValueError):
        object_digest("x_", {}, 2)


def test_sample_id_depends_on_root_and_path() -> None:
    first = sample_id("a" * 64, "images/one.png")
    assert first == sample_id("a" * 64, "images/one.png")
    assert first != sample_id("a" * 64, "images/two.png")
    assert first != sample_id("b" * 64, "images/one.png")


def test_sample_id_normalises_path_separators() -> None:
    assert sample_id("a" * 64, "images\\one.png") == sample_id("a" * 64, "images/one.png")


def test_object_id_quantises_box_and_score() -> None:
    box = BoundingBox(x_min=10.0, y_min=20.0, x_max=60.0, y_max=80.0)
    nudged = BoundingBox(x_min=10.4, y_min=20.4, x_max=60.4, y_max=80.4)
    assert object_id("a" * 64, 0, "car", box, 0.80) == object_id("a" * 64, 0, "car", nudged, 0.81)
    assert object_id("a" * 64, 0, "car", box, 0.80) != object_id("a" * 64, 0, "car", box, 0.40)


def test_object_id_changes_with_class_and_frame() -> None:
    box = BoundingBox(x_min=10.0, y_min=20.0, x_max=60.0, y_max=80.0)
    base = object_id("a" * 64, 0, "car", box, 0.9)
    assert base != object_id("a" * 64, 1, "car", box, 0.9)
    assert base != object_id("a" * 64, 0, "van", box, 0.9)


def test_candidate_id_is_stable_and_specific() -> None:
    first = candidate_id("s_1", "o_1", "sedan-1")
    assert first == candidate_id("s_1", "o_1", "sedan-1")
    assert first != candidate_id("s_1", "o_1", "sedan-2")
    assert first != candidate_id("s_1", "o_2", "sedan-1")


def test_node_id_is_stable_and_rejects_negative_ordinal() -> None:
    assert node_id("detect.objects", "sample:s_1") == node_id("detect.objects", "sample:s_1")
    with pytest.raises(ValueError):
        node_id("detect.objects", "sample:s_1", -1)


def test_lineage_keys_nest() -> None:
    sample = lineage_key_candidate("s_1", "o_1", "sedan-1")
    assert sample.startswith("sample:s_1/object:o_1/candidate:sedan-1")


def test_job_id_depends_on_marker_so_two_attempts_differ() -> None:
    assert job_id("hash", 1, "2024-01-01T00:00Z") != job_id("hash", 1, "2024-01-01T00:01Z")
    assert job_id("hash", 1, "marker") == job_id("hash", 1, "marker")


def test_derive_seed_is_deterministic_and_label_sensitive() -> None:
    assert derive_seed(7, "sample", "s_1") == derive_seed(7, "sample", "s_1")
    assert derive_seed(7, "sample", "s_1") != derive_seed(7, "sample", "s_2")
    assert derive_seed(7, "a", "b") != derive_seed(7, "ab")


def test_derive_seed_rejects_out_of_range_parent() -> None:
    with pytest.raises(ValueError):
        derive_seed(-1, "x")


def test_seed_tree_path_is_recorded() -> None:
    tree = SeedTree(11).for_candidate("s_1", "o_1", "sedan-1")
    assert tree.path == ("sample", "s_1", "object", "o_1", "candidate", "sedan-1")
    assert tree.path_label.endswith("candidate/sedan-1")


def test_seed_tree_candidate_seeds_are_independent_of_neighbours() -> None:
    """Adding a candidate must not shift the seeds of the existing ones."""
    root = SeedTree(99)
    first = root.for_candidate("s_1", "o_1", "sedan-1").seed
    root.for_candidate("s_1", "o_1", "suv-2")
    assert root.for_candidate("s_1", "o_1", "sedan-1").seed == first


def test_seed_tree_node_seed_differs_from_candidate_seed() -> None:
    tree = SeedTree(5).for_candidate("s", "o", "c")
    assert tree.for_node("n_1").seed != tree.seed


def test_artifact_digest_key_namespaces_by_kind() -> None:
    from vidliner.core.identity import artifact_digest_key

    assert artifact_digest_key(ArtifactKind.MASK, "abc") == "mask:abc"
    assert artifact_digest_key(ArtifactKind.IMAGE, "abc") != artifact_digest_key(ArtifactKind.MASK, "abc")
