"""Annotation conversion tests, including roundtrips through both supported families."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidliner.annotations import (
    CocoReader,
    CocoWriter,
    YoloReader,
    YoloWriter,
    read_annotations,
    write_annotations,
)
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.annotations import AnnotationBundle, AnnotationObject, CategorySpec, ExportProfile
from vidliner.domain.masks import PolygonMask
from vidliner.domain.shapes import BoundingBox, ImageShape, Point
from vidliner.fixtures import build_dataset, write_coco, write_yolo


def _bundle(sample_id: str = "img_000.png") -> AnnotationBundle:
    polygon = PolygonMask(
        rings=[[Point(x=10, y=10), Point(x=50, y=10), Point(x=50, y=40), Point(x=10, y=40)]]
    )
    return AnnotationBundle(
        sample_id=sample_id,
        image_shape=ImageShape(width=64, height=48),
        categories=(CategorySpec(category_id=0, name="car"), CategorySpec(category_id=1, name="van")),
        objects=(
            AnnotationObject(
                object_id="car_0000",
                category_id=0,
                bbox=BoundingBox(x_min=10, y_min=10, x_max=50, y_max=40),
                polygon=polygon,
                score=0.9,
                attributes={"source_class": "car"},
            ),
            AnnotationObject(
                object_id="van_0001",
                category_id=1,
                bbox=BoundingBox(x_min=2, y_min=2, x_max=18, y_max=20),
                iscrowd=True,
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# COCO
# --------------------------------------------------------------------------- #


def test_coco_reader_reads_the_fixture_dataset(tmp_path: Path) -> None:
    build_dataset(tmp_path / "data", count=3, annotations=True)
    reader = CocoReader(tmp_path / "data")
    categories = reader.categories()
    assert [category.name for category in categories] == ["car"]
    records = reader.read_image("img_000.png", ImageShape(width=160, height=120))
    assert len(records) == 1
    assert records[0].bbox.width > 0
    assert records[0].polygon is not None


def test_coco_reader_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationFailure) as error:
        CocoReader(tmp_path).categories()
    assert error.value.code is ErrorCode.ANNOTATION_INVALID


def test_coco_reader_rejects_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "instances.json").write_text("{not json")
    with pytest.raises(ValidationFailure):
        CocoReader(tmp_path).categories()


def test_coco_reader_ignores_entries_without_images() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        (root / "annotations").mkdir()
        document = {
            "images": [{"id": 1, "file_name": "a.png", "width": 10, "height": 10}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 2, 2]},
                {"id": 2, "image_id": 99, "category_id": 0, "bbox": [1, 1, 2, 2]},
                {"id": 3, "image_id": 1, "category_id": 0, "bbox": [0, 0, 0, 0]},
            ],
            "categories": [{"id": 0, "name": "car"}],
        }
        (root / "annotations" / "instances.json").write_text(json.dumps(document))
        reader = CocoReader(root)
        records = reader.read_image("a.png", ImageShape(width=10, height=10))
        assert len(records) == 1


def test_coco_writer_build_is_structurally_valid() -> None:
    writer = CocoWriter(instance=True)
    document = writer.build(
        bundles=[("img_000.png", _bundle())],
        categories=_bundle().categories,
    )
    assert set(document) == {"info", "licenses", "images", "annotations", "categories"}
    assert document["images"][0]["file_name"] == "img_000.png"
    assert len(document["annotations"]) == 2
    assert document["annotations"][0]["segmentation"][0][:2] == [10.0, 10.0]
    assert document["annotations"][1]["iscrowd"] == 1


def test_coco_detection_writer_omits_segmentation() -> None:
    document = CocoWriter(instance=False).build(
        bundles=[("img.png", _bundle())], categories=_bundle().categories
    )
    assert all("segmentation" not in entry for entry in document["annotations"])


def test_coco_roundtrip_preserves_boxes_and_polygons(tmp_path: Path) -> None:
    bundle = _bundle()
    document = write_annotations(
        "coco-instance",
        [("img_000.png", bundle)],
        categories=bundle.categories,
        image_paths={"img_000.png": "images/img_000.png"},
    )
    root = tmp_path / "coco"
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "instances.json").write_text(json.dumps(document))
    reader = CocoReader(root)
    records = reader.read_image("img_000.png", bundle.image_shape)
    assert len(records) == 2
    assert records[0].bbox.as_xyxy() == pytest.approx((10.0, 10.0, 50.0, 40.0))
    assert records[0].polygon is not None
    assert records[0].polygon.rings[0][0] == Point(x=10.0, y=10.0)
    assert records[1].iscrowd is True


# --------------------------------------------------------------------------- #
# YOLO
# --------------------------------------------------------------------------- #


def test_yolo_reader_reads_boxes(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path / "data", count=3, annotations=False)
    write_yolo(tmp_path / "data", dataset, list(dataset.splits), segmentation=False)
    reader = YoloReader(tmp_path / "data", segmentation=False)
    assert [category.name for category in reader.categories()] == ["car"]
    records = reader.read_image("img_000.png", ImageShape(width=160, height=120))
    assert len(records) == 1
    assert records[0].bbox.area > 0


def test_yolo_reader_reads_segmentation(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path / "data", count=2, annotations=False)
    write_yolo(tmp_path / "data", dataset, list(dataset.splits), segmentation=True)
    records = YoloReader(tmp_path / "data", segmentation=True).read_image(
        "img_000.png", ImageShape(width=160, height=120)
    )
    assert records[0].polygon is not None
    assert records[0].bbox.area > 0


def test_yolo_reader_requires_class_names(tmp_path: Path) -> None:
    (tmp_path / "labels").mkdir()
    with pytest.raises(ValidationFailure, match="no class names"):
        YoloReader(tmp_path).categories()


def test_yolo_reader_reads_names_from_data_yaml(tmp_path: Path) -> None:
    (tmp_path / "data.yaml").write_text("names:\n  0: car\n  1: van\n")
    names = [category.name for category in YoloReader(tmp_path).categories()]
    assert names == ["car", "van"]


def test_yolo_reader_rejects_a_malformed_line(tmp_path: Path) -> None:
    (tmp_path / "classes.txt").write_text("car\n")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "a.txt").write_text("0 not-a-number\n")
    with pytest.raises(ValidationFailure, match="non-numeric"):
        YoloReader(tmp_path).read_image("a.png", ImageShape(width=10, height=10))


def test_yolo_reader_rejects_a_short_line(tmp_path: Path) -> None:
    (tmp_path / "classes.txt").write_text("car\n")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "a.txt").write_text("0 0.5\n")
    with pytest.raises(ValidationFailure, match="fewer than five"):
        YoloReader(tmp_path).read_image("a.png", ImageShape(width=10, height=10))


def test_yolo_writer_emits_labels_classes_and_data_yaml() -> None:
    bundle = _bundle()
    document = YoloWriter(segmentation=False, image_paths={"img_000.png": "images/img_000.png"}).build(
        bundles=[("img_000.png", bundle)], categories=bundle.categories
    )
    assert "labels/img_000.txt" in document["labels"]
    assert document["classes"] == ["car", "van"]
    assert document["data_yaml"]["nc"] == 2
    lines = [line for line in document["labels"]["labels/img_000.txt"].splitlines() if line]
    assert len(lines) == 2
    fields = lines[0].split()
    assert len(fields) == 5
    assert all(0.0 <= float(value) <= 1.0 for value in fields[1:])


def test_yolo_segmentation_writer_emits_polygons() -> None:
    bundle = _bundle()
    document = YoloWriter(segmentation=True, image_paths={}).build(
        bundles=[("img_000.png", bundle)], categories=bundle.categories
    )
    fields = document["labels"]["labels/img_000.txt"].splitlines()[0].split()
    assert len(fields) == 1 + 8


def test_yolo_roundtrip_preserves_boxes(tmp_path: Path) -> None:
    root = tmp_path / "yolo"
    root.mkdir()
    bundle = _bundle()
    document = write_annotations(
        "yolo-detection", [("img_000.png", bundle)], categories=bundle.categories, image_paths={}
    )
    (root / "classes.txt").write_text("\n".join(document["classes"]) + "\n")
    for relative, text in document["labels"].items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    reader = YoloReader(root, segmentation=False)
    records = reader.read_image("img_000.png", bundle.image_shape)
    assert len(records) == 2
    assert records[0].bbox.as_xyxy() == pytest.approx((10.0, 10.0, 50.0, 40.0), abs=0.5)


def test_yolo_segmentation_roundtrip_preserves_polygons(tmp_path: Path) -> None:
    root = tmp_path / "yolo"
    root.mkdir()
    bundle = _bundle()
    document = write_annotations(
        "yolo-segmentation", [("img_000.png", bundle)], categories=bundle.categories, image_paths={}
    )
    (root / "classes.txt").write_text("\n".join(document["classes"]) + "\n")
    for relative, text in document["labels"].items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    records = YoloReader(root, segmentation=True).read_image("img_000.png", bundle.image_shape)
    assert records[0].polygon is not None
    assert len(records[0].polygon.rings[0]) == 4


# --------------------------------------------------------------------------- #
# Batch reading and shared behaviour
# --------------------------------------------------------------------------- #


def test_read_annotations_reads_a_whole_coco_dataset(tmp_path: Path) -> None:
    build_dataset(tmp_path / "data", count=4, annotations=True)
    bundles, categories = read_annotations(
        "coco-instance",
        tmp_path / "data",
        shape_lookup=lambda relative: ImageShape(width=160, height=120),
    )
    assert len(bundles) == 4
    assert [category.name for category in categories] == ["car"]
    assert all(bundle.object_count == 1 for bundle in bundles.values())


def test_read_annotations_reads_a_whole_yolo_dataset(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path / "data", count=3, annotations=False)
    write_yolo(tmp_path / "data", dataset, list(dataset.splits), segmentation=False)
    bundles, categories = read_annotations(
        "yolo-detection",
        tmp_path / "data",
        shape_lookup=lambda relative: ImageShape(width=160, height=120),
    )
    assert len(bundles) == 3
    assert [category.name for category in categories] == ["car"]


def test_unknown_format_is_refused() -> None:
    with pytest.raises(ValidationFailure) as error:
        read_annotations("pascal-voc", Path(), shape_lookup=lambda relative: None)
    assert error.value.code is ErrorCode.ANNOTATION_INVALID


def test_export_profile_reports_family() -> None:
    coco = ExportProfile(format="coco-detection")
    assert coco.is_coco
    assert not coco.is_yolo
    assert not coco.is_segmentation
    yolo = ExportProfile(format="yolo-segmentation")
    assert yolo.is_yolo
    assert yolo.is_segmentation


def test_writing_a_dataset_with_no_objects_produces_an_empty_but_valid_document() -> None:
    empty = AnnotationBundle(
        sample_id="empty.png",
        image_shape=ImageShape(width=8, height=8),
        categories=(CategorySpec(category_id=0, name="car"),),
    )
    document = write_annotations(
        "coco-instance", [("empty.png", empty)], categories=empty.categories, image_paths={}
    )
    assert document["annotations"] == []
    assert len(document["images"]) == 1


def test_coco_annotation_file_for_a_split_is_written_by_the_fixture(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path / "data", count=10, annotations=False, split_mode="directory")
    write_coco(tmp_path / "data", dataset, ["train/img_000.png"], filename="annotations/instances_train.json")
    document = json.loads((tmp_path / "data" / "annotations" / "instances_train.json").read_text())
    assert document["images"][0]["file_name"] == "train/img_000.png"
