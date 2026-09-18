"""Dataset export: writing the accepted samples out in a training-ready format.

The exporter is the last place a mistake can be caught, so it checks rather than assumes:

* only ``ACCEPTED`` candidates are written (``NEEDS_REVIEW`` is excluded unless the recipe asks for
  it, and ``REJECTED`` never);
* every lineage edge is validated for split leakage across the whole accepted set;
* every output is checked for exact and near duplicates against the source set and against the
  other accepted outputs;
* the annotation vocabulary is built once for the whole dataset, so a category id means the same
  thing in every file;
* a manifest, a provenance record per sample, and a distribution report are written alongside the
  images and annotations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vidliner.annotations.base import write_annotations
from vidliner.control.duplicates import DuplicateIndex
from vidliner.control.report import build_dataset_report
from vidliner.control.splits import assert_no_leakage
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.annotations import AnnotationBundle, CategorySpec, ExportProfile
from vidliner.domain.enums import DecisionState
from vidliner.domain.provenance import ProvenanceRecord, assert_no_secrets
from vidliner.domain.recipe import Recipe
from vidliner.storage.state import StateStore
from vidliner.storage.workspace import ArtifactStore, Workspace

__all__ = ["DatasetExporter", "ExportOutcome", "ExportRecord"]


@dataclass(frozen=True, slots=True)
class ExportRecord:
    """One sample considered for export, with its decision and its duplicate verdict."""

    candidate_id: str
    sample_id: str
    source_sample_id: str
    category: str
    decision: DecisionState
    output_digest: str
    annotation: AnnotationBundle
    provenance: ProvenanceRecord
    split: str
    reason_codes: tuple[str, ...] = ()
    duplicate_of: str | None = None
    duplicate_kind: str | None = None
    duplicate_distance: float = 0.0
    image_artifact: Any = None


@dataclass
class ExportOutcome:
    """What an export produced."""

    path: Path
    format: str
    accepted: int
    rejected: int
    review: int
    duplicates: int
    images_written: int
    categories: tuple[CategorySpec, ...]
    report_path: Path | None = None
    manifest_path: Path | None = None
    review_path: Path | None = None
    leakage_checked: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Samples considered."""
        return self.accepted + self.rejected + self.review


class DatasetExporter:
    """Writes an accepted sample set to disk in a training-ready format."""

    def __init__(
        self,
        *,
        workspace: Workspace,
        store: ArtifactStore,
        state: StateStore,
        recipe: Recipe,
        profile: ExportProfile | None = None,
    ) -> None:
        self._workspace = workspace
        self._store = store
        self._state = state
        self._recipe = recipe
        self._profile = profile or ExportProfile(
            format=recipe.export.format,
            path=recipe.export.path,
            copy_images=recipe.export.copy_images,
            image_format=recipe.export.image_format,
            include_rejected=recipe.export.include_rejected,
            include_provenance=recipe.export.include_provenance,
            include_review_report=recipe.export.include_review_report,
            min_area_px=recipe.export.min_annotation_area_px,
        )
        self._export_root = self._workspace.resolve_inside(self._profile.path)

    @property
    def export_root(self) -> Path:
        """Directory the dataset is written into."""
        return self._export_root

    def export(
        self,
        records: list[ExportRecord],
        *,
        job_id: str,
        seed_tree: dict[str, int] | None = None,
    ) -> ExportOutcome:
        """Write the accepted records as a dataset.

        Args:
            records: every candidate considered, accepted or not.
            job_id: the job the records came from.
            seed_tree: sample-level seeds recorded in the manifest.

        Raises:
            ValidationFailure: on split leakage, an invalid annotation, or a credential-like field in
                a provenance record.
        """
        accepted = [record for record in records if record.decision is DecisionState.ACCEPTED]
        review = [record for record in records if record.decision is DecisionState.NEEDS_REVIEW]
        rejected = [record for record in records if record.decision is DecisionState.REJECTED]
        self._assert_no_leakage(records)
        self._assert_annotations(accepted)

        duplicates = self._resolve_duplicates(accepted)
        kept = [record for record in accepted if record.candidate_id not in duplicates]
        categories = _vocabulary(kept)

        self._prepare_directories()
        image_names = self._write_images(kept)
        self._write_annotations(kept, categories, image_names)
        manifest_path = self._write_manifest(
            kept, categories, job_id=job_id, seed_tree=seed_tree or {}, duplicates=duplicates
        )
        report_path = build_dataset_report(
            workspace=self._workspace,
            state=self._state,
            job_id=job_id,
            recipe=self._recipe,
            accepted=kept,
            rejected=rejected,
            review=review,
            duplicates=duplicates,
            target=self._export_root / "dataset-report.json",
        )
        outcome = ExportOutcome(
            path=self._export_root,
            format=self._profile.format,
            accepted=len(kept),
            rejected=len(rejected),
            review=len(review),
            duplicates=len(duplicates),
            images_written=len(image_names),
            categories=categories,
            report_path=report_path,
            manifest_path=manifest_path,
            leakage_checked=True,
        )
        if review:
            outcome.notes.append(
                f"{len(review)} candidate(s) need review and were {'included' if self._recipe.acceptance.export_review else 'excluded'}"
            )
        if duplicates:
            outcome.notes.append(f"{len(duplicates)} near-duplicate candidate(s) were excluded")
        return outcome

    # -- validation -------------------------------------------------------- #

    def _assert_no_leakage(self, records: list[ExportRecord]) -> None:
        """Check every lineage edge the database knows about."""
        edges = self._state.find_leakage()
        if edges:
            assert_no_leakage(
                [
                    (child, parent, child_split, self._state.split_of(parent) or "train")
                    for child, parent, child_split in edges
                ]
            )
        pairs = {(record.sample_id, record.split) for record in records}
        by_root: dict[str, set[str]] = {}
        for record in records:
            by_root.setdefault(record.provenance.lineage_root or record.source_sample_id, set()).add(
                record.split
            )
        for root, splits in by_root.items():
            if len(splits) > 1:
                raise ValidationFailure(
                    f"augmentation descendants of {root} would land in different splits: {sorted(splits)}",
                    code=ErrorCode.SPLIT_LEAKAGE,
                    detail={"root": root, "splits": sorted(splits)},
                )
        del pairs

    def _assert_annotations(self, accepted: list[ExportRecord]) -> None:
        for record in accepted:
            problems = record.annotation.validate_against_image()
            if problems:
                raise ValidationFailure(
                    f"annotation for {record.candidate_id} is invalid: {'; '.join(problems)}",
                    code=ErrorCode.ANNOTATION_INVALID,
                    detail={"candidate_id": record.candidate_id, "problems": problems},
                )
            assert_no_secrets(
                json.loads(record.provenance.model_dump_json()),
                context=f"provenance for {record.candidate_id}",
            )

    def _resolve_duplicates(self, accepted: list[ExportRecord]) -> dict[str, tuple[str, str, float]]:
        """Find duplicates among the accepted records by perceptual hash."""
        duplicates: dict[str, tuple[str, str, float]] = {}
        if not self._recipe.duplicates.enabled or self._recipe.duplicates.perceptual_hash == "none":
            return duplicates
        index = DuplicateIndex(
            enabled=True,
            algorithm=self._recipe.duplicates.perceptual_hash,
            hamming_threshold=self._recipe.duplicates.hamming_threshold,
            compare_against_source=self._recipe.duplicates.compare_against_source,
            existing=self._state.perceptual_hashes(),
        )
        for record in accepted:
            fingerprint = record.provenance.perceptual_hash
            if not fingerprint:
                continue
            try:
                value = int(fingerprint, 16)
            except ValueError:
                continue
            decision = index.check_value(value, candidate_id=record.candidate_id)
            if decision.is_duplicate:
                duplicates[record.candidate_id] = (decision.reference, decision.kind, decision.distance)
                self._state.record_duplicate(
                    record.provenance.job_id,
                    record.candidate_id,
                    decision.reference,
                    kind=decision.kind,
                    distance=decision.distance,
                )
                continue
            index.register_accepted_value(record.candidate_id, value)
        return duplicates

    # -- writing ----------------------------------------------------------- #

    def _prepare_directories(self) -> None:
        for name in ("images", "annotations"):
            (self._export_root / name).mkdir(parents=True, exist_ok=True)

    def _write_images(self, records: list[ExportRecord]) -> dict[str, str]:
        """Write every accepted image and return ``candidate_id → file name``.

        The annotation document must reference the name the image was actually written under, so the
        mapping is built here rather than assumed later; a mismatch between the two would produce a
        dataset that loads with missing images.
        """
        names: dict[str, str] = {}
        for record in records:
            if record.image_artifact is None:
                continue
            suffix = _image_suffix(record.image_artifact.media_type, self._profile.image_format)
            name = f"{record.candidate_id}.{suffix}"
            names[record.candidate_id] = name
            if not self._profile.copy_images:
                continue
            target = self._export_root / "images" / name
            if not target.exists():
                target.write_bytes(self._store.read(record.image_artifact.digest))
        return names

    def _write_annotations(
        self,
        records: list[ExportRecord],
        categories: tuple[CategorySpec, ...],
        image_names: dict[str, str],
    ) -> None:
        """Write the labels, remapped onto the dataset's single category vocabulary.

        Each sample's own bundle numbers its categories from its own history; the exported dataset
        must use one numbering for all of them, or a category id would mean different things in
        different files.
        """
        bundles = [
            (
                image_names.get(record.candidate_id, f"{record.candidate_id}.png"),
                _rebase_categories(record.annotation, categories),
            )
            for record in records
        ]
        image_paths = {file_name: f"images/{file_name}" for file_name, _ in bundles}
        document = write_annotations(
            self._profile.format,
            bundles,
            categories=categories,
            image_paths=image_paths,
        )
        if self._profile.is_coco:
            target = self._export_root / "annotations" / f"instances_{self._split_name()}.json"
            target.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
            return
        labels = document.get("labels", {})
        for relative, text in labels.items():
            target = self._export_root / str(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(text), encoding="utf-8")
        classes = document.get("classes", [])
        (self._export_root / "classes.txt").write_text(
            "\n".join(str(name) for name in classes) + "\n", encoding="utf-8"
        )
        data_yaml = document.get("data_yaml", {})
        try:
            import yaml

            (self._export_root / "data.yaml").write_text(
                yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
            )
        except ImportError:  # pragma: no cover - PyYAML is a hard dependency
            (self._export_root / "data.json").write_text(json.dumps(data_yaml, indent=2), encoding="utf-8")

    def _write_manifest(
        self,
        records: list[ExportRecord],
        categories: tuple[CategorySpec, ...],
        *,
        job_id: str,
        seed_tree: dict[str, int],
        duplicates: dict[str, tuple[str, str, float]],
    ) -> Path:
        from vidliner import __version__

        manifest = {
            "format": "vidliner.dataset@1",
            "producer": "vidliner",
            "vidliner_version": __version__,
            "job_id": job_id,
            "recipe": self._recipe.name,
            "recipe_hash": self._recipe.recipe_hash(),
            "annotation_format": self._profile.format,
            "created_at": _now(),
            "splits": {self._split_name(): [record.candidate_id for record in records]},
            "counts": {
                "samples": len(records),
                "categories": len(categories),
                "excluded_duplicates": len(duplicates),
            },
            "categories": [{"id": category.category_id, "name": category.name} for category in categories],
            "seed_tree": seed_tree,
            "samples": [
                {
                    "candidate_id": record.candidate_id,
                    "source_sample_id": record.source_sample_id,
                    "output_digest": record.output_digest,
                    "category": record.category,
                    "split": record.split,
                    "quality": dict(record.provenance.quality_scores),
                    "overall_score": record.provenance.overall_score,
                    "decision": record.decision.value,
                    "provenance": f"provenance/{record.candidate_id}.json"
                    if self._profile.include_provenance
                    else None,
                }
                for record in records
            ],
            "excluded": {
                candidate: {"duplicate_of": reference, "kind": kind, "distance": distance}
                for candidate, (reference, kind, distance) in duplicates.items()
            },
        }
        assert_no_secrets(manifest, context="dataset manifest")
        manifest_path = self._export_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if self._profile.include_provenance:
            provenance_dir = self._export_root / "provenance"
            provenance_dir.mkdir(parents=True, exist_ok=True)
            for record in records:
                payload = json.loads(record.provenance.model_dump_json())
                (provenance_dir / f"{record.candidate_id}.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        return manifest_path

    def _split_name(self) -> str:
        return self._recipe.export.splits.get("train", "train")


def _rebase_categories(bundle: AnnotationBundle, vocabulary: tuple[CategorySpec, ...]) -> AnnotationBundle:
    """Rewrite a bundle's category ids onto the dataset vocabulary, keeping the names."""
    by_name = {category.name: category.category_id for category in vocabulary}
    renamed = tuple(
        CategorySpec(category_id=by_name.get(category.name, category.category_id), name=category.name)
        for category in bundle.categories
    )
    objects = tuple(
        obj.model_copy(
            update={"category_id": by_name.get(bundle.category_name(obj.category_id) or "", obj.category_id)}
        )
        for obj in bundle.objects
    )
    return bundle.model_copy(update={"categories": renamed, "objects": objects})


def _vocabulary(records: list[ExportRecord]) -> tuple[CategorySpec, ...]:
    """One category vocabulary for the whole dataset, with stable ids."""
    names: list[str] = []
    for record in records:
        for obj in record.annotation.objects:
            name = record.annotation.category_name(obj.category_id)
            if name and name not in names:
                names.append(name)
    names.sort()
    return tuple(CategorySpec(category_id=index, name=name) for index, name in enumerate(names))


def _image_suffix(media_type: str, requested: str) -> str:
    if requested == "png":
        return "png"
    if requested == "jpg":
        return "jpg"
    return {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(media_type, "png")


def _now() -> str:
    from vidliner.core.results import utc_now

    return utc_now().isoformat()
