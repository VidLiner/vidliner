"""Dataset sources: discovering what to augment, and where it belongs.

Two jobs live here, and both of them are dataset-integrity work rather than convenience:

* **discovery** — turn a directory plus a recipe into a list of
  :class:`~vidliner.domain_samples.DatasetSample`, each with a content digest, a split, and a
  lineage root;
* **split assignment** — decide, from the directory layout, the file names, or a manifest, which
  split each sample belongs to, and refuse to augment a split the recipe does not allow.

Nothing here writes to the dataset directory. Sources are read-only for the whole lifetime of a job.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from vidliner.core.canonical import digest_file
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.core.identity import sample_id
from vidliner.domain.annotations import AnnotationBundle
from vidliner.domain.enums import MediaKind, SplitName
from vidliner.domain.media import IMAGE_SUFFIXES, VIDEO_SUFFIXES, probe_image_shape
from vidliner.domain.recipe import Recipe
from vidliner.domain.shapes import ImageShape

__all__ = ["DatasetSample", "DatasetSource", "DiscoveryResult", "discover_dataset"]

_METADATA_DIRECTORIES: frozenset[str] = frozenset(
    {"annotations", "labels", "provenance", "review", "reports"}
)
"""Dataset subdirectories that hold metadata rather than media, and are never samples."""


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """One source sample discovered on disk."""

    sample_id: str
    relative_path: str
    absolute_path: Path
    digest: str
    size_bytes: int
    media_kind: MediaKind
    shape: ImageShape
    split: str
    root_digest: str
    annotation_path: Path | None = None
    bundled_annotation: AnnotationBundle | None = None
    perceptual_hash: str | None = None

    @property
    def is_augmentable(self) -> bool:
        """Whether the sample's split permits augmentation under the recipe."""
        return True  # actual policy is enforced by :meth:`DiscoveryResult.assert_augmentable`

    @property
    def lineage_root(self) -> str:
        """The digest at the root of this sample's augmentation lineage."""
        return self.root_digest or self.digest


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Everything a job needs to know about its input dataset."""

    root: Path
    samples: tuple[DatasetSample, ...]
    splits: tuple[str, ...]
    augment_splits: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...] = ()
    """``(relative path, reason)`` for every file deliberately not ingested."""
    notes: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        """Number of discovered samples."""
        return len(self.samples)

    @property
    def augmentable(self) -> tuple[DatasetSample, ...]:
        """Samples whose split the recipe permits augmenting."""
        allowed = set(self.augment_splits)
        return tuple(sample for sample in self.samples if sample.split in allowed)

    @property
    def split_counts(self) -> dict[str, int]:
        """Sample count per split."""
        counts: dict[str, int] = {}
        for sample in self.samples:
            counts[sample.split] = counts.get(sample.split, 0) + 1
        return counts

    def assert_augmentable(self, sample: DatasetSample) -> None:
        """Raise when a sample's split may not be augmented.

        Raises:
            ValidationFailure: with ``SOURCE_SPLIT_NOT_ALLOWED`` semantics, so the refusal is
                machine-readable rather than a message a caller has to parse.
        """
        if sample.split not in set(self.augment_splits):
            raise ValidationFailure(
                f"sample {sample.relative_path} belongs to split {sample.split!r}, which the recipe "
                f"does not permit augmenting (allowed: {sorted(self.augment_splits)})",
                code=ErrorCode.SPLIT_NOT_AUGMENTABLE,
                detail={"sample": sample.sample_id, "split": sample.split},
            )

    def by_id(self, identifier: str) -> DatasetSample | None:
        """Look up a sample by identity."""
        for sample in self.samples:
            if sample.sample_id == identifier:
                return sample
        return None


@dataclass
class DatasetSource:
    """Discovery over one dataset directory."""

    root: Path
    recipe: Recipe
    include_video: bool = False
    notes: list[str] = field(default_factory=list)

    def discover(self) -> DiscoveryResult:
        """Walk the dataset and build the sample list."""
        if not self.root.is_dir():
            raise ValidationFailure(
                f"dataset directory {self.root} does not exist",
                code=ErrorCode.WORKSPACE_INVALID,
                detail={"path": str(self.root)},
            )
        split_map = self._split_lookup()
        augment_splits = tuple(self.recipe.dataset.augment_splits)
        samples: list[DatasetSample] = []
        skipped: list[tuple[str, str]] = []
        limit = self.recipe.dataset.max_samples
        for path in self._iter_files():
            relative = path.relative_to(self.root).as_posix()
            kind = _media_kind(path, include_video=self.include_video)
            if kind is None:
                skipped.append((relative, "unsupported media extension"))
                continue
            if kind is MediaKind.VIDEO and not self.include_video:
                skipped.append((relative, "video ingest is a phase-2 capability"))
                continue
            split = split_map(relative)
            try:
                if kind is MediaKind.IMAGE:
                    shape = probe_image_shape(path)
                else:
                    shape = ImageShape(width=1, height=1)
            except ValidationFailure as exc:
                skipped.append((relative, exc.message))
                continue
            digest = digest_file(path)
            root_digest = digest
            samples.append(
                DatasetSample(
                    sample_id=sample_id(root_digest, relative),
                    relative_path=relative,
                    absolute_path=path,
                    digest=digest,
                    size_bytes=path.stat().st_size,
                    media_kind=kind,
                    shape=shape,
                    split=split,
                    root_digest=root_digest,
                )
            )
            if limit and len(samples) >= limit:
                self.notes.append(f"stopped after {limit} samples (dataset.max_samples)")
                break
        samples.sort(key=lambda item: item.relative_path)
        return DiscoveryResult(
            root=self.root,
            samples=tuple(samples),
            splits=tuple(sorted({sample.split for sample in samples})),
            augment_splits=augment_splits,
            skipped=tuple(skipped),
            notes=tuple(self.notes),
        )

    # -- internals --------------------------------------------------------- #

    def _iter_files(self) -> Iterator[Path]:
        pattern = "**/*" if self.recipe.dataset.recursive else "*"
        for path in sorted(self.root.glob(pattern)):
            relative_parts = path.relative_to(self.root).parts
            if not path.is_file() or any(part.startswith(".") for part in relative_parts):
                continue
            if len(relative_parts) > 1 and relative_parts[0] in _METADATA_DIRECTORIES:
                continue
            yield path

    def _split_lookup(self):
        """Build a function mapping a relative path to a split name."""
        spec = self.recipe.dataset.splits
        names = tuple(spec.names)
        if spec.mode == "none":
            return lambda _path: SplitName.TRAIN.value
        if spec.mode == "directory":

            def by_directory(relative: str) -> str:
                parts = Path(relative).parts
                # Only the first directory segment can name a split, and it must not be a dataset
                # metadata directory (an 'annotations/' folder is full of JSON, not images, and the
                # file walk already ignores it).
                head = parts[0] if len(parts) > 1 else ""
                for name in names:
                    if head.lower() == name.lower():
                        return _canonical_split(name)
                return SplitName.UNASSIGNED.value

            return by_directory
        if spec.mode == "filename":

            def by_filename(relative: str) -> str:
                stem = Path(relative).stem.lower()
                for name in names:
                    if stem.startswith(name.lower()) or f"_{name.lower()}" in stem:
                        return _canonical_split(name)
                return SplitName.UNASSIGNED.value

            return by_filename
        manifest_path = self.root / (spec.manifest or "splits.json")
        if not manifest_path.is_file():
            raise ValidationFailure(
                f"split mode 'manifest' requires {manifest_path} to exist",
                code=ErrorCode.WORKSPACE_INVALID,
                detail={"path": str(manifest_path)},
            )
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationFailure(
                f"split manifest {manifest_path} is not valid JSON: {exc}",
                code=ErrorCode.WORKSPACE_INVALID,
            ) from exc
        if not isinstance(document, dict):
            raise ValidationFailure(
                f"split manifest {manifest_path} must map relative paths to split names",
                code=ErrorCode.WORKSPACE_INVALID,
            )
        mapping = {
            str(key).replace("\\", "/"): _canonical_split(str(value)) for key, value in document.items()
        }
        return lambda relative: mapping.get(relative, SplitName.UNASSIGNED.value)


def _canonical_split(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in {"val", "valid", "validation"}:
        return SplitName.VALIDATION.value
    if lowered in {"test", "testing"}:
        return SplitName.TEST.value
    if lowered in {"train", "training"}:
        return SplitName.TRAIN.value
    return lowered or SplitName.UNASSIGNED.value


def _media_kind(path: Path, *, include_video: bool) -> MediaKind | None:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return MediaKind.IMAGE
    if suffix in VIDEO_SUFFIXES and include_video:
        return MediaKind.VIDEO
    return None


def discover_dataset(recipe: Recipe, root: Path, *, include_video: bool = False) -> DiscoveryResult:
    """Convenience wrapper around :class:`DatasetSource`."""
    return DatasetSource(root=root, recipe=recipe, include_video=include_video).discover()
