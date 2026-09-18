"""Storage layout, artifact store, and backend I/O.

Layout of a VidLiner workspace (all names are this project's own; no other project's layout is
mirrored):

```
<workspace>/
  runtime.yaml                  runtime profile (optional; may also be given with --runtime)
  state.db                      SQLite job/candidate/cache state
  artifacts/<aa>/<digest>.<ext> content-addressed artifact store
  cache/                        optional on-disk scratch for backends that need a work directory
  runs/<job_id>/
    manifest.json               job manifest
    plan.json                   compiled plan and estimate
    events.jsonl                structured event log
    samples/<sample_id>/
      source.png                the untouched source frame, for comparison
      target_mask.png           the mask the generator was given
      generated.png             the raw generated candidate
      refined.png               after the refinement operators
      difference.png            absolute difference against the source
      annotation_overlay.png    the rebuilt annotation drawn on the generated image
      annotation.json           the rebuilt annotation in the internal IR
      quality.json              the quality report
      decision.json             the acceptance decision with reason codes
      provenance.json           the provenance record
    accepted/                   images and annotations of accepted samples
    rejected/                   diagnostics of rejected samples (unless discarded)
    review/
      index.html                static review report
    dataset/                    exported dataset (created by `vidliner export`)
      images/
      annotations/
      manifest.json
      dataset-report.json
    job-summary.json
```

Sources are read-only: the pipeline never writes to the dataset directory it read from. Every
derived file is either content-addressed in ``artifacts/`` or a report inside the job's run
directory.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from vidliner.core.canonical import digest_bytes, digest_file
from vidliner.core.errors import ErrorCode, InfrastructureFailure, ValidationFailure
from vidliner.core.results import ArtifactRef
from vidliner.domain.enums import ArtifactKind
from vidliner.domain.shapes import ImageShape

__all__ = [
    "ARTIFACT_KIND_SUFFIX",
    "ArtifactStore",
    "BackendIO",
    "Workspace",
]

#: Preferred file extension per artifact kind. The store never relies on it for correctness, only
#: for making the artifact directory human-navigable.
ARTIFACT_KIND_SUFFIX: dict[ArtifactKind, str] = {
    ArtifactKind.SOURCE_MEDIA: "bin",
    ArtifactKind.IMAGE: "png",
    ArtifactKind.MASK: "png",
    ArtifactKind.POLYGON: "json",
    ArtifactKind.DEPTH: "npy",
    ArtifactKind.TRACK: "json",
    ArtifactKind.CANDIDATE: "png",
    ArtifactKind.REFINED_IMAGE: "png",
    ArtifactKind.DIFFERENCE: "png",
    ArtifactKind.ANNOTATION: "json",
    ArtifactKind.METRICS: "json",
    ArtifactKind.REPORT: "json",
    ArtifactKind.DATASET_SLICE: "json",
    ArtifactKind.MANIFEST: "json",
}

_MEDIA_TYPE_SUFFIX: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "video/mp4": "mp4",
    "application/json": "json",
    "application/octet-stream": "bin",
    "application/x-npy": "npy",
}

_SUFFIX_MEDIA_TYPE: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "mp4": "video/mp4",
    "json": "application/json",
    "npy": "application/x-npy",
    "bin": "application/octet-stream",
}

_DEFAULT_MAX_DIMENSION = 16384
"""Guard against decompression bombs: an image claiming to be larger than this is refused."""


class Workspace:
    """A VidLiner workspace: the boundary every path is validated against."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).expanduser().resolve()

    @property
    def root(self) -> Path:
        """Absolute workspace root."""
        return self._root

    @property
    def state_path(self) -> Path:
        """Path of the SQLite state database."""
        return self._root / "state.db"

    @property
    def artifacts_dir(self) -> Path:
        """Content-addressed artifact directory."""
        return self._root / "artifacts"

    @property
    def runs_dir(self) -> Path:
        """Per-job run directories."""
        return self._root / "runs"

    @property
    def cache_dir(self) -> Path:
        """Scratch directory for backends that need one."""
        return self._root / "cache"

    @property
    def datasets_dir(self) -> Path:
        """Default location for exported datasets."""
        return self._root / "datasets"

    def run_dir(self, job_id: str) -> Path:
        """Run directory for one job, created on demand."""
        path = self.runs_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def sample_dir(self, job_id: str, sample_id: str) -> Path:
        """Per-sample evidence directory inside a job run."""
        path = self.run_dir(job_id) / "samples" / sample_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def initialize(self) -> Workspace:
        """Create the workspace directories. Idempotent."""
        for directory in (
            self._root,
            self.artifacts_dir,
            self.runs_dir,
            self.cache_dir,
            self.datasets_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def is_initialized(self) -> bool:
        """True when the workspace looks like a VidLiner workspace."""
        return self.artifacts_dir.is_dir() and self.runs_dir.is_dir()

    def resolve_inside(self, path: str | Path, *, must_exist: bool = False) -> Path:
        """Resolve ``path`` and refuse anything that escapes the workspace.

        Relative paths resolve against the workspace root. Symlinks are resolved too, so a symlink
        pointing outside is refused rather than silently followed.

        Raises:
            ValidationFailure: on traversal outside the workspace, or when ``must_exist`` is set and
                the path is absent.
        """
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve()
        if resolved != self._root and self._root not in resolved.parents:
            raise ValidationFailure(
                f"path {path!s} resolves outside the workspace boundary {self._root}",
                code=ErrorCode.PATH_OUTSIDE_WORKSPACE,
                detail={"path": str(path), "workspace": str(self._root)},
            )
        if must_exist and not resolved.exists():
            raise ValidationFailure(
                f"path {resolved} does not exist",
                code=ErrorCode.WORKSPACE_INVALID,
                detail={"path": str(resolved)},
            )
        return resolved

    def relative(self, path: Path) -> str:
        """Express ``path`` relative to the workspace when possible."""
        try:
            return path.resolve().relative_to(self._root).as_posix()
        except ValueError:
            return path.as_posix()

    @contextmanager
    def scratch(self, prefix: str = "vidliner-") -> Iterator[Path]:
        """A temporary directory inside the workspace cache, removed on exit."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=self.cache_dir))
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)


class ArtifactStore:
    """Content-addressed artifact storage.

    Writing the same bytes twice writes one file. Reading always verifies the digest, so a corrupted
    or truncated cache can never be mistaken for a valid result.
    """

    def __init__(
        self,
        workspace: Workspace,
        *,
        max_artifact_bytes: int = 256 * 1024 * 1024,
        max_dimension: int = _DEFAULT_MAX_DIMENSION,
    ) -> None:
        self._workspace = workspace
        self._max_bytes = max_artifact_bytes
        self._max_dimension = max_dimension

    @property
    def workspace(self) -> Workspace:
        """The workspace this store writes into."""
        return self._workspace

    @property
    def max_artifact_bytes(self) -> int:
        """Largest artifact this store will accept."""
        return self._max_bytes

    def path_for(self, digest: str, *, suffix: str | None = None) -> Path:
        """Deterministic storage path for a digest, sharded by its first two characters."""
        if len(digest) < 4:
            raise ValueError("artifact digest is too short to shard")
        shard = digest[:2]
        extension = suffix or "bin"
        return self._workspace.artifacts_dir / shard / f"{digest}.{extension}"

    def _locate(self, digest: str) -> Path | None:
        shard_dir = self._workspace.artifacts_dir / digest[:2]
        if not shard_dir.is_dir():
            return None
        matches = sorted(shard_dir.glob(f"{digest}.*"))
        return matches[0] if matches else None

    def write(
        self,
        data: bytes,
        *,
        kind: ArtifactKind,
        media_type: str,
        suffix: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> ArtifactRef:
        """Write ``data`` and return its reference. Existing content is reused, never rewritten.

        Raises:
            ValidationFailure: when the payload exceeds the size cap or declares implausible
                dimensions.
        """
        if len(data) > self._max_bytes:
            raise ValidationFailure(
                f"artifact of {len(data)} bytes exceeds the {self._max_bytes} byte cap",
                code=ErrorCode.MEDIA_TOO_LARGE,
                detail={"bytes": len(data), "cap": self._max_bytes, "media_type": media_type},
            )
        if width is not None and height is not None:
            self._validate_dimensions(width, height, media_type=media_type)
        digest = digest_bytes(data)
        extension = suffix or _MEDIA_TYPE_SUFFIX.get(media_type) or ARTIFACT_KIND_SUFFIX.get(kind, "bin")
        target = self.path_for(digest, suffix=extension)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write(target, data)
        return ArtifactRef(
            kind=kind,
            digest=digest,
            media_type=media_type,
            size_bytes=len(data),
            width=width,
            height=height,
            suffix=extension,
        )

    def write_text(
        self,
        text: str,
        *,
        kind: ArtifactKind,
        media_type: str = "application/json",
        suffix: str = "json",
    ) -> ArtifactRef:
        """Write UTF-8 text as an artifact."""
        return self.write(text.encode("utf-8"), kind=kind, media_type=media_type, suffix=suffix)

    def write_json(self, payload: object, *, kind: ArtifactKind, suffix: str = "json") -> ArtifactRef:
        """Write a JSON artifact with stable formatting."""
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False, default=str)
        return self.write_text(text, kind=kind, media_type="application/json", suffix=suffix)

    def _atomic_write(self, target: Path, data: bytes) -> None:
        """Write ``data`` to ``target`` atomically: temp file, fsync, rename."""
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".tmp") as handle:
                tmp_path = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            tmp_path.replace(target)
        except OSError as exc:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise InfrastructureFailure(
                f"failed to write artifact {target}: {exc}",
                code=ErrorCode.STATE_STORE_FAILED,
                detail={"path": str(target)},
            ) from exc

    def _validate_dimensions(self, width: int, height: int, *, media_type: str) -> None:
        if width <= 0 or height <= 0:
            raise ValidationFailure(
                f"artifact declares a non-positive dimension ({width}x{height})",
                code=ErrorCode.MEDIA_CORRUPT,
                detail={"width": width, "height": height, "media_type": media_type},
            )
        if width > self._max_dimension or height > self._max_dimension:
            raise ValidationFailure(
                f"artifact dimension {width}x{height} exceeds the {self._max_dimension}px guard",
                code=ErrorCode.MEDIA_TOO_LARGE,
                detail={"width": width, "height": height, "cap": self._max_dimension},
            )

    def validate_image_dimensions(self, width: int, height: int, *, media_type: str = "image/*") -> None:
        """Public dimension guard, used by :class:`BackendIO` after decoding an image."""
        self._validate_dimensions(width, height, media_type=media_type)

    def exists(self, digest: str) -> bool:
        """True when bytes for ``digest`` are present, without verifying them."""
        return self._locate(digest) is not None

    def path_of(self, digest: str) -> Path:
        """Path of a stored artifact.

        Raises:
            ValidationFailure: when the artifact is absent.
        """
        path = self._locate(digest)
        if path is None:
            raise ValidationFailure(
                f"artifact {digest[:12]} is not present in the store",
                code=ErrorCode.ARTIFACT_MISSING,
                detail={"digest": digest},
            )
        return path

    def read(self, digest: str, *, verify: bool = True) -> bytes:
        """Read artifact bytes, verifying the digest by default.

        Raises:
            ValidationFailure: when the artifact is missing or its bytes no longer match.
        """
        path = self.path_of(digest)
        data = path.read_bytes()
        if verify:
            actual = digest_bytes(data)
            if actual != digest:
                raise ValidationFailure(
                    f"artifact {digest[:12]} is corrupt: stored bytes hash to {actual[:12]}",
                    code=ErrorCode.ARTIFACT_DIGEST_MISMATCH,
                    detail={"expected": digest, "actual": actual, "path": str(path)},
                )
        return data

    def read_json(self, digest: str) -> object:
        """Read and parse a JSON artifact."""
        return json.loads(self.read(digest).decode("utf-8"))

    def register(
        self, source: Path, *, kind: ArtifactKind, media_type: str, move: bool = False
    ) -> ArtifactRef:
        """Import an existing file into the store.

        Args:
            source: file to import; it is read (or moved) but never modified in place.
            kind: artifact kind for the catalogue.
            media_type: declared MIME type.
            move: when true the source file is moved instead of copied, used for backend outputs
                written into a scratch directory.
        """
        if not source.is_file():
            raise ValidationFailure(
                f"cannot register missing file {source}",
                code=ErrorCode.ARTIFACT_MISSING,
                detail={"path": str(source)},
            )
        size = source.stat().st_size
        if size > self._max_bytes:
            raise ValidationFailure(
                f"file {source.name} of {size} bytes exceeds the {self._max_bytes} byte cap",
                code=ErrorCode.MEDIA_TOO_LARGE,
                detail={"path": str(source), "bytes": size},
            )
        digest = digest_file(source)
        extension = source.suffix.lstrip(".").lower() or ARTIFACT_KIND_SUFFIX.get(kind, "bin")
        target = self.path_for(digest, suffix=extension)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(str(source), str(target))
            else:
                shutil.copy2(source, target)
        elif move:
            source.unlink(missing_ok=True)
        media = _SUFFIX_MEDIA_TYPE.get(extension, media_type)
        width = height = None
        shape = self.probe_image_shape(target) if media.startswith("image/") else None
        if shape is not None:
            width, height = shape.width, shape.height
        return ArtifactRef(
            kind=kind,
            digest=digest,
            media_type=media,
            size_bytes=size,
            width=width,
            height=height,
            suffix=extension,
        )

    def probe_image_shape(self, path: Path) -> ImageShape | None:
        """Read image dimensions, returning ``None`` for non-image or unreadable content."""
        from vidliner.domain.media import probe_image_shape

        try:
            shape = probe_image_shape(path)
        except ValidationFailure:
            return None
        self._validate_dimensions(shape.width, shape.height, media_type="image/*")
        return shape

    def stats(self) -> dict[str, int]:
        """Artifact count and total bytes on disk."""
        count = 0
        total = 0
        for path in self._workspace.artifacts_dir.rglob("*"):
            if path.is_file() and not path.name.endswith(".tmp"):
                count += 1
                total += path.stat().st_size
        return {"artifacts": count, "bytes": total}


class BackendIO:
    """What a backend is allowed to do with storage.

    A backend can load an artifact, save new artifacts, and get a scratch directory. It cannot
    resolve workspace paths, read the state database, or write anywhere else.
    """

    def __init__(self, store: ArtifactStore, *, request_id: str = "") -> None:
        self._store = store
        self._request_id = request_id

    @property
    def store(self) -> ArtifactStore:
        """The underlying artifact store (read-only use expected)."""
        return self._store

    @property
    def request_id(self) -> str:
        """Identifier of the node this I/O is serving, for scratch directory naming."""
        return self._request_id

    def load(self, artifact: ArtifactRef) -> bytes:
        """Read the bytes of an artifact, verifying its digest."""
        return self._store.read(artifact.digest)

    def load_json(self, artifact: ArtifactRef) -> object:
        """Read a JSON artifact."""
        return self._store.read_json(artifact.digest)

    def image(self, artifact: ArtifactRef):
        """Load an artifact as a Pillow image in RGB or RGBA mode."""
        import io

        from PIL import Image

        data = self.load(artifact)
        with Image.open(io.BytesIO(data)) as handle:
            handle.load()
            self._store.validate_image_dimensions(handle.width, handle.height, media_type=artifact.media_type)
            if handle.mode in {"RGB", "RGBA"}:
                return handle.copy()
            return handle.convert("RGBA")

    def save_image(self, image: Any, *, kind: ArtifactKind = ArtifactKind.IMAGE) -> ArtifactRef:
        """Serialize a Pillow image to PNG and store it."""
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        data = buffer.getvalue()
        from PIL import Image

        with Image.open(io.BytesIO(data)) as handle:
            width, height = handle.width, handle.height
        return self._store.write(
            data, kind=kind, media_type="image/png", suffix="png", width=width, height=height
        )

    def save_mask(self, mask: Any) -> ArtifactRef:
        """Serialize a boolean mask as an 8-bit PNG and store it as a mask artifact."""
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(np.asarray(mask)).convert("L").save(buffer, format="PNG")
        data = buffer.getvalue()
        with Image.open(io.BytesIO(data)) as handle:
            width, height = handle.width, handle.height
        return self._store.write(
            data, kind=ArtifactKind.MASK, media_type="image/png", suffix="png", width=width, height=height
        )

    def load_mask(self, artifact: ArtifactRef):
        """Load a mask artifact as a boolean numpy array."""
        import io

        import numpy as np
        from PIL import Image

        data = self.load(artifact)
        with Image.open(io.BytesIO(data)) as handle:
            array = np.asarray(handle.convert("L"))
        return array > 127

    def save_json(self, payload: object, *, kind: ArtifactKind = ArtifactKind.REPORT) -> ArtifactRef:
        """Store a JSON payload as an artifact."""
        return self._store.write_json(payload, kind=kind)

    def save_bytes(self, data: bytes, *, kind: ArtifactKind, media_type: str) -> ArtifactRef:
        """Store raw bytes as an artifact."""
        return self._store.write(data, kind=kind, media_type=media_type)
