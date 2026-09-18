"""Media assets: immutable source files and the metadata read from them.

Two rules drive this module:

1. **Source files are never modified.** ``MediaAsset.path`` is read-only evidence; every derived
   file lives in the artifact store under its content digest.
2. **Dimensions are read from the header, not by decoding the picture.** The pipeline must be able
   to inspect thousands of files cheaply, and header parsing avoids importing an image library into
   the domain layer.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidliner.core.canonical import digest_file
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.enums import MediaKind
from vidliner.domain.shapes import ImageShape

__all__ = [
    "IMAGE_SUFFIXES",
    "VIDEO_SUFFIXES",
    "FrameRef",
    "MediaAsset",
    "VideoSpec",
    "inspect_media",
    "media_kind_for_suffix",
    "probe_image_shape",
    "probe_video_spec",
]

IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
VIDEO_SUFFIXES: Final[frozenset[str]] = frozenset({".mp4"})

_MEDIA_TYPES: Final[dict[str, str]] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
}


class VideoSpec(BaseModel):
    """Container-level facts about a video asset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fps: float | None = Field(default=None, gt=0)
    frame_count: int | None = Field(default=None, gt=0)
    duration_s: float | None = Field(default=None, ge=0)
    codec: str | None = None
    timescale: int | None = Field(default=None, gt=0)


class FrameRef(BaseModel):
    """Addresses one frame inside a media asset.

    Image samples use ``frame_index=0`` and ``timestamp_s=0.0``; that keeps the annotation and
    quality models identical for stills and video, which is what lets the video phase reuse them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_digest: str = Field(min_length=16, max_length=64)
    frame_index: int = Field(default=0, ge=0)
    timestamp_s: float = Field(default=0.0, ge=0)

    @property
    def is_still(self) -> bool:
        """True when this reference addresses a still image."""
        return self.frame_index == 0 and self.timestamp_s == 0.0


class MediaAsset(BaseModel):
    """One immutable source file.

    ``path`` is stored as given by the caller (the ingest step normalises it to a
    workspace-relative POSIX path). ``digest`` identifies the bytes and is the root of every
    derived identity in the pipeline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    kind: MediaKind
    digest: str = Field(min_length=16, max_length=64)
    size_bytes: int = Field(ge=0)
    media_type: str
    shape: ImageShape
    video: VideoSpec | None = None
    exif: dict[str, str] = Field(default_factory=dict)
    color_profile: str | None = None

    @model_validator(mode="after")
    def _check_video_consistency(self) -> MediaAsset:
        if self.kind is MediaKind.IMAGE and self.video is not None:
            raise ValueError("an image asset must not carry a video spec")
        return self

    @property
    def suffix(self) -> str:
        """Lower-case file extension including the dot."""
        return Path(self.path).suffix.lower()

    @property
    def frame_ref(self) -> FrameRef:
        """The frame reference for a still image asset."""
        return FrameRef(asset_digest=self.digest, frame_index=0, timestamp_s=0.0)

    def frame_ref_at(self, frame_index: int, timestamp_s: float = 0.0) -> FrameRef:
        """The frame reference for one frame of a video asset."""
        if frame_index < 0:
            raise ValueError("frame index must not be negative")
        return FrameRef(asset_digest=self.digest, frame_index=frame_index, timestamp_s=timestamp_s)


def media_kind_for_suffix(suffix: str) -> MediaKind | None:
    """Classify a file extension, or return ``None`` when VidLiner does not handle it."""
    lowered = suffix.lower()
    if lowered in IMAGE_SUFFIXES:
        return MediaKind.IMAGE
    if lowered in VIDEO_SUFFIXES:
        return MediaKind.VIDEO
    return None


def media_type_for_suffix(suffix: str) -> str:
    """Return the MIME type for a supported extension."""
    try:
        return _MEDIA_TYPES[suffix.lower()]
    except KeyError as exc:  # pragma: no cover - guarded by callers
        raise ValidationFailure(
            f"unsupported media extension {suffix!r}",
            code=ErrorCode.MEDIA_UNSUPPORTED,
        ) from exc


def probe_image_shape(path: Path) -> ImageShape:
    """Read image dimensions from the file header.

    Supports PNG, JPEG, and WebP, which is the ingest set required by the product specification.

    Raises:
        ValidationFailure: when the header is unreadable or the format is unsupported.
    """
    with path.open("rb") as handle:
        header = handle.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", header[16:24])
            return ImageShape(width=int(width), height=int(height))
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            return _probe_webp(handle, header)
        if header.startswith(b"\xff\xd8"):
            return _probe_jpeg(handle)
    raise ValidationFailure(
        f"{path.name} is not a PNG, JPEG, or WebP image, or its header is unreadable",
        code=ErrorCode.MEDIA_UNSUPPORTED,
        detail={"path": str(path)},
    )


def _probe_webp(handle: object, header: bytes) -> ImageShape:
    chunk = header[12:16]
    if chunk == b"VP8X":
        # VP8X stores 24-bit little-endian (width-1) and (height-1) after the 4-byte chunk header.
        import io

        assert isinstance(handle, io.BufferedReader)
        handle.seek(24)
        raw = handle.read(6)
        width = 1 + int.from_bytes(raw[0:3], "little")
        height = 1 + int.from_bytes(raw[3:6], "little")
        return ImageShape(width=width, height=height)
    if chunk == b"VP8 ":
        import io

        assert isinstance(handle, io.BufferedReader)
        handle.seek(26)
        raw = handle.read(4)
        width = int.from_bytes(raw[0:2], "little") & 0x3FFF
        height = int.from_bytes(raw[2:4], "little") & 0x3FFF
        return ImageShape(width=width, height=height)
    if chunk == b"VP8L":
        import io

        assert isinstance(handle, io.BufferedReader)
        handle.seek(21)
        raw = handle.read(4)
        bits = int.from_bytes(raw, "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return ImageShape(width=width, height=height)
    raise ValidationFailure("unrecognised WebP chunk layout", code=ErrorCode.MEDIA_UNSUPPORTED)


def _probe_jpeg(handle: object) -> ImageShape:
    """Walk JPEG segments until a start-of-frame marker is found."""
    import io

    assert isinstance(handle, io.BufferedReader)
    handle.seek(2)
    sof_markers = set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
    while True:
        byte = handle.read(1)
        if not byte:
            break
        if byte != b"\xff":
            continue
        marker = handle.read(1)
        while marker == b"\xff":  # padding bytes between segments are legal
            marker = handle.read(1)
        if not marker:
            break
        code = marker[0]
        if code in {0x01, *range(0xD0, 0xDA)}:
            continue
        raw_length = handle.read(2)
        if len(raw_length) < 2:
            break
        length = int.from_bytes(raw_length, "big")
        if code in sof_markers:
            body = handle.read(5)
            if len(body) < 5:
                break
            height = int.from_bytes(body[1:3], "big")
            width = int.from_bytes(body[3:5], "big")
            return ImageShape(width=width, height=height)
        handle.seek(length - 2, io.SEEK_CUR)
    raise ValidationFailure("JPEG file has no start-of-frame marker", code=ErrorCode.MEDIA_CORRUPT)


def probe_video_spec(path: Path) -> VideoSpec:
    """Read duration and timescale from an ISO base media (MP4) ``mvhd`` box.

    Frame rate and frame count require decoding the sample table, so they stay ``None`` until a
    video backend provides them; the model is explicit about the absence rather than guessing.
    """
    with path.open("rb") as handle:
        for box_type, payload in _iter_mp4_boxes(handle, limit=64):
            if box_type == b"mvhd":
                version = payload[0]
                if version == 1:
                    timescale = int.from_bytes(payload[20:24], "big")
                    duration = int.from_bytes(payload[24:32], "big")
                else:
                    timescale = int.from_bytes(payload[12:16], "big")
                    duration = int.from_bytes(payload[16:20], "big")
                if timescale <= 0:
                    break
                return VideoSpec(timescale=timescale, duration_s=duration / timescale)
    return VideoSpec()


def _iter_mp4_boxes(handle: object, limit: int) -> list[tuple[bytes, bytes]]:
    import io

    assert isinstance(handle, io.BufferedReader)
    boxes: list[tuple[bytes, bytes]] = []
    handle.seek(0)
    while len(boxes) < limit:
        header = handle.read(8)
        if len(header) < 8:
            break
        size = int.from_bytes(header[0:4], "big")
        box_type = header[4:8]
        if size == 1:
            extended = handle.read(8)
            if len(extended) < 8:
                break
            size = int.from_bytes(extended, "big")
        if size < 8:
            break
        payload = handle.read(min(size - 8, 256))
        boxes.append((box_type, payload))
        handle.seek(size - 8 - len(payload), io.SEEK_CUR)
    return boxes


def inspect_media(path: Path, *, relative_to: Path | None = None, hash_bytes: bool = True) -> MediaAsset:
    """Build a :class:`MediaAsset` for a file on disk.

    Args:
        path: file to inspect. It is opened read-only and never written.
        relative_to: when given, the stored ``path`` is expressed relative to this directory.
        hash_bytes: set ``False`` only for cheap probing passes where the digest is already known.

    Raises:
        ValidationFailure: when the file is missing, unsupported, or unreadable.
    """
    if not path.is_file():
        raise ValidationFailure(f"media file not found: {path}", code=ErrorCode.MEDIA_UNSUPPORTED)
    kind = media_kind_for_suffix(path.suffix)
    if kind is None:
        raise ValidationFailure(
            f"unsupported media extension {path.suffix!r} for {path.name}",
            code=ErrorCode.MEDIA_UNSUPPORTED,
        )
    stored_path = path.relative_to(relative_to).as_posix() if relative_to is not None else path.as_posix()
    digest = digest_file(path) if hash_bytes else "0" * 64
    size_bytes = path.stat().st_size
    if kind is MediaKind.IMAGE:
        shape = probe_image_shape(path)
        video = None
    else:
        shape = ImageShape(width=1, height=1)
        video = probe_video_spec(path)
    return MediaAsset(
        path=stored_path,
        kind=kind,
        digest=digest,
        size_bytes=size_bytes,
        media_type=media_type_for_suffix(path.suffix),
        shape=shape,
        video=video,
    )
