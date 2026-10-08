"""Deterministic local video generation built on the FFmpeg command line.

The renderer is intentionally narrow: it turns the existing declarative variant sampler into real
video files without pretending that a local pixel transform is a semantic video model. Each output
gets a manifest containing the exact source, variant and command-independent media measurements.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vidliner.core.canonical import digest_file
from vidliner.core.errors import ErrorCode, ValidationFailure
from vidliner.domain.video import (
    LabelTransform,
    PlacementTransform,
    TimeTransform,
    VideoAugmentSpec,
    VideoOperation,
    VideoVariant,
    enumerate_video_variants,
)

__all__ = ["RenderResult", "VideoProbe", "generate_variants", "probe_video", "render_variant"]


@dataclass(frozen=True, slots=True)
class VideoProbe:
    """Stable media properties read from one input video."""

    path: Path
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool
    codec: str | None


@dataclass(frozen=True, slots=True)
class RenderResult:
    """One rendered variant and its measured output."""

    variant_id: str
    input_path: Path
    output_path: Path
    input_digest: str
    output_digest: str
    operations: tuple[VideoOperation, ...]
    probe: VideoProbe
    transform: LabelTransform

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-safe provenance record."""
        return {
            "variant_id": self.variant_id,
            "input": str(self.input_path),
            "output": str(self.output_path),
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "operations": [operation.model_dump(mode="json") for operation in self.operations],
            "probe": {
                "duration_s": self.probe.duration_s,
                "width": self.probe.width,
                "height": self.probe.height,
                "fps": self.probe.fps,
                "has_audio": self.probe.has_audio,
                "codec": self.probe.codec,
            },
            "label_transform": self.transform.model_dump(mode="json"),
        }


def _tool(name: str) -> str:
    """Resolve a required media executable from PATH."""
    import shutil

    path = shutil.which(name)
    if path is None:
        raise ValidationFailure(f"{name} is required for video rendering", code=ErrorCode.VIDEO_NOT_SUPPORTED)
    return path


def probe_video(path: Path) -> VideoProbe:
    """Read dimensions, duration, frame rate and audio presence without decoding frames."""
    if not path.is_file():
        raise ValidationFailure(f"video input {path} does not exist", code=ErrorCode.ARTIFACT_MISSING)
    command = [
        _tool("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as exc:
        raise ValidationFailure(f"cannot probe video {path}: {exc}", code=ErrorCode.MEDIA_CORRUPT) from exc
    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not isinstance(video, dict):
        raise ValidationFailure(f"video {path} has no video stream", code=ErrorCode.MEDIA_CORRUPT)
    duration = float((payload.get("format") or {}).get("duration") or 0.0)
    fps = _rate(video.get("avg_frame_rate"))
    if int(video.get("width") or 0) <= 0 or int(video.get("height") or 0) <= 0:
        raise ValidationFailure(f"video {path} has invalid dimensions", code=ErrorCode.MEDIA_CORRUPT)
    return VideoProbe(
        path=path,
        duration_s=round(max(0.0, duration), 6),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
        codec=str(video.get("codec_name")) if video.get("codec_name") else None,
    )


def _rate(value: object) -> float:
    if not isinstance(value, str) or "/" not in value:
        return 0.0
    numerator, denominator = value.split("/", 1)
    try:
        return round(float(numerator) / float(denominator), 6) if float(denominator) else 0.0
    except ValueError:
        return 0.0


def _number(params: dict[str, str | int | float | bool], key: str, default: float) -> float:
    try:
        value = float(params.get(key, default))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationFailure(f"{key} must be a finite number", code=ErrorCode.VIDEO_NOT_SUPPORTED) from exc
    if not math.isfinite(value):
        raise ValidationFailure(f"{key} must be a finite number", code=ErrorCode.VIDEO_NOT_SUPPORTED)
    return value


def _aspect_size(aspect: str) -> tuple[int, int]:
    sizes = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (720, 720), "4:3": (960, 720)}
    try:
        return sizes[aspect]
    except KeyError as exc:
        raise ValidationFailure(
            f"unsupported reframe aspect {aspect!r}; use one of {', '.join(sorted(sizes))}",
            code=ErrorCode.VIDEO_NOT_SUPPORTED,
        ) from exc


def _reframe_geometry(source: VideoProbe, aspect: str) -> tuple[int, int, int, int, int, int]:
    width, height = _aspect_size(aspect)
    fit = min(width / source.width, height / source.height)
    content_width = max(2, math.floor(source.width * fit / 2) * 2)
    content_height = max(2, math.floor(source.height * fit / 2) * 2)
    x = math.floor((width - content_width) / 4) * 2
    y = math.floor((height - content_height) / 4) * 2
    return width, height, content_width, content_height, x, y


def _label_transform(source: VideoProbe, operations: tuple[VideoOperation, ...]) -> LabelTransform:
    """Describe the pixel and clock mapping produced by the local renderer."""
    placement = PlacementTransform()
    scale = 1.0
    offset = 0.0
    for operation in operations:
        params = operation.params
        if operation.op == "format.reframe":
            aspect = params.get("aspect", "16:9")
            if not isinstance(aspect, str):
                raise ValidationFailure("reframe aspect must be a string", code=ErrorCode.VIDEO_NOT_SUPPORTED)
            width, height, content_width, content_height, x, y = _reframe_geometry(source, aspect)
            placement = PlacementTransform(
                x=x / width,
                y=y / height,
                w=content_width / width,
                h=content_height / height,
            )
        elif operation.op in {"temporal.speed", "time.speed"}:
            factor = _number(params, "factor", 1.0)
            if factor <= 0:
                raise ValidationFailure("speed factor must be positive", code=ErrorCode.VIDEO_NOT_SUPPORTED)
            scale /= factor
            offset /= factor
        elif operation.op in {"temporal.trim", "time.trim"}:
            start = max(0.0, _number(params, "start", 0.0))
            offset -= start
    return LabelTransform(
        placement=placement,
        time=TimeTransform(scale=scale, offset=offset),
        semantics="edited"
        if any(op.op in {"audio.mute", "sound.mute"} for op in operations)
        else "preserved",
        identity={"renderer": "ffmpeg"},
        text={"label_policy": "project labels and clip to output bounds before verification"},
    )


def _atempo(factor: float) -> str:
    if factor <= 0:
        raise ValidationFailure("speed factor must be positive", code=ErrorCode.VIDEO_NOT_SUPPORTED)
    filters: list[str] = []
    remaining = factor
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.8g}")
    return ",".join(filters)


def _filters(
    operations: tuple[VideoOperation, ...],
    source: VideoProbe,
) -> tuple[list[str], list[str], bool]:
    video: list[str] = []
    audio: list[str] = []
    mute = False
    reframes = 0
    for operation in operations:
        if operation.kind != "local":
            raise ValidationFailure(
                "generative operations require a model backend", code=ErrorCode.VIDEO_NOT_SUPPORTED
            )
        if any(isinstance(value, float) and not math.isfinite(value) for value in operation.params.values()):
            raise ValidationFailure("render parameters must be finite", code=ErrorCode.VIDEO_NOT_SUPPORTED)
        params = operation.params
        if operation.op == "format.reframe":
            reframes += 1
            if reframes > 1:
                raise ValidationFailure(
                    "only one reframe per variant is supported", code=ErrorCode.VIDEO_NOT_SUPPORTED
                )
            value = params.get("aspect", "16:9")
            if not isinstance(value, str):
                raise ValidationFailure("reframe aspect must be a string", code=ErrorCode.VIDEO_NOT_SUPPORTED)
            width, height, content_width, content_height, x, y = _reframe_geometry(source, value)
            video.append(f"scale={content_width}:{content_height},pad={width}:{height}:{x}:{y},setsar=1")
        elif operation.op in {"temporal.speed", "time.speed"}:
            factor = _number(params, "factor", 1.0)
            video.append(f"setpts=PTS/{factor:.8g}")
            audio.append(_atempo(factor))
        elif operation.op in {"temporal.trim", "time.trim"}:
            start = max(0.0, _number(params, "start", 0.0))
            end_value = params.get("end")
            end = _number(params, "end", 0.0) if end_value is not None else None
            if end is not None and end <= start:
                raise ValidationFailure("trim end must be after start", code=ErrorCode.VIDEO_NOT_SUPPORTED)
            bounds = f"start={start:.8g}" + (f":end={end:.8g}" if end is not None else "")
            video.extend([f"trim={bounds}", "setpts=PTS-STARTPTS"])
            audio.extend([f"atrim={bounds}", "asetpts=PTS-STARTPTS"])
        elif operation.op == "appearance.grade":
            saturation = _number(params, "saturation", 1.0)
            brightness = _number(params, "brightness", 0.0)
            video.append(f"eq=saturation={saturation:.8g}:brightness={brightness:.8g}")
        elif operation.op in {"audio.mute", "sound.mute"}:
            mute = True
        else:
            raise ValidationFailure(
                f"renderer does not implement video operation {operation.op!r}",
                code=ErrorCode.VIDEO_NOT_SUPPORTED,
            )
    return video, audio, mute


def render_variant(
    input_path: Path, output_path: Path, variant: VideoVariant, *, overwrite: bool = False
) -> RenderResult:
    """Render one sampled variant through a safe, argument-vector FFmpeg invocation."""
    source = probe_video(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValidationFailure("output must not overwrite the source video", code=ErrorCode.EXPORT_INVALID)
    if output_path.exists() and not overwrite:
        raise ValidationFailure(
            f"video output {output_path} already exists; pass --force", code=ErrorCode.EXPORT_INVALID
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_filters, audio_filters, mute = _filters(variant.ops, source)
    command = [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n"]
    command += ["-i", str(input_path)]
    if video_filters:
        command += ["-vf", ",".join(video_filters)]
    if audio_filters and not mute and source.has_audio:
        command += ["-af", ",".join(audio_filters)]
    command += ["-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    command += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart"]
    if mute:
        command += ["-an"]
    command += [str(output_path)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", "")[-1000:] if hasattr(exc, "stderr") else ""
        raise ValidationFailure(
            f"FFmpeg could not render {variant.id}: {detail or exc}", code=ErrorCode.MEDIA_CORRUPT
        ) from exc
    rendered = probe_video(output_path)
    return RenderResult(
        variant_id=variant.id,
        input_path=input_path.resolve(),
        output_path=output_path.resolve(),
        input_digest=digest_file(input_path),
        output_digest=digest_file(output_path),
        operations=variant.ops,
        probe=rendered,
        transform=_label_transform(source, variant.ops),
    )


def generate_variants(
    input_path: Path,
    output_dir: Path,
    spec: VideoAugmentSpec,
    *,
    overwrite: bool = False,
) -> tuple[RenderResult, ...]:
    """Sample and render all variants, writing one ``manifest.json`` beside the outputs."""
    variants = enumerate_video_variants(spec)
    source = probe_video(input_path)
    # Validate the whole plan before producing an identity output or any partial variant set.
    for variant in variants:
        _filters(variant.ops, source)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise ValidationFailure(
            f"manifest {manifest_path} already exists; pass --force", code=ErrorCode.EXPORT_INVALID
        )
    results = tuple(
        render_variant(input_path, output_dir / f"{variant.slug}.mp4", variant, overwrite=overwrite)
        for variant in variants
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "vidliner/video-render@1",
        "spec": spec.model_dump(mode="json"),
        "input_digest": digest_file(input_path),
        "variants": [result.manifest() for result in results],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return results
