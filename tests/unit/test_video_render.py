"""FFmpeg-backed local video generation tests."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vidliner.cli.main import app
from vidliner.core.errors import ValidationFailure
from vidliner.domain.video import (
    VideoAugmentSpec,
    VideoOperation,
    VideoOperatorSpec,
    VideoVariant,
    enumerate_video_variants,
)
from vidliner.reports.video import render_video_review
from vidliner.video.render import generate_variants, probe_video, render_variant

pytestmark = pytest.mark.slow


@pytest.fixture
def source_video(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")
    path = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x180:rate=10",
            "-t",
            "0.8",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def _spec() -> VideoAugmentSpec:
    return VideoAugmentSpec(
        id="demo",
        seed=42,
        strategy="paired",
        operators=(
            VideoOperatorSpec(name="format.reframe", axis="format", domains={"aspect": ("1:1",)}),
            VideoOperatorSpec(name="appearance.grade", axis="style", domains={"saturation": (0.8,)}),
        ),
    )


def test_probe_and_render_variant(source_video: Path, tmp_path: Path) -> None:
    source = probe_video(source_video)
    assert source.width == 320 and source.height == 180
    variants = enumerate_video_variants(_spec())
    assert len(variants) == 3
    output = tmp_path / "square.mp4"
    result = render_variant(source_video, output, variants[1])
    assert result.output_path == output
    assert result.output_digest and result.input_digest
    assert result.probe.width == 720 and result.probe.height == 720
    assert result.transform.placement.y == pytest.approx(158 / 720)
    assert result.transform.placement.h == pytest.approx(404 / 720)
    assert result.manifest()["label_transform"]["identity"]["renderer"] == "ffmpeg"
    assert probe_video(output).duration_s > 0


def test_generate_writes_reproducible_manifest(source_video: Path, tmp_path: Path) -> None:
    results = generate_variants(source_video, tmp_path / "generated", _spec())
    manifest_path = tmp_path / "generated" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(results) == len(manifest["variants"]) == 3
    assert manifest["format"] == "vidliner/video-render@1"
    assert all(Path(item["output"]).is_file() for item in manifest["variants"])


def test_trim_then_speed_clock_and_duration(source_video: Path, tmp_path: Path) -> None:
    variant = VideoVariant(
        id="clock",
        slug="clock",
        seed=0,
        variant_seed=0,
        ops=(
            VideoOperation(
                op="temporal.trim",
                axis="time",
                kind="local",
                label_class="preserving",
                params={"start": 0.2, "end": 0.6},
            ),
            VideoOperation(
                op="temporal.speed",
                axis="tempo",
                kind="local",
                label_class="preserving",
                params={"factor": 2},
            ),
        ),
    )
    result = render_variant(source_video, tmp_path / "clock.mp4", variant)
    assert result.transform.time.map(0.2) == pytest.approx(0)
    assert result.transform.time.map(0.6) == pytest.approx(0.2)
    assert result.probe.duration_s == pytest.approx(0.2, abs=0.1)


def test_source_and_existing_output_are_protected(source_video: Path) -> None:
    variant = enumerate_video_variants(_spec())[0]
    with pytest.raises(ValidationFailure, match="source video"):
        render_variant(source_video, source_video, variant, overwrite=True)


@pytest.mark.parametrize("params,kind", [({"factor": "invalid"}, "local"), ({"factor": 1.0}, "generative")])
def test_invalid_plan_fails_before_rendering(
    source_video: Path, tmp_path: Path, params: dict, kind: str
) -> None:
    spec = VideoAugmentSpec.model_validate(
        {
            "id": "invalid",
            "seed": 0,
            "operators": [
                {
                    "name": "temporal.speed",
                    "axis": "tempo",
                    "kind": kind,
                    "domains": {key: [value] for key, value in params.items()},
                }
            ],
        }
    )
    target = tmp_path / "invalid"
    with pytest.raises(ValidationFailure):
        generate_variants(source_video, target, spec)
    assert not target.exists()


def test_public_cli_probe_render_generate(source_video: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    spec = tmp_path / "spec.json"
    spec.write_text(_spec().model_dump_json())
    probe = runner.invoke(app, ["video", "probe", str(source_video), "--json"])
    assert probe.exit_code == 0, probe.output
    assert json.loads(probe.output)["width"] == 320
    render = runner.invoke(
        app,
        [
            "video",
            "render",
            str(spec),
            "-i",
            str(source_video),
            "-o",
            str(tmp_path / "one.mp4"),
            "--variant",
            "1",
            "--json",
        ],
    )
    assert render.exit_code == 0, render.output
    assert json.loads(render.output)["probe"]["width"] == 720
    arguments = [
        "video",
        "generate",
        str(spec),
        "-i",
        str(source_video),
        "-o",
        str(tmp_path / "variants"),
        "--json",
    ]
    generated = runner.invoke(app, arguments)
    assert generated.exit_code == 0, generated.output
    assert len(json.loads(generated.output)["variants"]) == 3
    assert runner.invoke(app, arguments).exit_code == 2
    preview = tmp_path / "review.html"
    result = runner.invoke(
        app, ["video", "preview", str(tmp_path / "variants/manifest.json"), "-o", str(preview)]
    )
    assert result.exit_code == 0, result.output
    assert '<video id="source"' in preview.read_text()
    assert (
        runner.invoke(
            app,
            [
                "video",
                "preview",
                str(tmp_path / "variants/manifest.json"),
                "-o",
                str(source_video),
                "--force",
            ],
        ).exit_code
        == 2
    )


def test_video_review_strings_are_inert(source_video: Path, tmp_path: Path) -> None:
    malicious = '</script><img src=x onerror="alert(1)">'
    page = render_video_review(
        {
            "format": "vidliner/video-render@1",
            "variants": [
                {
                    "variant_id": malicious,
                    "input": str(source_video),
                    "output": str(source_video),
                }
            ],
        },
        tmp_path / "review.html",
    )
    assert malicious not in page
    payload = page.split('<script type="application/json" id="variants">')[1].split("</script>")[0]
    assert json.loads(payload)[0]["variant_id"] == malicious
