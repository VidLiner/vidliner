"""Video providers use official-shaped HTTP fixtures, never paid calls."""

import json
from pathlib import Path

import pytest

from vidliner.backends.http_video import HttpVideoGenerationBackend
from vidliner.capabilities.backend import PipelineContext
from vidliner.core.errors import BackendFailure
from vidliner.domain.video_generation import VideoGenerationRequest, VideoTask
from vidliner.operators.values import decode_value, encode_value
from vidliner.runtime.profile import BackendSpec, CredentialRef, load_profile
from vidliner.runtime.registry import BackendRegistry
from vidliner.runtime.secrets import CredentialResolver

httpx = pytest.importorskip("httpx")
CONTEXT = PipelineContext(job_id="job", node_id="node", seed=1)


def backend(provider="fal", **options):
    settings = {
        "provider": provider,
        "model_id": "fal-ai/kling-video/v2.5-turbo/pro/image-to-video" if provider == "fal" else "gen4.5",
        **options,
    }
    spec = BackendSpec(
        use="vidliner.backends.http_video:HttpVideoGenerationBackend",
        options=settings,
        credentials={"api_key": CredentialRef(source="env", name="VIDEO_KEY")},
    )
    return HttpVideoGenerationBackend(
        spec, settings, CredentialResolver(environ={"VIDEO_KEY": "test-secret-never-log"})
    )


def mock_http(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )


def fal_handle():
    root = "https://queue.fal.run/fal-ai/kling-video/requests/task-1"
    return {
        "request_id": "task-1",
        "status_url": root + "/status",
        "response_url": root,
        "cancel_url": root + "/cancel",
    }


@pytest.mark.asyncio
async def test_fal_submit_status_serialization_and_result(monkeypatch):
    calls = []

    def handle(request):
        calls.append((request.method, str(request.url)))
        assert request.headers["Authorization"] == "Key test-secret-never-log"
        if request.method == "POST":
            payload = json.loads(request.content)
            assert payload == {
                "prompt": "move slowly",
                "image_url": "https://assets.example/image.png",
                "duration": "5",
            }
            assert request.headers["X-Fal-No-Retry"] == "1"
            return httpx.Response(200, json=fal_handle())
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"status": "COMPLETED"})
        return httpx.Response(200, json={"video": {"url": "https://media.example/output.mp4"}})

    mock_http(monkeypatch, handle)
    adapter = backend(modes=["image-to-video"])
    task = await adapter.submit(
        VideoGenerationRequest(
            mode="image-to-video",
            prompt="move slowly",
            image_url="https://assets.example/image.png",
            parameters={"duration": "5"},
        ),
        CONTEXT,
    )
    restored = decode_value(encode_value(task))
    assert restored == task
    result = await adapter.status(restored, CONTEXT)
    assert result.status == "succeeded" and not result.verified
    assert result.label_transform.semantics == "synthesized"
    assert str(result.output_urls[0]) == "https://media.example/output.mp4"
    assert len(calls) == 3 and sum(method == "POST" for method, _ in calls) == 1
    assert "test-secret-never-log" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_runway_payload_status_and_cancellation_204(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request.method)
        assert request.headers["X-Runway-Version"] == "2024-11-06"
        if request.method == "POST":
            assert json.loads(request.content) == {
                "promptText": "a landscape",
                "model": "gen4.5",
                "duration": 5,
                "ratio": "1280:720",
                "seed": 3,
            }
            return httpx.Response(200, json={"id": "task-1"})
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"status": "RUNNING"})

    mock_http(monkeypatch, handle)
    adapter = backend("runway")
    task = await adapter.submit(
        VideoGenerationRequest(prompt="a landscape", seed=3, parameters={"duration": 5, "ratio": "1280:720"}),
        CONTEXT,
    )
    current = await adapter.cancel(task, CONTEXT)
    assert current.status == "running"
    assert calls == ["POST", "GET", "DELETE"]


@pytest.mark.asyncio
async def test_completed_runway_task_is_never_deleted(monkeypatch):
    def handle(request):
        assert request.method == "GET"
        return httpx.Response(
            200, json={"status": "SUCCEEDED", "output": ["https://media.example/video.mp4"]}
        )

    mock_http(monkeypatch, handle)
    adapter = backend("runway")
    task = VideoTask(backend_id=adapter.backend_id, model_id="gen4.5", task_id="task-1")
    assert (await adapter.cancel(task, CONTEXT)).status == "succeeded"


@pytest.mark.asyncio
async def test_probe_no_network_and_foreign_control_url_rejected(monkeypatch):
    def forbidden(request):
        raise AssertionError("network must not be reached")

    mock_http(monkeypatch, forbidden)
    adapter = backend()
    assert (await adapter.probe()).is_ready
    task = VideoTask(
        backend_id=adapter.backend_id,
        model_id=adapter.config.model_id,
        task_id="task-1",
        status_url="https://foreign.example/requests/task-1/status",
    )
    with pytest.raises(BackendFailure, match="untrusted"):
        await adapter.status(task, CONTEXT)
    with pytest.raises(BackendFailure, match="override"):
        await adapter.submit(VideoGenerationRequest(prompt="x", parameters={"prompt": "override"}), CONTEXT)


@pytest.mark.asyncio
async def test_timeout_never_retries_and_redacts_transport_detail(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("test-secret-never-log in URL")

    mock_http(monkeypatch, handle)
    adapter = backend("runway")
    with pytest.raises(BackendFailure) as error:
        await adapter.submit(VideoGenerationRequest(prompt="x"), CONTEXT)
    assert not error.value.safe_to_retry and len(calls) == 1
    assert "test-secret-never-log" not in str(error.value)


@pytest.mark.asyncio
async def test_response_cap(monkeypatch):
    adapter = backend("runway", max_response_mb=1)
    task = VideoTask(backend_id=adapter.backend_id, model_id="gen4.5", task_id="task-1")
    mock_http(monkeypatch, lambda request: httpx.Response(200, content=b"x" * (1024 * 1024 + 1)))
    with pytest.raises(BackendFailure, match="cap"):
        await adapter.status(task, CONTEXT)


@pytest.mark.asyncio
async def test_task_identity_includes_provider_origin(monkeypatch):
    def forbidden(request):
        raise AssertionError("foreign task must fail before network access")

    mock_http(monkeypatch, forbidden)
    original = backend("runway")
    proxy = backend("runway", base_url="https://proxy.example")
    task = VideoTask(backend_id=original.backend_id, model_id="gen4.5", task_id="task-1")
    with pytest.raises(BackendFailure, match="another backend"):
        await proxy.status(task, CONTEXT)


@pytest.mark.asyncio
async def test_example_profile_resolves_and_probes_without_provider_calls(monkeypatch):
    mock_http(monkeypatch, lambda request: pytest.fail("configuration must not generate videos"))
    profile = load_profile(Path(__file__).parents[2] / "examples/video/runtime-ai.yaml")
    registry = BackendRegistry(
        profile, CredentialResolver(environ={"FAL_KEY": "fixture", "RUNWAYML_API_SECRET": "fixture"})
    )
    assert (
        registry.resolve(("generation.video_generation.v1",)).backend_for("generation.video_generation.v1")
        == "wan"
    )
    for name in profile.backends:
        assert (await registry.backend(name).probe()).is_ready


@pytest.mark.parametrize(
    "options",
    [
        {"prompt_field": "image_url"},
        {"modes": []},
        {"result_field": "video..url"},
        {"provider": "runway", "seed_field": "promptText"},
    ],
)
def test_invalid_field_mappings_are_rejected(options):
    with pytest.raises(ValueError):
        backend(**options)
