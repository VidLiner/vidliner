"""JSON task adapter for Runway and fal; uses the existing runtime backend lifecycle.

Model-specific duration, ratio, references and other fields are explicit parameters, not guessed
universal defaults. Explicit image artifacts may be sent inline; returned media is not downloaded.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Literal
from urllib.parse import quote, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vidliner.backends.base import LocalBackend
from vidliner.capabilities.backend import BackendProbe, PipelineContext
from vidliner.capabilities.names import CAP_VIDEO_GENERATION
from vidliner.core.errors import BackendFailure, ErrorCode, ValidationFailure
from vidliner.domain.enums import Determinism
from vidliner.domain.video_generation import VideoGenerationRequest, VideoTask
from vidliner.runtime.secrets import CredentialResolver


class VideoHttpOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    provider: Literal["fal", "runway"]
    model_id: str = Field(min_length=1)
    base_url: str | None = None
    modes: tuple[Literal["text-to-video", "image-to-video", "video-to-video"], ...] = ("text-to-video",)
    prompt_field: str = "prompt"
    image_field: str = "image_url"
    video_field: str = "video_url"
    seed_field: str | None = "seed"
    result_field: str = "video.url"
    request_timeout_s: float = Field(default=60, gt=0)
    max_response_mb: int = Field(default=2, ge=1, le=32)
    max_image_mb: int = Field(default=4, ge=1, le=5)
    runway_version: str = "2024-11-06"
    device: str = "remote"

    @model_validator(mode="after")
    def validate_mapping(self) -> VideoHttpOptions:
        """Reject mappings that can overwrite a core field or cannot locate a result."""
        fields = (
            ["promptText", "promptImage", self.video_field]
            if self.provider == "runway"
            else [self.prompt_field, self.image_field, self.video_field]
        )
        if self.seed_field is not None:
            fields.append(self.seed_field)
        if not self.modes or any(not field.strip() or field == "model" for field in fields):
            raise ValueError("modes and non-reserved field mappings must be nonempty")
        if len(set(fields)) != len(fields):
            raise ValueError("generation field mappings must be distinct")
        if any(not part.strip() for part in self.result_field.split(".")):
            raise ValueError("result_field must be a nonempty dotted path")
        return self


class HttpVideoGenerationBackend(LocalBackend):
    """Submit once, query/cancel by persisted task ID; no hidden polling or paid retries."""

    backend_version = "1.0.0"
    capabilities = (CAP_VIDEO_GENERATION,)
    declared_capabilities = capabilities
    determinism = Determinism.NONDETERMINISTIC
    blocking = False
    external = True
    safe_to_retry = False

    def __init__(self, spec: Any, options: dict[str, Any], credentials: Any = None) -> None:
        super().__init__(spec, options, credentials)
        self.config = VideoHttpOptions.model_validate(options)
        default = "https://queue.fal.run" if self.config.provider == "fal" else "https://api.dev.runwayml.com"
        self.base = (self.config.base_url or default).rstrip("/")
        parsed = urlparse(self.base)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) origin without credentials, query or fragment")
        if parsed.path:
            raise ValueError("base_url must not contain a path")
        self.backend_id = f"http_video:{self.config.provider}:{self.base}:{self.config.model_id}"
        if self.config.provider == "fal" and (
            len(self.config.model_id.split("/")) < 2
            or not all(
                part
                and part not in {".", ".."}
                and all(c.isascii() and (c.isalnum() or c in "_.-") for c in part)
                for part in self.config.model_id.split("/")
            )
        ):
            raise ValueError("fal model_id must be a valid model endpoint path")
        if self.config.provider == "runway" and "video-to-video" in self.config.modes:
            raise ValueError("Runway video-to-video is reserved for a dedicated adapter")

    def _key(self) -> str:
        ref = self.spec.credentials.get("api_key")
        resolver = self.credentials
        if ref is None or not isinstance(resolver, CredentialResolver):
            raise BackendFailure(
                "video backend requires credential 'api_key'",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_AUTH_FAILED,
            )
        try:
            key = resolver.resolve(ref, backend=self.backend_id, name="api_key")
        except ValidationFailure as exc:
            raise BackendFailure(
                "video backend credential is unavailable",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_AUTH_FAILED,
            ) from exc
        if not key:
            raise BackendFailure(
                "video backend credential is unavailable",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_AUTH_FAILED,
            )
        return key

    def _headers(self) -> dict[str, str]:
        if self.config.provider == "fal":
            return {"Authorization": f"Key {self._key()}", "X-Fal-No-Retry": "1"}
        return {"Authorization": f"Bearer {self._key()}", "X-Runway-Version": self.config.runway_version}

    async def probe(self) -> BackendProbe:
        """Check credentials and optional HTTP support without contacting the provider."""
        problems = []
        try:
            import httpx  # noqa: F401

            self._key()
        except (ImportError, BackendFailure):
            problems.append("httpx and a resolvable api_key are required")
        return BackendProbe(
            backend_id=self.backend_id,
            version=self.backend_version,
            health="unavailable" if problems else "ready",
            capabilities=self.capabilities,
            external=True,
            safe_to_retry=False,
            determinism=self.determinism,
            device="remote",
            message="; ".join(problems) or "configuration ready; no generation request made",
            details={
                "provider": self.config.provider,
                "model_id": self.config.model_id,
                "modes": self.config.modes,
                "probe_makes_no_request": True,
            },
        )

    async def _json(self, method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        import httpx

        try:
            async with (
                httpx.AsyncClient(timeout=self.config.request_timeout_s, follow_redirects=False) as client,
                client.stream(method, url, json=payload, headers=self._headers()) as response,
            ):
                if response.status_code >= 300:
                    code = (
                        ErrorCode.BACKEND_AUTH_FAILED
                        if response.status_code in {401, 403}
                        else ErrorCode.BACKEND_RATE_LIMITED
                        if response.status_code == 429
                        else ErrorCode.BACKEND_REQUEST_FAILED
                    )
                    raise BackendFailure(
                        f"video provider returned HTTP {response.status_code}",
                        backend_id=self.backend_id,
                        code=code,
                        safe_to_retry=False,
                    )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self.config.max_response_mb * 1024 * 1024:
                        raise BackendFailure(
                            "video task response exceeds configured cap",
                            backend_id=self.backend_id,
                            code=ErrorCode.BACKEND_RESPONSE_TOO_LARGE,
                        )
            if not body and method in {"DELETE", "PUT"}:
                return {}
            result = json.loads(body)
            if not isinstance(result, dict):
                raise ValueError("task response must be an object")
            return result
        except (httpx.HTTPError, ValueError) as exc:
            # Avoid exception text: transport errors may contain signed URLs or credentials.
            raise BackendFailure(
                "video task request failed or returned invalid JSON; submission outcome may be unknown",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_REQUEST_FAILED,
                safe_to_retry=False,
            ) from exc

    def _payload(self, request: VideoGenerationRequest, context: PipelineContext) -> dict[str, Any]:
        if request.mode not in self.config.modes:
            raise BackendFailure(
                f"model is not configured for {request.mode}",
                backend_id=self.backend_id,
                code=ErrorCode.VIDEO_NOT_SUPPORTED,
            )
        runway = self.config.provider == "runway"
        prompt_field = "promptText" if runway else self.config.prompt_field
        image_field = "promptImage" if runway else self.config.image_field
        video_field = self.config.video_field
        seed_field = self.config.seed_field
        reserved = {prompt_field, image_field, video_field, "model"}
        if seed_field:
            reserved.add(seed_field)
        if reserved.intersection(request.parameters):
            raise BackendFailure(
                "parameters must not override core generation fields",
                backend_id=self.backend_id,
                code=ErrorCode.VIDEO_NOT_SUPPORTED,
            )
        payload = dict(request.parameters)
        payload[prompt_field] = request.prompt
        if runway:
            payload["model"] = self.config.model_id
        if request.seed is not None:
            if seed_field is None:
                raise BackendFailure(
                    "this model does not support a seed field",
                    backend_id=self.backend_id,
                    code=ErrorCode.VIDEO_NOT_SUPPORTED,
                )
            payload[seed_field] = request.seed
        if request.image is not None:
            if request.image.media_type not in {"image/png", "image/jpeg", "image/webp"}:
                raise BackendFailure(
                    "source artifact must be an image",
                    backend_id=self.backend_id,
                    code=ErrorCode.VIDEO_NOT_SUPPORTED,
                )
            data = context.require_io().load(request.image)
            if len(data) > self.config.max_image_mb * 1024 * 1024:
                raise BackendFailure(
                    "source image exceeds inline upload cap",
                    backend_id=self.backend_id,
                    code=ErrorCode.BACKEND_RESPONSE_TOO_LARGE,
                )
            payload[image_field] = f"data:{request.image.media_type};base64," + base64.b64encode(data).decode(
                "ascii"
            )
        elif request.image_url is not None:
            payload[image_field] = str(request.image_url)
        if request.video_url is not None:
            payload[video_field] = str(request.video_url)
        return payload

    def _task_url(self, task: VideoTask, url: object, suffix: str = "") -> str:
        if task.backend_id != self.backend_id or task.model_id != self.config.model_id:
            raise BackendFailure(
                "task belongs to another backend/model",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_REQUEST_FAILED,
            )
        if not task.task_id or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in task.task_id
        ):
            raise BackendFailure(
                "invalid provider task id", backend_id=self.backend_id, code=ErrorCode.BACKEND_REQUEST_FAILED
            )
        if self.config.provider == "runway":
            return f"{self.base}/v1/tasks/{task.task_id}"
        parsed = urlparse(str(url))
        origin = urlparse(self.base)
        paths = {
            "/" + model + "/requests/" + task.task_id + suffix
            for model in {self.config.model_id, "/".join(self.config.model_id.split("/")[:2])}
        }
        if (
            (parsed.scheme, parsed.netloc) != (origin.scheme, origin.netloc)
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in paths
        ):
            raise BackendFailure(
                "untrusted task control URL",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_REQUEST_FAILED,
            )
        return str(url)

    async def submit(self, request: VideoGenerationRequest, context: PipelineContext) -> VideoTask:
        """Submit one intent and validate its durable control handle without polling."""
        payload = self._payload(request, context)
        if self.config.provider == "fal":
            endpoint = self.base + "/" + quote(self.config.model_id, safe="/")
        else:
            endpoint = self.base + (
                "/v1/image_to_video" if request.mode == "image-to-video" else "/v1/text_to_video"
            )
        document = await self._json("POST", endpoint, payload)
        try:
            if self.config.provider == "fal":
                task = VideoTask(
                    backend_id=self.backend_id,
                    model_id=self.config.model_id,
                    task_id=document["request_id"],
                    status_url=document["status_url"],
                    result_url=document["response_url"],
                    cancel_url=document["cancel_url"],
                )
                self._task_url(task, task.status_url, "/status")
                self._task_url(task, task.result_url)
                self._task_url(task, task.cancel_url, "/cancel")
                return task
            task = VideoTask(
                backend_id=self.backend_id, model_id=self.config.model_id, task_id=document["id"]
            )
            self._task_url(task, None)
            return task
        except (KeyError, ValidationError) as exc:
            raise BackendFailure(
                "provider returned an invalid submission handle; task may exist remotely",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_REQUEST_FAILED,
                safe_to_retry=False,
            ) from exc

    async def status(self, task: VideoTask, context: PipelineContext) -> VideoTask:
        """Read current state and obtain video URLs only after provider success."""
        del context
        url = self._task_url(task, task.status_url, "/status")
        document = await self._json("GET", url)
        mapping = {
            "IN_QUEUE": "queued",
            "IN_PROGRESS": "running",
            "COMPLETED": "succeeded",
            "PENDING": "queued",
            "THROTTLED": "queued",
            "RUNNING": "running",
            "SUCCEEDED": "succeeded",
            "FAILED": "failed",
            "CANCELLED": "cancelled",
        }
        state = mapping.get(str(document.get("status")), "unknown")
        outputs: object = ()
        code = document.get("failureCode")
        if state == "succeeded":
            if self.config.provider == "fal":
                if document.get("error"):
                    state, code = "failed", str(document.get("error_type") or "provider_error")
                else:
                    result = await self._json("GET", self._task_url(task, task.result_url))
                    value: Any = result
                    for key in self.config.result_field.split("."):
                        value = value.get(key) if isinstance(value, dict) else None
                    outputs = [value] if isinstance(value, str) else []
            else:
                outputs = document.get("output")
            if state == "succeeded" and (not isinstance(outputs, list) or not outputs):
                raise BackendFailure(
                    "successful task has no video outputs",
                    backend_id=self.backend_id,
                    code=ErrorCode.BACKEND_REQUEST_FAILED,
                )
        try:
            return VideoTask.model_validate(
                {**task.model_dump(), "status": state, "output_urls": outputs or (), "failure_code": code}
            )
        except ValidationError as exc:
            raise BackendFailure(
                "invalid video task result", backend_id=self.backend_id, code=ErrorCode.BACKEND_REQUEST_FAILED
            ) from exc

    async def cancel(self, task: VideoTask, context: PipelineContext) -> VideoTask:
        """Request cancellation of an ongoing task; preserve finished evidence."""
        current = await self.status(task, context)
        if current.terminal:
            return current  # Runway DELETE also deletes finished tasks; never delete completed evidence.
        url = self._task_url(task, task.cancel_url, "/cancel")
        await self._json("PUT" if self.config.provider == "fal" else "DELETE", url)
        return current  # Cancellation acknowledgement is not a terminal-state guarantee.
