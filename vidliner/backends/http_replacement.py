"""Generic HTTP image-edit backend.

This adapter talks to *any* service that accepts a JSON request containing a base64 image plus a
prompt and returns a JSON body containing a base64 image. That covers a local inference server
wrapping a diffusion model, a self-hosted ComfyUI bridge, and most hosted editing APIs with a thin
shim in front of them.

The request and response shapes are configurable, because no single wire format is universal:

```yaml
http_replacement:
  use: vidliner.backends.http_replacement:HttpReplacementBackend
  options:
    endpoint: http://127.0.0.1:8080/v1/edit
    request_image_field: image_base64
    request_mask_field: mask_base64
    response_image_field: image_base64
    response_image_encoding: base64        # base64 | url
    extra_fields: { model: my-editor-v2 }
  credentials:
    api_key: { source: env, name: VIDLINER_EDIT_API_KEY }
```

Nothing in the core knows this class exists; it is bound to
``generation.object_replacement.v1`` through the runtime profile.

Safety properties enforced here, because the response is untrusted input:

* the response body is capped (``max_response_mb``);
* the declared media type must be in ``allowed_media_types`` and must match the decoded bytes;
* decoded dimensions are validated against the store's guard;
* the image is written through the artifact store, which validates size and dimensions again;
* the credential is sent as a header and never logged or persisted.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
from typing import Any
from urllib.parse import urlparse

from vidliner.backends.base import LocalBackend, elapsed_ms, timed
from vidliner.capabilities.backend import (
    ArtifactIO,
    BackendProbe,
    GenerationOutcome,
    GenerationRequest,
    PipelineContext,
)
from vidliner.capabilities.names import CAP_OBJECT_REPLACEMENT
from vidliner.core.errors import BackendFailure, ErrorCode
from vidliner.core.results import ArtifactRef
from vidliner.domain.enums import ArtifactKind, Determinism
from vidliner.runtime.secrets import CredentialResolver

__all__ = ["HttpReplacementBackend"]

_ALLOWED_SCHEMES = {"http", "https"}
_MAGIC_PREFIXES: dict[str, tuple[bytes, ...]] = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/webp": (b"RIFF",),
}


class HttpReplacementBackend(LocalBackend):
    """Call a remote image-edit service over JSON + base64.

    This backend is *not* blocking: the work is network-bound, so it is awaited directly and its
    capacity is governed by the runtime profile's per-backend limit and period.
    """

    backend_id = "http_replacement"
    backend_version = "1.0.0"
    capabilities = (CAP_OBJECT_REPLACEMENT,)
    determinism = Determinism.NONDETERMINISTIC
    safe_to_retry = False
    external = True
    blocking = False
    declared_capabilities = (CAP_OBJECT_REPLACEMENT,)

    def __init__(
        self, spec: Any, options: dict[str, Any], credentials: CredentialResolver | None = None
    ) -> None:
        super().__init__(spec, options, credentials)
        endpoint = str(self.option("endpoint", ""))
        parsed = urlparse(endpoint)
        if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.netloc:
            raise ValueError(
                f"backend {self.backend_id} requires an http(s) endpoint option, got {endpoint!r}"
            )
        self._endpoint = endpoint
        self._timeout = float(self.option("request_timeout_s", 120.0))
        self._max_bytes = int(self.option("max_response_mb", 32)) * 1024 * 1024
        allowed = self.option("allowed_media_types", ["image/png", "image/jpeg", "image/webp"])
        self._allowed_media = tuple(str(item) for item in allowed)
        self._api_key: str | None = None

    # -- protocol ---------------------------------------------------------- #

    async def replace(self, request: GenerationRequest, context: PipelineContext) -> GenerationOutcome:
        """Send one edit request and validate the returned image."""
        import httpx  # imported here so the core install does not require the dependency

        started = timed()
        io = context.require_io()
        payload = self._payload(request, io)
        headers = {"content-type": "application/json", "accept": "application/json"}
        api_key = self._credential("api_key")
        if api_key is not None:
            header_name = str(self.option("auth_header", "authorization"))
            prefix = str(self.option("auth_prefix", "Bearer "))
            headers[header_name] = f"{prefix}{api_key}" if prefix else api_key

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._endpoint, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise BackendFailure(
                f"{self.backend_id} timed out after {self._timeout}s",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_REQUEST_FAILED,
                safe_to_retry=False,
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendFailure(
                f"{self.backend_id} request failed: {exc}",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_REQUEST_FAILED,
                safe_to_retry=True,
            ) from exc

        if response.status_code == 401 or response.status_code == 403:
            raise BackendFailure(
                f"{self.backend_id} rejected the credential (HTTP {response.status_code})",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_AUTH_FAILED,
            )
        if response.status_code == 429:
            raise BackendFailure(
                f"{self.backend_id} is rate limited (HTTP 429)",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_RATE_LIMITED,
                safe_to_retry=True,
            )
        if response.status_code >= 400:
            raise BackendFailure(
                f"{self.backend_id} returned HTTP {response.status_code}",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_REQUEST_FAILED,
                safe_to_retry=response.status_code >= 500,
            )
        body = response.content
        if len(body) > self._max_bytes:
            raise BackendFailure(
                f"{self.backend_id} returned {len(body)} bytes, above the {self._max_bytes} byte cap",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_RESPONSE_TOO_LARGE,
            )
        document = self._parse(body)
        image_bytes, declared_media = self._extract_image(document)
        artifact = self._store_image(io, image_bytes, declared_media)
        return GenerationOutcome(
            image=artifact,
            backend_id=self.backend_id,
            model_id=str(document.get("model") or self.model_id()),
            seed=request.seed,
            duration_ms=elapsed_ms(started),
            cost_estimate=_optional_float(document.get("cost")),
            currency=document.get("currency") if isinstance(document.get("currency"), str) else None,
            external=True,
            raw_metadata={
                "status": response.status_code,
                "service_keys": sorted(str(key) for key in document),
                "declared_media_type": declared_media,
                "endpoint_host": urlparse(self._endpoint).netloc,
            },
            applied_parameters={"intensity": request.intensity},
        )

    async def probe(self) -> BackendProbe:
        """Report readiness from configuration and credential presence, without calling the service."""
        problems: list[str] = []
        if self._requires_auth():
            try:
                if self._credential("api_key", required=False) is None:
                    problems.append("credential 'api_key' is configured but could not be resolved")
            except Exception as exc:
                problems.append(f"credential 'api_key' could not be resolved: {exc}")
        try:
            import httpx  # noqa: F401
        except ImportError:
            problems.append("the 'httpx' package is not installed")
        health = "unavailable" if problems else "ready"
        return BackendProbe(
            backend_id=self.backend_id,
            version=self.backend_version,
            health=health,
            capabilities=self.capabilities,
            device="remote",
            determinism=self.determinism,
            safe_to_retry=self.safe_to_retry,
            external=True,
            message="; ".join(problems) or f"endpoint {urlparse(self._endpoint).netloc}",
            details={"endpoint_host": urlparse(self._endpoint).netloc, "probe_makes_no_request": True},
        )

    # -- internals --------------------------------------------------------- #

    def _requires_auth(self) -> bool:
        return "api_key" in (self.spec.credentials or {})

    def _credential(self, name: str, *, required: bool = True) -> str | None:
        """Resolve a named credential reference, or ``None`` when it is not configured."""
        reference = self.spec.credentials.get(name)
        if reference is None:
            return None
        resolver = self.credentials
        if not isinstance(resolver, CredentialResolver):
            return None
        if not required and not getattr(reference, "required", True):
            return None
        try:
            value = resolver.resolve(reference, backend=self.backend_id, name=name)
        except Exception:
            if required:
                raise
            return None
        if name == "api_key":
            self._api_key = value
        return value

    def _payload(self, request: GenerationRequest, io: ArtifactIO) -> dict[str, Any]:
        image_bytes = io.load(request.source_image)
        mask_bytes = io.load(request.target_mask)
        payload: dict[str, Any] = {
            str(self.option("request_image_field", "image_base64")): base64.b64encode(image_bytes).decode(
                "ascii"
            ),
            str(self.option("request_mask_field", "mask_base64")): base64.b64encode(mask_bytes).decode(
                "ascii"
            ),
            str(self.option("request_prompt_field", "prompt")): request.positive_prompt,
            str(self.option("request_negative_field", "negative_prompt")): ", ".join(
                request.negative_constraints
            ),
            "seed": request.seed,
            "candidate_key": request.candidate_key,
            "expected_category": request.expected_category,
            "intensity": request.intensity,
            "image_media_type": request.source_image.media_type,
            "mask_media_type": request.target_mask.media_type,
        }
        reference = request.reference_image
        if reference is not None:
            reference_bytes = io.load(reference)
            payload[str(self.option("request_reference_field", "reference_base64"))] = base64.b64encode(
                reference_bytes
            ).decode("ascii")
        extra = self.option("extra_fields")
        if isinstance(extra, dict):
            payload.update(extra)
        return payload

    def _parse(self, body: bytes) -> dict[str, Any]:
        try:
            document = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendFailure(
                f"{self.backend_id} returned a body that is not JSON",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_RESPONSE_INVALID,
            ) from exc
        if not isinstance(document, dict):
            raise BackendFailure(
                f"{self.backend_id} returned a JSON value that is not an object",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_RESPONSE_INVALID,
            )
        return document

    def _extract_image(self, document: dict[str, Any]) -> tuple[bytes, str]:
        field = str(self.option("response_image_field", "image_base64"))
        encoding = str(self.option("response_image_encoding", "base64"))
        declared = str(document.get("media_type") or document.get("mime_type") or "image/png")
        raw = document.get(field)
        if not isinstance(raw, str) or not raw:
            raise BackendFailure(
                f"{self.backend_id} response has no usable {field!r} field",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_RESPONSE_INVALID,
                detail={"service_keys": sorted(str(key) for key in document)},
            )
        if encoding == "url":
            raise BackendFailure(
                f"{self.backend_id} is configured for url responses, which this adapter does not fetch; "
                "put the image in a base64 field instead",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_RESPONSE_INVALID,
            )
        try:
            image_bytes = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise BackendFailure(
                f"{self.backend_id} returned a {field!r} value that is not valid base64",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_RESPONSE_INVALID,
            ) from exc
        return image_bytes, declared

    def _store_image(self, io: ArtifactIO, image_bytes: bytes, declared_media: str) -> ArtifactRef:
        if len(image_bytes) > self._max_bytes:
            raise BackendFailure(
                f"{self.backend_id} returned an {len(image_bytes)} byte image, above the response cap",
                backend_id=self.backend_id,
                code=ErrorCode.BACKEND_RESPONSE_TOO_LARGE,
            )
        media = _sniff_media_type(image_bytes, declared_media, self._allowed_media, self.backend_id)
        width, height = _decode_dimensions(image_bytes, self.backend_id)
        store = io.store
        store.validate_image_dimensions(width, height, media_type=media)
        return store.write(
            image_bytes,
            kind=ArtifactKind.CANDIDATE,
            media_type=media,
            suffix=_suffix_for(media),
            width=width,
            height=height,
        )


def _sniff_media_type(data: bytes, declared: str, allowed: tuple[str, ...], backend_id: str) -> str:
    """Verify that the declared media type matches the bytes and is permitted."""
    sniffed = None
    for media, prefixes in _MAGIC_PREFIXES.items():
        if any(data.startswith(prefix) for prefix in prefixes):
            sniffed = media
            break
    if sniffed is None:
        raise BackendFailure(
            f"{backend_id} returned bytes that are not a PNG, JPEG, or WebP image",
            backend_id=backend_id,
            code=ErrorCode.MEDIA_TYPE_REJECTED,
        )
    if sniffed not in allowed:
        raise BackendFailure(
            f"{backend_id} returned a {sniffed} image, which is not in allowed_media_types",
            backend_id=backend_id,
            code=ErrorCode.MEDIA_TYPE_REJECTED,
            detail={"media_type": sniffed, "allowed": list(allowed)},
        )
    if (
        declared
        and declared != sniffed
        and declared not in {"application/octet-stream", "binary/octet-stream"}
    ):
        raise BackendFailure(
            f"{backend_id} declared {declared} but the bytes are {sniffed}",
            backend_id=backend_id,
            code=ErrorCode.MEDIA_TYPE_REJECTED,
            detail={"declared": declared, "sniffed": sniffed},
        )
    return sniffed


def _suffix_for(media: str) -> str:
    return {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(media, "png")


def _decode_dimensions(data: bytes, backend_id: str) -> tuple[int, int]:
    """Decode the image header to confirm the payload is a real, plausibly sized image."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as handle:
            handle.verify()
        with Image.open(io.BytesIO(data)) as handle:
            return int(handle.width), int(handle.height)
    except (UnidentifiedImageError, OSError) as exc:
        raise BackendFailure(
            f"{backend_id} returned an image that could not be decoded: {exc}",
            backend_id=backend_id,
            code=ErrorCode.MEDIA_CORRUPT,
        ) from exc


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
