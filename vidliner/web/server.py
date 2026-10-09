"""Token-protected loopback HTTP transport; provider credentials stay in the runtime."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from pydantic import ValidationError

from vidliner.core.errors import VidlinerError
from vidliner.pipeline.canvas_host import CanvasHost
from vidliner.pipeline.service import Session
from vidliner.pipeline.workflow import load_workflow, operator_catalogue
from vidliner.reports.workflow import render_workflow


class CanvasHTTPServer(ThreadingHTTPServer):
    """Serve one saved workflow on IPv4 loopback with a per-process API token."""

    daemon_threads = True

    def __init__(
        self,
        document_path: Path,
        workspace: Path,
        *,
        runtime_path: Path | None = None,
        port: int = 8767,
        allow_external: bool = False,
    ) -> None:
        super().__init__(("127.0.0.1", port), _Handler)
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.token = secrets.token_urlsafe(32)
        self.loop = asyncio.new_event_loop()
        self.worker = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.worker.start()

        async def initialize() -> CanvasHost:
            document = load_workflow(document_path)
            session = Session.open(workspace, runtime_path=runtime_path, create=True)
            key = hashlib.sha256(str(document_path.resolve()).encode()).hexdigest()[:16]
            return CanvasHost(
                session,
                document,
                session.workspace.cache_dir / f"canvas-{key}.db",
                allow_external=allow_external,
            )

        try:
            self.host = asyncio.run_coroutine_threadsafe(initialize(), self.loop).result(timeout=30)
        except Exception:
            super().server_close()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.worker.join(timeout=5)
            self.loop.close()
            raise

    async def dispatch(self, method: str, path: str, payload: dict[str, Any]) -> Any:
        """Route authenticated commands to one event-loop-owned host."""
        if method == "GET":
            if path == "/":
                return render_workflow(
                    self.host.store.document(),
                    host_config={
                        **self.host.draft(),
                        "token": self.token,
                        "catalogue": operator_catalogue(),
                    },
                )
            if path == "/api/workflow":
                return self.host.draft()
            if path == "/api/catalog":
                return operator_catalogue()
            if path == "/api/jobs":
                return self.host.store.jobs()
            if path.startswith("/api/jobs/"):
                return self.host.store.job(path.split("/")[-1])
        if method == "POST":
            if path == "/api/edits":
                return self.host.edit(payload)
            if path == "/api/nodes":
                return self.host.node(payload)
            if path == "/api/execute":
                return await self.host.execute(payload)
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                return await self.host.cancel(parts[2])
            if len(parts) == 6 and parts[:2] == ["api", "jobs"] and parts[3] == "tasks":
                return await self.host.task_action(parts[2], unquote(parts[4]), parts[5])
        raise LookupError("unknown canvas route")

    def server_close(self) -> None:
        """Close jobs and SQLite on their owning loop, then release the socket."""
        super().server_close()
        if hasattr(self, "host"):
            asyncio.run_coroutine_threadsafe(self.host.close(), self.loop).result(timeout=15)
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.worker.join(timeout=5)
            self.loop.close()


class _Handler(BaseHTTPRequestHandler):
    server: CanvasHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        """Keep private requests and signed media URLs out of access logs."""
        pass  # No request bodies, task URLs or auth material in access logs.

    def do_GET(self) -> None:
        """Serve the canvas or authenticated private execution state."""
        self._handle("GET")

    def do_POST(self) -> None:
        """Accept bounded JSON commands only from the local token holder."""
        self._handle("POST")

    def _respond(self, status: int, body: Any, *, html: bool = False) -> None:
        data = (body if html else json.dumps(body, ensure_ascii=False)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'unsafe-inline'; "
            "style-src 'unsafe-inline'; connect-src 'self' https://api.github.com/repos/VidLiner/vidliner; "
            "media-src http: https: blob:; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )
        self.end_headers()
        self.wfile.write(data)

    def _handle(self, method: str) -> None:
        origin = self.server.origin
        if (
            self.headers.get("Host") != origin.removeprefix("http://")
            or self.headers.get("Origin") not in {None, origin}
            or self.headers.get("Sec-Fetch-Site") in {"cross-site", "same-site"}
        ):
            self._respond(403, {"error": "local origin required"})
            return
        if (self.path != "/" or method != "GET") and not hmac.compare_digest(
            self.headers.get("X-Vidliner-Token", ""), self.server.token
        ):
            self._respond(403, {"error": "canvas token required"})
            return
        payload: dict[str, Any] = {}
        try:
            if method == "POST":
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024 or self.headers.get("Transfer-Encoding"):
                    self._respond(413, {"error": "JSON body must be at most 1 MiB"})
                    return
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    self._respond(415, {"error": "application/json required"})
                    return
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("body must be an object")

            async def deliver() -> Any:
                value = await self.server.dispatch(method, self.path, payload)
                # Detach mutable jobs before crossing into the HTTP worker thread.
                return value if isinstance(value, str) else json.loads(json.dumps(value))

            result = asyncio.run_coroutine_threadsafe(deliver(), self.server.loop).result(timeout=180)
            self._respond(200, result, html=self.path == "/")
        except LookupError:
            self._respond(404, {"error": "unknown route or node"})
        except (ValidationError, ValueError) as exc:
            # Pydantic JSON errors can include inputs; expose only typed locations/messages.
            message = "Invalid request"
            if isinstance(exc, ValidationError):
                message = "; ".join(
                    str(item["msg"]) for item in exc.errors(include_input=False, include_context=False)
                )
            self._respond(422, {"error": message})
        except VidlinerError as exc:
            self._respond(409, {"error": str(exc), "code": exc.code.value})
        except Exception:
            self._respond(500, {"error": "Request failed; check saved job state before retrying execution."})
