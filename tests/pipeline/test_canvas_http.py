"""Real loopback HTTP checks: origin, token, CAS and graph execution."""

import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tests.unit.test_workflow_interfaces import document
from vidliner.web.server import CanvasHTTPServer


def test_loopback_api_edits_executes_and_rejects_foreign_requests(tmp_path):
    source = tmp_path / "workflow.json"
    source.write_text(document().model_dump_json())
    server = CanvasHTTPServer(source, tmp_path / "workspace", port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def call(path, payload=None, *, token=True, origin=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Vidliner-Token"] = server.token
        if origin:
            headers["Origin"] = origin
        request = Request(
            server.origin + path,
            headers=headers,
            data=json.dumps(payload).encode() if payload is not None else None,
        )
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, response.read().decode()
        except HTTPError as error:
            return error.code, error.read().decode()

    try:
        status, html = call("/", token=False)
        assert status == 200 and "Execute" in html and "data-direction" in html
        assert call("/api/jobs", token=False)[0] == 403
        assert call("/api/jobs", origin="http://foreign.example")[0] == 403
        draft = json.loads(call("/api/workflow")[1])
        edit = {
            "expected_digest": draft["digest"],
            "edits": [{"op": "move_node", "node_id": "plan", "position": {"x": 200}}],
        }
        assert call("/api/edits", edit)[0] == 200
        assert call("/api/edits", edit)[0] == 409
        draft = json.loads(call("/api/workflow")[1])
        intent = {"expected_digest": draft["digest"], "job_id": "http-job"}
        assert call("/api/execute", intent)[0] == 200
        for _ in range(50):
            job = json.loads(call("/api/jobs/http-job")[1])
            if job["status"] == "succeeded":
                break
            time.sleep(0.02)
        assert job["status"] == "succeeded" and job["nodes"]["plan"]["status"] == "succeeded"
        assert json.loads(call("/api/execute", intent)[1])["status"] == "succeeded"
        assert call("/api/execute", {**intent, "seed": 80})[0] == 409
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
