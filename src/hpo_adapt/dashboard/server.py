"""DashboardServer: zero-dependency local web server for the run dashboard.

Serves:
  GET /            -> the single-page app (dashboard/app.html)
  GET /api/state   -> JSON snapshot of the recorded run document
  GET /api/health  -> {"ok": true}

Runs in a daemon thread; binds 127.0.0.1 by default; port is auto-selected
when the preferred one is busy.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hpo_adapt.dashboard.state import RunRecorder

_APP_HTML_PATH = Path(__file__).with_name("app.html")
_PREFERRED_PORTS = tuple(range(8710, 8730))


class _Handler(BaseHTTPRequestHandler):
    server_version = "hpo-adapt-dashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path == "/api/state":
            payload = self.server.recorder.snapshot()  # type: ignore[attr-defined]
            payload["server_url"] = getattr(self.server, "url_label", "")  # type: ignore[attr-defined]
            self._send_json(payload)
        elif self.path == "/api/health":
            self._send_json({"ok": True})
        elif self.path in ("/", "/index.html"):
            self._send_bytes(_APP_HTML_PATH.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send_json({"error": "not found"}, status=404)

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # silence stdlib noise
        return


class DashboardServer:
    def __init__(
        self,
        recorder: RunRecorder,
        host: str = "127.0.0.1",
        port: int | None = None,
    ) -> None:
        self.recorder = recorder
        self.host = host
        self.port = port
        self.fallback_from: int | None = None  # preferred port lost to an older run
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("server not started")
        return f"http://{self.host}:{self.port}"

    def start(self) -> "DashboardServer":
        """Bind and serve in a daemon thread (non-blocking)."""
        candidates = [self.port] if self.port else list(_PREFERRED_PORTS)
        preferred = candidates[0]
        server: ThreadingHTTPServer | None = None
        for candidate in candidates:
            try:
                server = ThreadingHTTPServer((self.host, int(candidate)), _Handler)
                break
            except OSError:  # port busy
                continue
        if server is None:
            # fall back to an ephemeral port
            server = ThreadingHTTPServer((self.host, 0), _Handler)
        server.recorder = self.recorder  # type: ignore[attr-defined]
        server.daemon_threads = True
        self._server = server
        self.port = server.server_address[1]
        if self.port != preferred:
            self.fallback_from = preferred
        server.url_label = self.url  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="hpo-adapt-dashboard",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
