"""Exercise the smoke client's startup wait against real TCP connections."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading


def test_smoke_retries_connection_reset_during_startup():
    requests = []

    class StartingServer(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if len(requests) == 1:
                # Docker can accept a forwarded connection before the app listens.
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                           struct.pack("ii", 1, 0))
                self.connection.close()
                self.close_connection = True
                return
            if self.path == "/api/health":
                status, body = 200, b'{"status":"ok"}'
            elif self.path == "/":
                status, body = 200, b'<div id="root"></div>'
            else:
                status, body = 503, b'{"code":"model_unavailable"}'
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            requests.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"detail":{"code":"model_unavailable"}}')

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), StartingServer) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            script = Path(__file__).resolve().parents[1] / "scripts/smoke_container.py"
            result = subprocess.run(
                [sys.executable, str(script), "--base-url",
                 f"http://127.0.0.1:{server.server_port}"],
                capture_output=True, text=True, timeout=15,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert result.returncode == 0, result.stderr
    assert requests == ["/api/health", "/api/health", "/", "/api/ready", "/api/analyze"]
    assert "PASS:" in result.stdout
