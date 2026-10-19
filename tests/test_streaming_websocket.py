import json
import numpy as np
import pytest
from starlette.testclient import TestClient

from demo.api.server import app


def test_websocket_streaming_pipeline():
    """Verify continuous audio streaming, telemetry generation, and reset on WebSocket."""
    client = TestClient(app)
    with client.websocket_connect("/ws/stream_analyze") as websocket:
        sr = 8000
        # 1. Send small chunks that do not yet reach the 2.5s buffer threshold
        sub_chunk = np.zeros(sr, dtype=np.float32)  # 1.0 second chunk
        websocket.send_bytes(sub_chunk.tobytes())

        # 2. Send 2 more seconds so buffer reaches 3.0s (>= 2.5s threshold)
        active_chunk = (0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 2.0, sr * 2, endpoint=False))).astype(np.float32)
        websocket.send_bytes(active_chunk.tobytes())

        # Receive telemetry from the pipeline
        data = websocket.receive_json()
        assert data["type"] == "telemetry"
        assert "spoof_probability" in data
        assert 0.0 <= data["spoof_probability"] <= 1.0
        assert "confidence_interval_95" in data
        assert len(data["confidence_interval_95"]) == 2
        assert data["buffer_duration_sec"] >= 2.5
        assert data["alert_level"] in ["CRITICAL", "WARNING", "SECURE"]

        # 3. Test buffer reset
        websocket.send_text("RESET")
        ack = websocket.receive_json()
        assert ack["type"] == "reset_ack"
