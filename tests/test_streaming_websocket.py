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


def test_websocket_twilio_sip_streaming():
    """Verify live Twilio Media Stream / SIP trunk JSON protocol with base64 encoded G.711 mu-law."""
    import base64
    import audioop

    client = TestClient(app)
    with client.websocket_connect("/ws/stream_analyze") as websocket:
        # 1. Twilio Stream Start event
        start_event = {
            "event": "start",
            "streamSid": "MZ1234567890abcdef",
            "start": {"accountSid": "AC1234", "callSid": "CA1234"}
        }
        websocket.send_text(json.dumps(start_event))
        ack = websocket.receive_json()
        assert ack["type"] == "sip_start_ack"
        assert ack["streamSid"] == "MZ1234567890abcdef"

        # 2. Twilio Media chunks (8 kHz mu-law)
        sr = 8000
        # Generate 500ms audio chunks (4000 samples)
        # 6 chunks = 3.0 seconds (exceeds the 2.5s streaming threshold)
        raw_pcm = (0.3 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)) * 32767).astype(np.int16).tobytes()
        # Encode to mu-law without audioop using inverse lookup or precomputed mu-law array
        # Standard G.711 compression
        mulaw_chunk = audioop.lin2ulaw(raw_pcm, 2)
        b64_payload = base64.b64encode(mulaw_chunk).decode("utf-8")

        for _ in range(6):
            media_event = {
                "event": "media",
                "media": {"payload": b64_payload}
            }
            websocket.send_text(json.dumps(media_event))

        # Live telemetry MUST now be produced by the pipeline
        telemetry = websocket.receive_json()
        assert telemetry["type"] == "telemetry"
        assert "spoof_probability" in telemetry
        assert 0.0 <= telemetry["spoof_probability"] <= 1.0
        assert telemetry["buffer_duration_sec"] >= 2.5
        assert telemetry["alert_level"] in ["CRITICAL", "WARNING", "SECURE"]

        # 3. Twilio Stop event
        stop_event = {"event": "stop", "streamSid": "MZ1234567890abcdef"}
        websocket.send_text(json.dumps(stop_event))
        
        # Read until sip_stop_ack (draining any remaining chunk telemetry)
        final_msg = websocket.receive_json()
        while final_msg.get("type") == "telemetry":
            final_msg = websocket.receive_json()

        assert final_msg["type"] == "sip_stop_ack"
        assert final_msg["final_chunks"] >= 6


