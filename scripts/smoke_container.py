"""Verify a running, model-free demo image. This is not an accuracy benchmark."""
import argparse
import io
import json
import time
import urllib.error
import urllib.request
import wave


def fetch(url, data=None, headers=None):
    request = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        response = urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return response.status, response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    deadline = time.monotonic() + 60
    while True:
        try:
            status, body = fetch(base + "/api/health")
            if status == 200 and json.loads(body).get("status") == "ok":
                break
        except (urllib.error.URLError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("API liveness did not become healthy within 60 seconds")
        time.sleep(2)

    status, body = fetch(base + "/")
    assert status == 200 and b'<div id="root">' in body, "Dashboard was not served"

    status, body = fetch(base + "/api/ready")
    assert status == 503, ("Model-free image must not claim readiness", status, body)
    assert json.loads(body)["code"] == "model_unavailable"

    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 8000)
    boundary = "vaanishield-smoke-test"
    data = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="smoke.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio.getvalue() + f"\r\n--{boundary}--\r\n".encode()
    status, body = fetch(base + "/api/analyze", data, {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    })
    assert status == 503, ("Inference must fail closed without a model", status, body)
    payload = json.loads(body)
    assert payload["detail"]["code"] == "model_unavailable"
    assert "spoof_probability" not in payload
    print("PASS: dashboard and liveness available; readiness and inference return 503 without a model.")


if __name__ == "__main__":
    main()
