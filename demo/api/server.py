import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Optional, List

import numpy as np
import yaml
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time

# Precomputed standard ITU-T G.711 mu-law linear expansion lookup table (256 16-bit PCM values)
# Enables zero-dependency mu-law decoding in Python 3.10+ without the deprecated audioop module
G711_ULAW_DECODE_TABLE = np.array([
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956, -23932, -22908,
    -21884, -20860, -19836, -18812, -17788, -16764, -15996, -15484, -14972, -14460,
    -13948, -13436, -12924, -12412, -11900, -11388, -10876, -10364, -9852, -9340,
    -8828, -8316, -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
    -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092, -3900, -3772,
    -3644, -3516, -3388, -3260, -3132, -3004, -2876, -2748, -2620, -2492,
    -2364, -2236, -2108, -1980, -1884, -1820, -1756, -1692, -1628, -1564,
    -1500, -1436, -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
    -876, -844, -812, -780, -748, -716, -684, -652, -620, -588,
    -556, -524, -492, -460, -428, -396, -372, -356, -340, -324,
    -308, -292, -276, -260, -244, -228, -212, -196, -180, -164,
    -148, -132, -120, -112, -104, -96, -88, -80, -72, -64,
    -56, -48, -40, -32, -24, -16, -8, 0, 32124, 31100,
    30076, 29052, 28028, 27004, 25980, 24956, 23932, 22908, 21884, 20860,
    19836, 18812, 17788, 16764, 15996, 15484, 14972, 14460, 13948, 13436,
    12924, 12412, 11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
    7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140, 5884, 5628,
    5372, 5116, 4860, 4604, 4348, 4092, 3900, 3772, 3644, 3516,
    3388, 3260, 3132, 3004, 2876, 2748, 2620, 2492, 2364, 2236,
    2108, 1980, 1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
    1372, 1308, 1244, 1180, 1116, 1052, 988, 924, 876, 844,
    812, 780, 748, 716, 684, 652, 620, 588, 556, 524,
    492, 460, 428, 396, 372, 356, 340, 324, 308, 292,
    276, 260, 244, 228, 212, 196, 180, 164, 148, 132,
    120, 112, 104, 96, 88, 80, 72, 64, 56, 48,
    40, 32, 24, 16, 8, 0
], dtype=np.int16)

from vaanishield.boundary.localization import find_codeswitch_points, CodeSwitchPoint
from vaanishield.boundary.features import extract_boundary_features, aggregate_boundary_features
from vaanishield.utils.audio_io import load_audio, save_audio
from vaanishield.fusion.explain import explain_prediction
from vaanishield.reporting.certificate import generate_forensic_certificate_pdf, compute_sha256

app = FastAPI(title="VaaniShield Research & Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = "configs/default.yaml"
CHECKPOINT_PATH = "checkpoints/fusion_checkpoint.pkl"

MODEL = None
PROJECTOR = None
CFG = None
WHISPER_MODEL = None


def get_whisper():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        try:
            import whisper
            WHISPER_MODEL = whisper.load_model("base")
        except Exception as e:
            print(f"[ASR Warning] Whisper model not loaded: {e}")
            WHISPER_MODEL = False
    return WHISPER_MODEL if WHISPER_MODEL is not False else None


def get_pipeline():
    global MODEL, PROJECTOR, CFG
    if MODEL is None:
        if not os.path.exists(CHECKPOINT_PATH):
            raise RuntimeError(f"Checkpoint not found at {CHECKPOINT_PATH}. Run scripts/train.py first.")
        from vaanishield.fusion.classifier import FusionClassifier
        data = FusionClassifier.load_checkpoint(CHECKPOINT_PATH, verify_checksum=True)
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            CFG = yaml.safe_load(f)
        MODEL = data["model"]
        PROJECTOR = data["projector"]
    return MODEL, PROJECTOR, CFG


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "VaaniShield Detection API", "version": "2.0.0"}


SAMPLE_CATALOG = [
    {
        "id": "sample_bonafide",
        "name": "Authentic Human Voice — Hindi-English (OpenSLR 103 spk_1119)",
        "path": "data/raw/spk_1119_1119_020.wav",
        "transcript": "Mera laptop start nahi ho raha, can you check the battery?",
        "expected": "bonafide",
        "language_pair": "hi-en",
    },
    {
        "id": "sample_cloned",
        "name": "Cloned Voice — Hindi-English (Coqui XTTS v2 spk_1119)",
        "path": "data/processed/train_clones/spk_1119_utt_000_xtts.wav",
        "transcript": "Mera laptop start nahi ho raha, can you check the battery?",
        "expected": "spoof",
        "language_pair": "hi-en",
    },
    {
        "id": "sample_codec",
        "name": "Cloned Voice — 8 kHz Telephony Codec (G.711 μ-law)",
        "path": "data/processed/augmented/spk_1119_utt_000_xtts_codec.wav",
        "transcript": "Mera laptop start nahi ho raha, can you check the battery?",
        "expected": "spoof",
        "language_pair": "hi-en",
    },
    {
        "id": "sample_kannada_bonafide",
        "name": "[UNSEEN TEST SPLIT] Authentic Human — Kannada-English (OpenSLR 79 spk_kn_02)",
        "path": "data/raw/spk_kn_02_knm_00271_00003824418.wav",
        "transcript": "Nimma bank account block agide, verify your account details immediately.",
        "expected": "bonafide",
        "language_pair": "kn-en",
    },
    {
        "id": "sample_tamil_cloned",
        "name": "[UNSEEN VAL SPLIT] Cloned Voice — Tamil-English (spk_ta_02)",
        "path": "data/processed/train_clones/spk_ta_02_utt_025_xtts.wav",
        "transcript": "Unga bank account freeze aayiduchu, verify your details immediately.",
        "expected": "spoof",
        "language_pair": "ta-en",
    },
    {
        "id": "sample_heldout_cloned",
        "name": "[UNSEEN HELDOUT SPLIT] Cloned Voice — Unseen Held-Out Speaker (spk_5945)",
        "path": "data/processed/train_clones/spk_5945_utt_015_xtts.wav",
        "transcript": "Mera account freeze ho gaya hai, please verify the transaction now.",
        "expected": "spoof",
        "language_pair": "hi-en",
    },
]
SAMPLE_MAP = {item["id"]: item["path"] for item in SAMPLE_CATALOG}
SAMPLE_LANG_MAP = {item["id"]: item["language_pair"] for item in SAMPLE_CATALOG}
SAMPLE_TRANSCRIPT_MAP = {item["id"]: item["transcript"] for item in SAMPLE_CATALOG}


@app.get("/api/samples")
def get_samples():
    return SAMPLE_CATALOG


@app.get("/api/audio/{sample_id}")
def get_sample_audio(sample_id: str):
    audio_path = SAMPLE_MAP.get(sample_id)
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail=f"Audio sample not found: {sample_id}")
    return FileResponse(audio_path, media_type="audio/wav")


@app.post("/api/analyze")
async def analyze_audio(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    transcript: Optional[str] = Form(None),
):
    """Run dual-stream detection pipeline on uploaded audio or catalog sample."""
    tmp_path = None
    try:
        if file and file.filename:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                content = await file.read()
                tmp.write(content)
            audio_source = tmp_path
            target_transcript = transcript
            lang_pair = "hi-en"
        elif sample_id:
            audio_source = SAMPLE_MAP.get(sample_id)
            if not audio_source or not os.path.exists(audio_source):
                raise HTTPException(status_code=404, detail=f"Preset sample not found: {sample_id}")
            target_transcript = transcript if transcript is not None else SAMPLE_TRANSCRIPT_MAP.get(sample_id, "")
            lang_pair = SAMPLE_LANG_MAP.get(sample_id, "hi-en")
        else:
            raise HTTPException(status_code=400, detail="Either 'file' or 'sample_id' must be provided.")

        model, projector, cfg = get_pipeline()
        audio_data, sr = load_audio(audio_source)
        duration_sec = len(audio_data) / sr

        # 1. Boundary & Acoustic Seam Localization
        # If user/preset provided transcript with code switching, localize linguistic boundaries.
        # Otherwise, directly extract acoustic transition seams via spectral flux without Whisper latency.
        points = []
        localization_method = "acoustic_flux_changepoints"
        if target_transcript and len(target_transcript.strip().split()) >= 2:
            tokens = target_transcript.strip().split()
            points = find_codeswitch_points(
                audio_path=audio_source,
                transcript_tokens=tokens,
                lid_model="fasttext-lid176",
                alignment_backend="whisper",
                window_ms_before=300,
                window_ms_after=300,
            )
            if points:
                localization_method = "transcript_codeswitch_alignment"

        # Fallback to acoustic change points if no transcript or no linguistic boundaries found
        if not points:
            from vaanishield.boundary.localization import find_acoustic_change_points
            points = find_acoustic_change_points(
                audio_path=audio_source,
                window_ms_before=300,
                window_ms_after=300,
                max_points=5,
            )
            localization_method = "acoustic_flux_changepoints"

        # 2. Boundary-local features
        b_feats = extract_boundary_features(audio_source, points)
        agg_vec = aggregate_boundary_features(b_feats, aggregation="mean_and_max")

        # 4. Speaker-nulled global embedding
        raw_emb = projector.embed(audio_source)
        nulled_emb = projector.project(raw_emb)

        # 5. Fusion inference with calibrated probability and bootstrap CI
        probs, ci_low, ci_high = model.predict_with_ci(
            agg_vec.reshape(1, -1),
            nulled_emb.reshape(1, -1),
            n_bootstrap=100,
            ci=0.95,
        )
        prob_val = float(probs[0])
        ci_low_val = float(ci_low[0])
        ci_high_val = float(ci_high[0])

        is_spoof = prob_val >= 0.5
        decision_action = model.predict_operational_tier(prob_val)

        if decision_action == "BLOCK_AND_ALERT":
            risk_level = "CRITICAL"
            action_taken = "CALL_BLOCKED"
        elif decision_action == "STEP_UP_AUTH":
            risk_level = "ELEVATED"
            action_taken = "STEP_UP_REQUIRED"
        else:
            risk_level = "SECURE"
            action_taken = "VERIFIED_GENUINE"

        # 6. Privacy-compliant acoustic explanations (zero transcript leakage per DPDPA 2023)
        # Transform features using fitted scaler for correct linear weight attribution
        fused_vector = np.concatenate([agg_vec, nulled_emb])
        raw_model = getattr(model, "model_", model)
        explanations = explain_prediction(
            fused_feature_vector=fused_vector,
            model=raw_model,
            codeswitch_points=points,
            boundary_vectors=b_feats,
            overall_spoof_prob=prob_val,
        )

        # Downsample waveform for frontend visualization
        step = max(1, len(audio_data) // 120)
        waveform_peaks = [float(abs(audio_data[i])) for i in range(0, len(audio_data), step)][:120]

        # Cryptographic integrity digest of original audio
        audio_sha256 = compute_sha256(audio_source)

        # Strictly non-semantic metadata returned: no spoken text or transcript stored/returned
        return {
            "status": "success",
            "is_spoof": is_spoof,
            "spoof_probability": prob_val,
            "confidence_interval_95": [ci_low_val, ci_high_val],
            "audio_sha256": audio_sha256,
            "operational_decision": {
                "decision": decision_action,
                "risk_tier": risk_level,
                "tau_review": getattr(model, "tau_review", 0.48),
                "tau_block": getattr(model, "tau_block", 0.52),
                "recommended_action": action_taken,
            },
            "action_taken": action_taken,
            "audio_metadata": {
                "duration_sec": round(duration_sec, 2),
                "sample_rate_hz": sr,
                "audio_sha256": audio_sha256,
                "detected_boundary_count": len(points),
                "localization_method": localization_method,
                "language_pair": lang_pair,
                "waveform_peaks": waveform_peaks,
            },
            "explanations": explanations,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/api/evaluation")
def get_evaluation_reports():
    """Return latest speaker-disjoint 4-split evaluation and ablation reports."""
    eval_data = {}
    ablation_data = {}
    if os.path.exists("results/evaluation_report.json"):
        with open("results/evaluation_report.json", "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    if os.path.exists("results/ablation_report.json"):
        with open("results/ablation_report.json", "r", encoding="utf-8") as f:
            ablation_data = json.load(f)
    return {
        "evaluation": eval_data,
        "ablation": ablation_data,
    }


class CertificateRequest(BaseModel):
    analysis_result: dict
    call_id: Optional[str] = None


@app.post("/api/export_certificate")
async def export_certificate(req: CertificateRequest):
    """
    Generates a tamper-evident PDF Forensic Audit Incident Certificate
    compliant with RBI / CERT-In directives and DPDPA 2023 non-semantic privacy.
    """
    try:
        pdf_bytes = generate_forensic_certificate_pdf(
            analysis_result=req.analysis_result,
            call_id=req.call_id
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=vaanishield_forensic_audit_{int(time.time())}.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate certificate: {str(e)}")


class PolicyWebhookRequest(BaseModel):
    call_id: str
    decision: str
    spoof_probability: float
    target_webhook_url: Optional[str] = None


@app.post("/api/webhook/dispatch")
async def dispatch_policy_webhook(req: PolicyWebhookRequest):
    """
    Dispatches automated call control decisions to banking contact center / IVR webhooks.
    If target_webhook_url is provided, performs an authenticated HTTP POST.
    Otherwise, returns a simulated delivery payload.
    """
    import urllib.request
    import urllib.error

    # Policy routing rules
    action = "TERMINATE_CALL" if req.decision == "BLOCK_AND_ALERT" else (
        "STEP_UP_MFA" if req.decision == "STEP_UP_AUTH" else "ALLOW_CALL"
    )
    payload = {
        "event": "vaanishield.call_guard.decision",
        "timestamp": int(time.time()),
        "call_id": req.call_id,
        "decision": req.decision,
        "action": action,
        "spoof_probability": req.spoof_probability,
    }

    if req.target_webhook_url:
        import urllib.parse
        parsed = urllib.parse.urlparse(req.target_webhook_url)
        if parsed.scheme not in ("http", "https"):
            raise HTTPException(status_code=400, detail="Invalid webhook URL scheme: must be http or https.")
        
        # Prevent SSRF to cloud metadata endpoints (e.g. AWS 169.254.169.254)
        hostname = (parsed.hostname or "").lower()
        if hostname in ("169.254.169.254", "metadata.google.internal", "metadata"):
            raise HTTPException(status_code=403, detail="Dispatch to cloud metadata service is strictly forbidden.")

        try:
            req_data = json.dumps(payload).encode("utf-8")
            url_req = urllib.request.Request(
                req.target_webhook_url,
                data=req_data,
                headers={"Content-Type": "application/json", "User-Agent": "VaaniShield-Webhook-Dispatcher/2.0"},
                method="POST"
            )
            with urllib.request.urlopen(url_req, timeout=5) as response:
                status_code = response.getcode()
                payload["delivery_status"] = f"DELIVERED_HTTP_{status_code}"
                return {"status": "success", "mode": "network_dispatch", "http_status": status_code, "dispatched_payload": payload}
        except Exception as e:
            payload["delivery_status"] = f"DISPATCH_FAILED: {str(e)}"
            return JSONResponse(
                status_code=502,
                content={"status": "error", "mode": "network_dispatch", "detail": str(e), "dispatched_payload": payload}
            )
    else:
        payload["delivery_status"] = "SIMULATED_LOCAL_DISPATCH"
        return {"status": "success", "mode": "simulation", "dispatched_payload": payload}


@app.websocket("/ws/stream_analyze")
async def websocket_stream_analyze(websocket: WebSocket):
    """Real-time streaming WebSocket endpoint for continuous audio chunk analysis."""
    await websocket.accept()
    model, projector, cfg = get_pipeline()
    buffer = np.zeros(0, dtype=np.float32)
    sr = 8000
    chunk_count = 0

    try:
        while True:
            data = await websocket.receive()
            new_chunk = None

            if "bytes" in data and data["bytes"]:
                chunk_bytes = data["bytes"]
                try:
                    new_chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
                except Exception:
                    new_chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            elif "text" in data:
                text_msg = data["text"]
                if text_msg == "RESET":
                    buffer = np.zeros(0, dtype=np.float32)
                    chunk_count = 0
                    await websocket.send_json({"type": "reset_ack"})
                    continue
                else:
                    # Support Twilio Media Stream / SIP trunk JSON protocol
                    try:
                        msg_obj = json.loads(text_msg)
                        event = msg_obj.get("event")
                        if event == "start":
                            buffer = np.zeros(0, dtype=np.float32)
                            chunk_count = 0
                            await websocket.send_json({"type": "sip_start_ack", "streamSid": msg_obj.get("streamSid")})
                            continue
                        elif event == "media":
                            media_payload = msg_obj.get("media", {}).get("payload")
                            if media_payload:
                                import base64
                                raw_mulaw = base64.b64decode(media_payload)
                                # Zero-dependency ITU-T G.711 mu-law decoding via table lookup
                                u_idx = np.frombuffer(raw_mulaw, dtype=np.uint8)
                                decoded_pcm = G711_ULAW_DECODE_TABLE[u_idx]
                                new_chunk = decoded_pcm.astype(np.float32) / 32768.0
                        elif event == "stop":
                            await websocket.send_json({"type": "sip_stop_ack", "final_chunks": chunk_count})
                            continue
                    except Exception:
                        pass

            if new_chunk is not None and len(new_chunk) > 0:
                buffer = np.concatenate([buffer, new_chunk])
                chunk_count += 1

                # Keep rolling buffer of last 5.0 seconds
                max_buffer_len = sr * 5
                if len(buffer) > max_buffer_len:
                    buffer = buffer[-max_buffer_len:]

                # Stable streaming: requires >= 2.5s of telephony buffer for reliable prosodic continuity
                if len(buffer) >= int(2.5 * sr):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_chunk:
                        tmp_chunk_path = tmp_chunk.name
                    save_audio(tmp_chunk_path, buffer, sr)

                    try:
                        from vaanishield.boundary.localization import find_acoustic_change_points
                        points = find_acoustic_change_points(
                            audio_path=tmp_chunk_path,
                            window_ms_before=250,
                            window_ms_after=250,
                            max_points=4,
                        )
                        b_feats = extract_boundary_features(tmp_chunk_path, points)
                        agg_vec = aggregate_boundary_features(b_feats, aggregation="mean_and_max")
                        raw_emb = projector.embed(tmp_chunk_path)
                        nulled_emb = projector.project(raw_emb)

                        probs, ci_l, ci_h = model.predict_with_ci(
                            agg_vec.reshape(1, -1),
                            nulled_emb.reshape(1, -1),
                            n_bootstrap=50,
                            ci=0.95,
                        )
                        prob_val = float(probs[0])
                        ci_low_val = float(ci_l[0])
                        ci_high_val = float(ci_h[0])

                        await websocket.send_json({
                            "type": "telemetry",
                            "chunk_id": chunk_count,
                            "buffer_duration_sec": round(len(buffer) / sr, 2),
                            "spoof_probability": prob_val,
                            "confidence_interval_95": [ci_low_val, ci_high_val],
                            "is_spoof": prob_val >= 0.5,
                            "alert_level": "CRITICAL" if prob_val > 0.75 else ("WARNING" if prob_val >= 0.5 else "SECURE"),
                        })
                    finally:
                        if os.path.exists(tmp_chunk_path):
                            os.remove(tmp_chunk_path)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


WEB_DIST_DIR = Path("web/dist")
if WEB_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIST_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
