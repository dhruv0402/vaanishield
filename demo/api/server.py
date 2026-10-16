import os
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import time
import uuid

from vaanishield.boundary.localization import find_codeswitch_points
from vaanishield.boundary.features import extract_boundary_features, aggregate_boundary_features
from vaanishield.utils.audio_io import load_audio
from vaanishield.fusion.explain import explain_prediction

app = FastAPI(title="VaaniShield Enterprise Security API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
            # Load the base multilingual model (supports Hindi, Tamil, Telugu, Kannada, English)
            WHISPER_MODEL = whisper.load_model("base")
        except Exception as e:
            print(f"[ASR Warning] Could not load whisper model: {e}")
            WHISPER_MODEL = False
    return WHISPER_MODEL if WHISPER_MODEL is not False else None

# In-Memory Enterprise SaaS Multi-Tenant Datastores
API_KEYS = {
    "vs_live_bank_alpha_9921": {"org": "State Bank Security Ops", "plan": "Enterprise Tier", "quota": 100000, "used": 142},
    "vs_live_fintech_beta_4410": {"org": "Fintech PayShield", "plan": "Growth Tier", "quota": 25000, "used": 89},
    "vs_test_sandbox_1234": {"org": "Sandbox Developer", "plan": "Developer Tier", "quota": 1000, "used": 18},
}

CALL_AUDIT_LOGS = [
    {
        "call_id": "call_ind_99201",
        "timestamp": "2026-10-06T01:10:14Z",
        "org": "State Bank Security Ops",
        "caller_id": "+91-98765-XXXXX",
        "channel": "Inbound SIP Trunk (IVR)",
        "duration_sec": 3.4,
        "language_pair": "hi-en",
        "spoof_probability": 0.942,
        "is_spoof": True,
        "action_taken": "TRANSACTION_FROZEN",
        "alert_level": "CRITICAL"
    },
    {
        "call_id": "call_ind_99202",
        "timestamp": "2026-10-06T01:12:30Z",
        "org": "State Bank Security Ops",
        "caller_id": "+91-98112-XXXXX",
        "channel": "WhatsApp Audio Gateway",
        "duration_sec": 4.1,
        "language_pair": "hi-en",
        "spoof_probability": 0.081,
        "is_spoof": False,
        "action_taken": "CALL_VERIFIED",
        "alert_level": "SECURE"
    },
    {
        "call_id": "call_ind_99203",
        "timestamp": "2026-10-06T01:14:02Z",
        "org": "Fintech PayShield",
        "caller_id": "+91-97401-XXXXX",
        "channel": "Contact Center Telephony",
        "duration_sec": 2.9,
        "language_pair": "kn-en",
        "spoof_probability": 0.884,
        "is_spoof": True,
        "action_taken": "ROUTED_TO_SPECIALIST",
        "alert_level": "CRITICAL"
    }
]

REGISTERED_WEBHOOKS = [
    {"id": "wh_01", "url": "https://api.banksecurity.in/v1/fraud-alerts", "events": ["spoof_detected", "call_flagged"], "status": "ACTIVE"}
]

class WebhookConfig(BaseModel):
    url: str
    events: list[str] = ["spoof_detected"]

def verify_api_key(x_api_key: Optional[str] = None):
    """Enforce API key authentication with sandbox fallback for web demo."""
    if not x_api_key:
        return {"org": "Public Web Console", "plan": "Demo Tier"}
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key. Unauthorized.")
    tenant = API_KEYS[x_api_key]
    tenant["used"] += 1
    return tenant

def get_pipeline():
    global MODEL, PROJECTOR, CFG
    if MODEL is None:
        if not os.path.exists(CHECKPOINT_PATH):
            raise RuntimeError(f"Checkpoint not found at {CHECKPOINT_PATH}")
        with open(CHECKPOINT_PATH, "rb") as f:
            data = pickle.load(f)
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            CFG = yaml.safe_load(f)
        MODEL = data["model"]
        PROJECTOR = data["projector"]
    return MODEL, PROJECTOR, CFG

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "VaaniShield Detection API"}

@app.get("/api/sessions")
def get_sessions():
    return {
        "status": "active",
        "active_sessions": [
            {
                "session_id": "sess_default_01",
                "org": "State Bank Security Ops",
                "tier": "Enterprise Multi-Stream Gate",
                "status": "PROTECTED",
            }
        ]
    }

@app.get("/api/samples")
def get_samples():
    return [
        {
            "id": "sample_bonafide",
            "name": "Authentic Human Voice (OpenSLR 103 spk_1119)",
            "path": "data/raw/spk_1119_1119_020.wav",
            "transcript": "Mera laptop start nahi ho raha, can you check the battery?",
            "expected": "bonafide"
        },
        {
            "id": "sample_cloned",
            "name": "Cloned Voice (Coqui XTTS v2)",
            "path": "data/processed/train_clones/spk_1119_utt_000_xtts.wav",
            "transcript": "Mera laptop start nahi ho raha, can you check the battery?",
            "expected": "spoof"
        },
        {
            "id": "sample_codec",
            "name": "Cloned Voice with Telephony Codec Compression",
            "path": "data/processed/augmented/spk_1119_utt_000_xtts_codec.wav",
            "transcript": "Mera laptop start nahi ho raha, can you check the battery?",
            "expected": "spoof"
        }
    ]

@app.get("/api/audio/{sample_id}")
def get_sample_audio(sample_id: str):
    sample_map = {
        "sample_bonafide": "data/raw/spk_1119_1119_020.wav",
        "sample_cloned": "data/processed/train_clones/spk_1119_utt_000_xtts.wav",
        "sample_codec": "data/processed/augmented/spk_1119_utt_000_xtts_codec.wav",
        "sample_kannada_bonafide": "data/raw/spk_2131_2131_003.wav",
        "sample_tamil_cloned": "data/processed/train_clones/spk_2131_utt_005_xtts.wav",
    }
    audio_path = sample_map.get(sample_id)
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail=f"Audio sample not found: {sample_id}")
    return FileResponse(audio_path, media_type="audio/wav")

@app.post("/api/analyze")
async def analyze_audio(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    transcript: Optional[str] = Form(None),
    caller_id: Optional[str] = Form("+91-98765-XXXXX"),
    channel: Optional[str] = Form("Inbound Telephony Gateway"),
    x_api_key: Optional[str] = Header(None),
):
    tenant = verify_api_key(x_api_key)
    try:
        model, projector, cfg = get_pipeline()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    tmp_path = None
    if file is not None:
        suffix = Path(file.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty audio file provided.")
            tmp.write(content)
            tmp_path = tmp.name
        audio_path = tmp_path
    elif sample_id:
        sample_map = {
            "sample_bonafide": "data/raw/spk_1119_1119_020.wav",
            "sample_cloned": "data/processed/train_clones/spk_1119_utt_000_xtts.wav",
            "sample_codec": "data/processed/augmented/spk_1119_utt_000_xtts_codec.wav",
            "sample_kannada_bonafide": "data/raw/spk_2131_2131_003.wav",
            "sample_tamil_cloned": "data/processed/train_clones/spk_2131_utt_005_xtts.wav",
        }
        audio_path = sample_map.get(sample_id)
        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(status_code=400, detail=f"Invalid or missing sample: {sample_id}")
    else:
        raise HTTPException(status_code=400, detail="Either file upload or sample_id must be provided.")

    try:
        audio, sr = load_audio(audio_path)
        duration_sec = float(len(audio) / sr)

        transcription_method = "user_provided"
        if transcript and transcript.strip():
            effective_transcript = transcript.strip()
        else:
            whisper_engine = get_whisper()
            if whisper_engine:
                try:
                    asr_res = whisper_engine.transcribe(audio_path)
                    effective_transcript = asr_res.get("text", "").strip()
                    transcription_method = "whisper_asr_auto"
                except Exception as asr_err:
                    print(f"[ASR Error] Transcription failed: {asr_err}")
                    effective_transcript = "Mera laptop start nahi ho raha, can you check the battery?"
                    transcription_method = "fallback_default"
            else:
                effective_transcript = "Mera laptop start nahi ho raha, can you check the battery?"
                transcription_method = "fallback_default"

        if not effective_transcript:
            effective_transcript = "Mera laptop start nahi ho raha, can you check the battery?"

        tokens = [t.strip() for t in effective_transcript.split() if t.strip()]

        points = find_codeswitch_points(
            audio_path=audio_path,
            transcript_tokens=tokens,
            window_ms_before=300,
            window_ms_after=300,
        )

        b_feats = extract_boundary_features(audio_path, points)
        agg_boundary_vector = aggregate_boundary_features(b_feats, aggregation="mean_and_max")

        raw_emb = projector.embed(audio_path)
        nulled_emb = projector.project(raw_emb)

        probs, ci_low, ci_high = model.predict_with_ci(
            agg_boundary_vector.reshape(1, -1),
            nulled_emb.reshape(1, -1),
            n_bootstrap=500,
            ci=0.95,
        )

        prob_val = float(probs[0])
        ci_low_val = float(ci_low[0])
        ci_high_val = float(ci_high[0])

        fused_vec = np.concatenate([agg_boundary_vector, nulled_emb])
        raw_explanations = explain_prediction(fused_vec, model._calibrator, points)

        explanations = [
            {
                "boundary_index": exp["boundary_index"],
                "timestamp_sec": float(exp["timestamp_sec"]),
                "primary_acoustic_cue": exp["primary_acoustic_cue"],
                "contribution_score": float(exp["contribution_score"]),
                "description": exp["description"]
            }
            for exp in raw_explanations
        ]

        # Evaluate two-tier operational decision policy (0.48 review, 0.52 block)
        tier_results = model.predict_operational_tier(
            agg_boundary_vector.reshape(1, -1),
            nulled_emb.reshape(1, -1),
            tau_review=0.48,
            tau_block=0.52,
        )
        tier_decision = tier_results[0]
        decision_action = tier_decision["decision"]  # PASS, STEP_UP_REVIEW, BLOCK
        risk_level = tier_decision["risk_tier"]      # LOW, SUSPICIOUS, CRITICAL

        is_spoof = prob_val >= 0.50
        classification = "SPOOF / CLONED" if is_spoof else "BONA-FIDE HUMAN"
        action_taken = "TRANSACTION_FROZEN_BLOCK" if decision_action == "BLOCK" else (
            "STEP_UP_VOICE_OTP_REQUIRED" if decision_action == "STEP_UP_REVIEW" else "CALL_CLEARED_PASS"
        )

        # Log to SaaS Audit Trail
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        audit_record = {
            "call_id": call_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "org": tenant["org"],
            "caller_id": caller_id,
            "channel": channel,
            "duration_sec": round(duration_sec, 2),
            "language_pair": "hi-en",
            "spoof_probability": round(prob_val, 3),
            "is_spoof": is_spoof,
            "action_taken": action_taken,
            "alert_level": risk_level,
        }
        CALL_AUDIT_LOGS.insert(0, audit_record)
        if len(CALL_AUDIT_LOGS) > 50:
            CALL_AUDIT_LOGS.pop()

        return {
            "success": True,
            "call_id": call_id,
            "tenant": tenant["org"],
            "classification": classification,
            "is_spoof": is_spoof,
            "spoof_probability": prob_val,
            "confidence_interval_95": [ci_low_val, ci_high_val],
            "operational_decision": {
                "decision": decision_action,
                "risk_tier": risk_level,
                "tau_review": 0.48,
                "tau_block": 0.52,
                "recommended_action": action_taken,
            },
            "action_taken": action_taken,
            "transcription": effective_transcript,
            "transcription_method": transcription_method,
            "audio_metadata": {
                "duration_sec": round(duration_sec, 2),
                "sample_rate_hz": sr,
                "detected_boundary_count": len(points)
            },
            "explanations": explanations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/api/threats")
def get_threat_logs(x_api_key: Optional[str] = Header(None)):
    """Retrieve multi-tenant live call threat activity stream."""
    _ = verify_api_key(x_api_key)
    return {
        "active_call_streams": len(CALL_AUDIT_LOGS),
        "threat_ratio_percent": round(
            sum(1 for c in CALL_AUDIT_LOGS if c["is_spoof"]) / max(1, len(CALL_AUDIT_LOGS)) * 100, 1
        ),
        "logs": CALL_AUDIT_LOGS
    }


@app.get("/api/webhooks")
def list_webhooks():
    return REGISTERED_WEBHOOKS


@app.post("/api/webhooks")
def create_webhook(config: WebhookConfig):
    wh = {
        "id": f"wh_{uuid.uuid4().hex[:6]}",
        "url": config.url,
        "events": config.events,
        "status": "ACTIVE",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    REGISTERED_WEBHOOKS.append(wh)
    return wh


@app.websocket("/ws/stream_analyze")
async def websocket_stream_analyze(websocket: WebSocket):
    """Real-time streaming WebSocket endpoint for continuous live call analysis.
    
    Accepts binary PCM / Float32 audio chunks from the client microphone,
    maintains a rolling buffer, extracts acoustic transition features,
    and streams dynamic risk probabilities and telemetry gauges.
    """
    await websocket.accept()
    model, projector, cfg = get_pipeline()
    buffer = np.zeros(0, dtype=np.float32)
    sr = 16000
    chunk_count = 0

    try:
        while True:
            # Receive binary audio chunk or JSON message
            data = await websocket.receive()
            if "bytes" in data and data["bytes"]:
                chunk_bytes = data["bytes"]
                # Convert 16-bit PCM or Float32 to float array
                try:
                    chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
                except Exception:
                    chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                buffer = np.concatenate([buffer, chunk])
                chunk_count += 1

                # Keep rolling window of last 3 seconds
                max_buffer_len = sr * 3
                if len(buffer) > max_buffer_len:
                    buffer = buffer[-max_buffer_len:]

                # Analyze every ~1 second of accumulated audio (at least 1.5s total)
                if len(buffer) >= int(1.2 * sr):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_chunk:
                        tmp_chunk_path = tmp_chunk.name
                    from vaanishield.utils.audio_io import save_audio
                    save_audio(tmp_chunk_path, buffer, sr)

                    try:
                        points = find_codeswitch_points(
                            audio_path=tmp_chunk_path,
                            transcript_tokens=["stream", "call", "audio", "buffer"],
                            window_ms_before=300,
                            window_ms_after=300,
                        )
                        b_feats = extract_boundary_features(tmp_chunk_path, points)
                        agg_vec = aggregate_boundary_features(b_feats, aggregation="mean_and_max")
                        raw_emb = projector.embed(tmp_chunk_path)
                        nulled_emb = projector.project(raw_emb)

                        probs, ci_l, ci_h = model.predict_with_ci(
                            agg_vec.reshape(1, -1),
                            nulled_emb.reshape(1, -1),
                            n_bootstrap=100,
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

            elif "text" in data:
                text_msg = data["text"]
                if text_msg == "RESET":
                    buffer = np.zeros(0, dtype=np.float32)
                    chunk_count = 0
                    await websocket.send_json({"type": "reset_ack"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
