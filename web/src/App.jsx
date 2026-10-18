import React, { useState, useEffect, useRef } from "react";

const API_BASE =
  typeof window !== "undefined" && window.location.port === "8000"
    ? ""
    : "http://127.0.0.1:8000";

// Encode an AudioBuffer into a standard 16-bit PCM RIFF .wav Blob
function encodeWavFromAudioBuffer(audioBuffer) {
  const numChannels = 1;
  const sampleRate = audioBuffer.sampleRate || 16000;
  const channelData = audioBuffer.getChannelData(0);
  const numSamples = channelData.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true);
  view.setUint16(32, numChannels * 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, numSamples * 2, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, channelData[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

export default function App() {
  const [currentPage, setCurrentPage] = useState("analyzer");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Form states
  const [selectedSample, setSelectedSample] = useState("sample_bonafide");
  const [uploadedFile, setUploadedFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(`${API_BASE}/api/audio/sample_bonafide`);
  const [transcript, setTranscript] = useState("Mera laptop start nahi ho raha, can you check the battery?");
  
  // Validation and process states
  const [formError, setFormError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [apiError, setApiError] = useState("");
  const [successMsg, setSuccessMsg] = useState("");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [showTechnicalLog, setShowTechnicalLog] = useState(false);

  // Live Microphone states
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingTimerRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);

  const fileInputRef = useRef(null);

  // Preset sample list
  const presets = [
    {
      id: "sample_bonafide",
      category: "human",
      title: "Real Voice (Hindi)",
      speaker: "Human (OpenSLR 103)",
      lang: "Hindi-English",
      transcript: "Mera laptop start nahi ho raha, can you check the battery?",
    },
    {
      id: "sample_kannada_bonafide",
      category: "human",
      title: "Real Voice (Kannada)",
      speaker: "Human (OpenSLR 79)",
      lang: "Kannada-English",
      transcript: "Nimma bank account block agide, verify your account details immediately.",
    },
    {
      id: "sample_cloned",
      category: "clone",
      title: "AI Clone (Hinglish)",
      speaker: "Neural Synthesizer",
      lang: "Hindi-English XTTS",
      transcript: "Mera laptop start nahi ho raha, can you check the battery?",
    },
    {
      id: "sample_tamil_cloned",
      category: "clone",
      title: "AI Clone (Tamil)",
      speaker: "Neural Synthesizer",
      lang: "Tamil-English",
      transcript: "Unga bank account freeze aayiduchu, verify your details immediately.",
    },
    {
      id: "sample_codec",
      category: "clone",
      title: "AI Clone (8kHz Codec)",
      speaker: "PSTN Telephony G.711",
      lang: "Telephony Codec",
      transcript: "Mera laptop start nahi ho raha, can you check the battery?",
    },
    {
      id: "sample_heldout_cloned",
      category: "clone",
      title: "AI Clone (Held-Out)",
      speaker: "Unseen Speaker 5945",
      lang: "Hindi-English",
      transcript: "Mera account freeze ho gaya hai, please verify the transaction now.",
    },
  ];

  // Handle preset select
  const selectPreset = (p) => {
    setSelectedSample(p.id);
    setTranscript(p.transcript);
    setUploadedFile(null);
    setAudioUrl(`${API_BASE}/api/audio/${p.id}`);
    setFormError("");
    setApiError("");
    setSuccessMsg("");
    setAnalysisResult(null);
  };

  // Live mic visualizer
  const startVisualizer = (stream) => {
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      audioContextRef.current = audioCtx;
      analyserRef.current = analyser;

      const canvas = canvasRef.current;
      if (!canvas) return;
      const canvasCtx = canvas.getContext("2d");
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const draw = () => {
        animationFrameRef.current = requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(dataArray);

        canvasCtx.fillStyle = "#090c10";
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = "#10b981";
        canvasCtx.beginPath();

        const sliceWidth = (canvas.width * 1.0) / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0;
          const y = (v * canvas.height) / 2;
          if (i === 0) canvasCtx.moveTo(x, y);
          else canvasCtx.lineTo(x, y);
          x += sliceWidth;
        }

        canvasCtx.lineTo(canvas.width, canvas.height / 2);
        canvasCtx.stroke();
      };

      draw();
    } catch (e) {
      console.warn("Visualizer init skipped:", e);
    }
  };

  const stopVisualizer = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
    }
  };

  // Start microphone
  const startRecording = async () => {
    try {
      setFormError("");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const rawBlob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType || "audio/webm" });
        stream.getTracks().forEach((t) => t.stop());
        stopVisualizer();

        try {
          const decodeCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
          const arrayBuffer = await rawBlob.arrayBuffer();
          const decoded = await decodeCtx.decodeAudioData(arrayBuffer);
          const wavBlob = encodeWavFromAudioBuffer(decoded);
          const wavFile = new File([wavBlob], `mic_${Date.now()}.wav`, { type: "audio/wav" });

          setUploadedFile(wavFile);
          setSelectedSample("");
          setAudioUrl(URL.createObjectURL(wavBlob));
          setSuccessMsg(`Recorded ${(decoded.duration).toFixed(1)}s audio.`);
          decodeCtx.close().catch(() => {});
        } catch (convErr) {
          const fallbackFile = new File([rawBlob], "mic_input.wav", { type: rawBlob.type });
          setUploadedFile(fallbackFile);
          setSelectedSample("");
          setAudioUrl(URL.createObjectURL(rawBlob));
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingDuration(0);
      startVisualizer(stream);

      recordingTimerRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      setFormError(`Microphone access error: ${err.message}`);
    }
  };

  // Stop microphone
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(recordingTimerRef.current);
    }
  };

  // Handle file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".wav")) {
      setFormError("Please upload an uncompressed .wav audio file.");
      return;
    }

    setUploadedFile(file);
    setSelectedSample("");
    setAudioUrl(URL.createObjectURL(file));
    setFormError("");
    setApiError("");
    setSuccessMsg(`Loaded ${file.name}`);
  };

  // Run detection analysis
  const runAnalysis = async () => {
    setFormError("");
    setApiError("");
    setSuccessMsg("");

    if (!selectedSample && !uploadedFile) {
      setFormError("Select an audio preset or record/upload audio.");
      return;
    }

    setIsLoading(true);
    const formData = new FormData();

    if (uploadedFile) {
      formData.append("file", uploadedFile);
    } else {
      formData.append("sample_id", selectedSample);
    }

    if (transcript.trim()) {
      formData.append("transcript", transcript.trim());
    }

    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with ${res.status}`);
      }

      const data = await res.json();
      setAnalysisResult(data);
    } catch (err) {
      setApiError(err.message || "Failed to analyze audio.");
    } finally {
      setIsLoading(false);
    }
  };

  const navigateTo = (page) => {
    setCurrentPage(page);
    setMobileMenuOpen(false);
  };

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="site-header">
        <div className="brand-group">
          <a
            href="#detector"
            onClick={(e) => {
              e.preventDefault();
              navigateTo("analyzer");
            }}
            className="brand-logo"
          >
            <span className="brand-icon">🛡️</span>
            <span>VaaniShield</span>
          </a>
        </div>

        <nav className="nav-links">
          <button
            type="button"
            className={currentPage === "analyzer" ? "active" : ""}
            onClick={() => navigateTo("analyzer")}
          >
            🛡️ Voice Detector
          </button>
          <button
            type="button"
            className={currentPage === "pipeline" ? "active" : ""}
            onClick={() => navigateTo("pipeline")}
          >
            ⚡ Pipeline Flow
          </button>
          <button
            type="button"
            className={currentPage === "specs" ? "active" : ""}
            onClick={() => navigateTo("specs")}
          >
            🔬 Research & Evaluation
          </button>
        </nav>

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span className="brand-badge" style={{ color: "#10b981", borderColor: "rgba(16, 185, 129, 0.3)" }}>
            ENGINE ACTIVE
          </span>
          <button
            type="button"
            className="mobile-menu-btn"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? "✕" : "☰"}
          </button>
        </div>

        <div className={`mobile-nav ${mobileMenuOpen ? "open" : ""}`}>
          <button type="button" onClick={() => navigateTo("analyzer")}>🛡️ Voice Detector</button>
          <button type="button" onClick={() => navigateTo("pipeline")}>⚡ Pipeline Flow</button>
          <button type="button" onClick={() => navigateTo("specs")}>🔬 Research & Evaluation</button>
        </div>
      </header>

      {/* Main View: Voice Detector */}
      {currentPage === "analyzer" && (
        <main className="main-dashboard">
          {/* Left Column: Audio Controls & Presets */}
          <section className="dashboard-col left-col">
            <div className="control-card">
              <h3 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "8px" }}>
                1. Test Audio Sample
              </h3>

              <div className="sample-grid">
                {presets.map((p) => {
                  const isSel = selectedSample === p.id && !uploadedFile;
                  return (
                    <button
                      key={p.id}
                      type="button"
                      className={`sample-btn ${isSel ? "active" : ""}`}
                      onClick={() => selectPreset(p)}
                    >
                      <div className="sample-title">{p.title}</div>
                      <div className="sample-sub">{p.speaker} &bull; {p.lang}</div>
                    </button>
                  );
                })}
              </div>

              {/* Upload & Mic Actions */}
              <div style={{ display: "flex", gap: "8px", marginTop: "10px" }}>
                <input
                  type="file"
                  accept=".wav"
                  ref={fileInputRef}
                  style={{ display: "none" }}
                  onChange={handleFileUpload}
                />
                <button
                  type="button"
                  className="btn-secondary"
                  style={{ flex: 1, padding: "8px" }}
                  onClick={() => fileInputRef.current.click()}
                >
                  📁 Upload .WAV
                </button>

                {!isRecording ? (
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{ flex: 1, padding: "8px", borderColor: "#ef4444", color: "#ef4444" }}
                    onClick={startRecording}
                  >
                    🎙️ Record Mic
                  </button>
                ) : (
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{ flex: 1, padding: "8px", background: "rgba(239, 68, 68, 0.2)", color: "#fca5a5" }}
                    onClick={stopRecording}
                  >
                    ⏹️ Stop ({recordingDuration}s)
                  </button>
                )}
              </div>

              {/* Live Canvas for mic */}
              {isRecording && (
                <div style={{ marginTop: "8px", borderRadius: "6px", overflow: "hidden", border: "1px solid #334155" }}>
                  <canvas ref={canvasRef} width={380} height={40} style={{ display: "block", width: "100%" }} />
                </div>
              )}

              {/* Audio Player */}
              {audioUrl && (
                <div style={{ marginTop: "12px", background: "#090c10", padding: "8px", borderRadius: "6px" }}>
                  <audio
                    src={audioUrl}
                    controls
                    style={{ width: "100%", height: "32px", display: "block" }}
                  />
                </div>
              )}

              {/* Transcript context (optional) */}
              <div style={{ marginTop: "10px" }}>
                <label style={{ fontSize: "11px", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>
                  TRANSCRIPT CONTEXT (FOR CODE-SWITCH BOUNDARY ALIGNMENT)
                </label>
                <input
                  type="text"
                  value={transcript}
                  onChange={(e) => setTranscript(e.target.value)}
                  placeholder="e.g. Mera laptop start nahi ho raha..."
                  style={{
                    width: "100%",
                    padding: "8px 10px",
                    background: "#090c10",
                    border: "1px solid var(--border-color)",
                    color: "var(--text-primary)",
                  }}
                />
              </div>

              {/* Error & Success Messages */}
              {formError && <div className="banner error-banner">{formError}</div>}
              {apiError && <div className="banner error-banner">{apiError}</div>}
              {successMsg && <div className="banner success-banner">{successMsg}</div>}

              {/* Execute Scan Button */}
              <button
                type="button"
                className="btn-scan"
                disabled={isLoading}
                onClick={runAnalysis}
                style={{ width: "100%", marginTop: "12px" }}
              >
                {isLoading ? "EXTRACTING DUAL-STREAM ACOUSTIC RESIDUALS..." : "🔍 VERIFY VOICE AUTHENTICITY"}
              </button>
            </div>
          </section>

          {/* Right Column: Detection Verdict & Forensic Explanations */}
          <section className="dashboard-col right-col">
            {!analysisResult && !isLoading && (
              <div
                className="control-card"
                style={{
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  textAlign: "center",
                  color: "var(--text-muted)",
                }}
              >
                <div style={{ fontSize: "36px", marginBottom: "8px" }}>🛡️</div>
                <h3 style={{ color: "var(--text-primary)", marginBottom: "4px" }}>
                  Ready to Inspect Voice Clip
                </h3>
                <p style={{ maxWidth: "340px", fontSize: "12px", lineHeight: "1.5" }}>
                  Select one of the 6 test audio samples on the left or record audio. Click <strong>Verify Voice Authenticity</strong> to inspect boundary phase discontinuities and speaker-nulled residuals.
                </p>
              </div>
            )}

            {isLoading && (
              <div
                className="control-card"
                style={{
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  textAlign: "center",
                }}
              >
                <div style={{ fontSize: "28px", animation: "spin 1.5s linear infinite" }}>⚙️</div>
                <h3 style={{ marginTop: "12px", color: "var(--text-primary)" }}>
                  Analyzing Acoustic Micro-Seams...
                </h3>
                <p style={{ color: "var(--text-muted)", fontSize: "12px", marginTop: "4px" }}>
                  Checking pitch jumps (F0), formant trajectories, and SNAP orthogonal subspace.
                </p>
              </div>
            )}

            {analysisResult && (
              <div className="control-card" style={{ height: "100%", overflowY: "auto" }}>
                {/* Big Result Badge */}
                <div
                  className={`verdict-box ${
                    analysisResult.is_spoof
                      ? "spoof"
                      : analysisResult.operational_decision?.risk_tier === "ELEVATED"
                      ? "suspicious"
                      : "bonafide"
                  }`}
                >
                  <div className="verdict-icon">
                    {analysisResult.is_spoof ? "🚨" : analysisResult.operational_decision?.risk_tier === "ELEVATED" ? "⚠️" : "✅"}
                  </div>
                  <div>
                    <div className="verdict-title">
                      {analysisResult.is_spoof
                        ? "SYNTHETIC AI VOICE CLONE"
                        : analysisResult.operational_decision?.risk_tier === "ELEVATED"
                        ? "UNCERTAIN / STEP-UP VERIFICATION"
                        : "AUTHENTIC HUMAN VOICE"}
                    </div>
                    <div className="verdict-desc">
                      {analysisResult.is_spoof
                        ? "Acoustic seam discontinuities and neural vocoder spectral artifacts detected."
                        : analysisResult.operational_decision?.risk_tier === "ELEVATED"
                        ? "Marginal acoustic confidence under channel noise; step-up biometric prompt suggested."
                        : "Continuous natural vocal tract dynamics and zero neural synthesis seams."}
                    </div>
                  </div>
                </div>

                {/* Metrics Bar */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "8px", margin: "12px 0" }}>
                  <div className="risk-meter-box">
                    <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>SPOOF PROBABILITY</div>
                    <div
                      style={{
                        fontSize: "20px",
                        fontWeight: 800,
                        color: analysisResult.is_spoof ? "#f87171" : "#4ade80",
                      }}
                    >
                      {(analysisResult.spoof_probability * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div className="risk-meter-box">
                    <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>95% BOOTSTRAP CI</div>
                    <div className="mono-text" style={{ fontSize: "12px", marginTop: "4px" }}>
                      [{(analysisResult.confidence_interval_95[0] * 100).toFixed(1)}%, {(analysisResult.confidence_interval_95[1] * 100).toFixed(1)}%]
                    </div>
                  </div>

                  <div className="risk-meter-box">
                    <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>POLICY TIER</div>
                    <div style={{ fontSize: "12px", fontWeight: 700, color: "var(--text-primary)", marginTop: "4px" }}>
                      {analysisResult.operational_decision?.decision || "PASS"}
                    </div>
                  </div>
                </div>

                {/* Waveform & Boundary Markers */}
                <div style={{ margin: "12px 0" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text-muted)", marginBottom: "4px" }}>
                    <span>ACOUSTIC ENERGY & BOUNDARY WINDOW LOCATIONS</span>
                    <span>{analysisResult.audio_metadata?.duration_sec}s duration</span>
                  </div>
                  <div
                    style={{
                      height: "44px",
                      background: "#090c10",
                      borderRadius: "6px",
                      position: "relative",
                      display: "flex",
                      alignItems: "center",
                      padding: "0 6px",
                      gap: "2px",
                      border: "1px solid var(--border-color)",
                    }}
                  >
                    {(analysisResult.audio_metadata?.waveform_peaks || []).map((val, idx) => (
                      <div
                        key={idx}
                        style={{
                          flex: 1,
                          height: `${Math.max(4, Math.min(38, val * 38))}px`,
                          background: analysisResult.is_spoof ? "#ef4444" : "#10b981",
                          opacity: 0.75,
                          borderRadius: "1px",
                        }}
                      />
                    ))}

                    {/* Boundary Marker Lines */}
                    {(analysisResult.explanations || []).map((exp, eIdx) => {
                      const dur = analysisResult.audio_metadata?.duration_sec || 5.0;
                      const pct = Math.min(98, Math.max(2, (exp.timestamp_sec / dur) * 100));
                      return (
                        <div
                          key={eIdx}
                          title={`Boundary #${exp.boundary_index}: ${exp.language_transition} at ${exp.timestamp_sec}s`}
                          style={{
                            position: "absolute",
                            left: `${pct}%`,
                            top: 0,
                            bottom: 0,
                            width: "2px",
                            background: "#f59e0b",
                            boxShadow: "0 0 6px #f59e0b",
                            zIndex: 10,
                          }}
                        />
                      );
                    })}
                  </div>
                </div>

                {/* Acoustic Explanations List (Zero Transcript Content) */}
                <div style={{ marginTop: "12px" }}>
                  <div style={{ fontSize: "12px", fontWeight: 700, marginBottom: "6px", display: "flex", alignItems: "center", gap: "6px" }}>
                    <span>🔬 Acoustic Seam Attributions</span>
                    <span style={{ fontSize: "10px", color: "var(--text-muted)", fontWeight: 400 }}>
                      (Zero transcript content stored / DPDPA compliant)
                    </span>
                  </div>

                  {(!analysisResult.explanations || analysisResult.explanations.length === 0) ? (
                    <div style={{ fontSize: "12px", color: "var(--text-muted)", padding: "8px", background: "#090c10", borderRadius: "6px" }}>
                      Monolingual or continuous vocal tract contour — no cross-lingual boundary seam detected.
                    </div>
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                      {analysisResult.explanations.map((exp, idx) => (
                        <div
                          key={idx}
                          style={{
                            background: "#090c10",
                            border: "1px solid var(--border-color)",
                            borderRadius: "6px",
                            padding: "8px 10px",
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                          }}
                        >
                          <div>
                            <div style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-primary)" }}>
                              Boundary #{exp.boundary_index} &bull; {exp.language_transition} @ {exp.timestamp_sec}s
                            </div>
                            <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
                              {exp.description}
                            </div>
                          </div>
                          <div style={{ textAlign: "right" }}>
                            <span
                              style={{
                                fontSize: "11px",
                                fontWeight: 700,
                                padding: "2px 6px",
                                borderRadius: "4px",
                                background: exp.contribution_score > 0.6 ? "rgba(239, 68, 68, 0.2)" : "rgba(16, 185, 129, 0.2)",
                                color: exp.contribution_score > 0.6 ? "#fca5a5" : "#86efac",
                              }}
                            >
                              {(exp.contribution_score * 100).toFixed(0)}% Seam Anomaly
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Diagnostics Toggle */}
                <div style={{ marginTop: "12px" }}>
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{ fontSize: "11px", padding: "4px 8px" }}
                    onClick={() => setShowTechnicalLog(!showTechnicalLog)}
                  >
                    {showTechnicalLog ? "Hide Diagnostic Vector" : "Show Diagnostic Vector"}
                  </button>

                  {showTechnicalLog && (
                    <pre
                      className="mono-text"
                      style={{
                        marginTop: "8px",
                        padding: "8px",
                        background: "#050709",
                        borderRadius: "6px",
                        fontSize: "11px",
                        color: "#94a3b8",
                        overflowX: "auto",
                      }}
                    >
                      {JSON.stringify(
                        {
                          decision_action: analysisResult.operational_decision?.decision,
                          audio_metadata: analysisResult.audio_metadata,
                          confidence_interval_95: analysisResult.confidence_interval_95,
                          explanations: analysisResult.explanations,
                        },
                        null,
                        2
                      )}
                    </pre>
                  )}
                </div>
              </div>
            )}
          </section>
        </main>
      )}

      {/* View 2: Pipeline Flow & Architecture */}
      {currentPage === "pipeline" && (
        <article className="page-article">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
            <div>
              <h2>VaaniShield End-to-End Pipeline Architecture</h2>
              <p>How VaaniShield tracks synthetic speech and prevents voice fraud in Indian multilingual calls.</p>
            </div>
            <button type="button" className="btn-scan" onClick={() => navigateTo("analyzer")}>
              TRY LIVE DETECTOR
            </button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px", marginBottom: "20px" }}>
            <div className="control-card" style={{ padding: "14px" }}>
              <div style={{ fontSize: "20px", marginBottom: "6px" }}>📡 Step 1</div>
              <h4 style={{ fontSize: "13px", color: "var(--text-primary)", marginBottom: "4px" }}>Audio Ingestion</h4>
              <p style={{ fontSize: "12px", color: "var(--text-muted)", lineHeight: "1.4" }}>
                Ingests streaming 8 kHz PSTN / VoIP telephony or 16 kHz HD speech. Downsamples and strips loudness offsets.
              </p>
            </div>

            <div className="control-card" style={{ padding: "14px" }}>
              <div style={{ fontSize: "20px", marginBottom: "6px" }}>📍 Step 2</div>
              <h4 style={{ fontSize: "13px", color: "var(--text-primary)", marginBottom: "4px" }}>Boundary Localization</h4>
              <p style={{ fontSize: "12px", color: "var(--text-muted)", lineHeight: "1.4" }}>
                Identifies code-switch points (e.g. Hindi &rarr; English). Extracts localized ±300ms analysis windows around transitions.
              </p>
            </div>

            <div className="control-card" style={{ padding: "14px" }}>
              <div style={{ fontSize: "20px", marginBottom: "6px" }}>🔬 Step 3</div>
              <h4 style={{ fontSize: "13px", color: "var(--text-primary)", marginBottom: "4px" }}>Dual-Stream Extraction</h4>
              <p style={{ fontSize: "12px", color: "var(--text-muted)", lineHeight: "1.4" }}>
                <strong>Stream A:</strong> Boundary micro-seams (F0 reset, formant transitions).<br />
                <strong>Stream B:</strong> SNAP speaker-nulling orthogonal projection.
              </p>
            </div>

            <div className="control-card" style={{ padding: "14px" }}>
              <div style={{ fontSize: "20px", marginBottom: "6px" }}>⚖️ Step 4</div>
              <h4 style={{ fontSize: "13px", color: "var(--text-primary)", marginBottom: "4px" }}>Calibrated Fusion</h4>
              <p style={{ fontSize: "12px", color: "var(--text-muted)", lineHeight: "1.4" }}>
                Calibrated logistic regression produces continuous probabilities with 95% bootstrap CIs and operational risk tiers.
              </p>
            </div>
          </div>

          <div className="control-card" style={{ padding: "16px", marginBottom: "16px" }}>
            <h3 style={{ fontSize: "14px", marginBottom: "8px" }}>Why Both Streams are Necessary: The Indian Telephony Dilemma</h3>
            <p style={{ fontSize: "12px", color: "var(--text-secondary)", lineHeight: "1.6" }}>
              1. <strong>Speaker Entanglement Problem:</strong> Standard deep learning models trained on voice samples easily memorize individual speakers or accent characteristics rather than synthesis artifacts. Our <em>Speaker-Nulling Artifact Projection (SNAP)</em> projects embeddings onto the orthogonal complement of the between-speaker subspace, stripping identity and preventing false alarms on non-standard Indian accents.<br />
              2. <strong>Intra-Utterance Boundary Seams:</strong> Voice cloning models frequently fail when transitioning between languages mid-sentence (e.g., Hinglish), leaving subtle phase resets and unnatural pitch jumps at switch points. Our boundary stream detects these localized anomalies even under compression.
            </p>
          </div>
        </article>
      )}

      {/* View 3: Technical & Research Specs */}
      {currentPage === "specs" && (
        <article className="page-article">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
            <div>
              <h2>Scientific Evaluation & Ablation Results</h2>
              <p>Empirical metrics computed across 4 strictly speaker-disjoint splits (N=400 recordings).</p>
            </div>
            <button type="button" className="btn-scan" onClick={() => navigateTo("analyzer")}>
              TRY LIVE DETECTOR
            </button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px", marginBottom: "20px" }}>
            <div className="risk-meter-box">
              <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>IN-DOMAIN TEST EER</div>
              <div style={{ fontSize: "20px", fontWeight: 800, color: "#10b981" }}>14.00%</div>
              <div className="mono-text" style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>95% CI: [2.0%, 30.0%]</div>
            </div>
            <div className="risk-meter-box">
              <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>HELDOUT SPEAKER EER</div>
              <div style={{ fontSize: "20px", fontWeight: 800, color: "#10b981" }}>2.00%</div>
              <div className="mono-text" style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>95% CI: [0.0%, 18.0%]</div>
            </div>
            <div className="risk-meter-box">
              <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>VALIDATION EER</div>
              <div style={{ fontSize: "20px", fontWeight: 800, color: "#38bdf8" }}>0.00%</div>
              <div className="mono-text" style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>95% CI: [0.0%, 0.0%]</div>
            </div>
          </div>

          <h3>1. Model Comparison across Splits (N=400)</h3>
          <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "8px" }}>
            Zero speaker or utterance overlap across splits. Verified generalization without speaker memorization:
          </p>
          <div className="table-wrapper" style={{ marginBottom: "20px" }}>
            <table className="raw-table">
              <thead>
                <tr>
                  <th>Model Configuration</th>
                  <th>Val EER</th>
                  <th>Test EER (In-Domain)</th>
                  <th>Held-Out Speaker EER</th>
                  <th>Test Precision / Recall</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Baseline A (Generic Acoustic + LR)</td>
                  <td>24.0%</td>
                  <td>39.0%</td>
                  <td>32.0%</td>
                  <td>66.0% / 62.0%</td>
                </tr>
                <tr>
                  <td>Baseline B (Indian-Accent Mel + LFCC)</td>
                  <td>34.0%</td>
                  <td>48.0%</td>
                  <td>22.0%</td>
                  <td>56.9% / 66.0%</td>
                </tr>
                <tr style={{ background: "rgba(16, 185, 129, 0.1)" }}>
                  <td><strong>VaaniShield Full Fused Model</strong></td>
                  <td style={{ color: "#10b981", fontWeight: 700 }}>0.0%</td>
                  <td style={{ color: "#10b981", fontWeight: 700 }}>14.0%</td>
                  <td style={{ color: "#10b981", fontWeight: 700 }}>2.0%</td>
                  <td style={{ color: "#10b981", fontWeight: 700 }}>88.4% / 76.0%</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3>2. Three-Way Ablation Analysis (Anti-Shortcut Verified)</h3>
          <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "8px" }}>
            Evaluating individual and combined stream contributions across test and held-out sets:
          </p>
          <div className="table-wrapper" style={{ marginBottom: "20px" }}>
            <table className="raw-table">
              <thead>
                <tr>
                  <th>Stream Variant</th>
                  <th>Test EER</th>
                  <th>Held-Out Speaker EER</th>
                  <th>Finding & Role in Architecture</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Boundary-Local Stream Only</td>
                  <td>51.0%</td>
                  <td>52.0%</td>
                  <td>Under narrowband telephony loss without vocoder conditioning, boundary cues alone operate near chance.</td>
                </tr>
                <tr>
                  <td>Speaker-Nulled Global Stream Only</td>
                  <td>22.0%</td>
                  <td>6.0%</td>
                  <td>Orthogonal projection removes speaker identity and detects vocoder synthesis artifacts.</td>
                </tr>
                <tr style={{ background: "rgba(16, 185, 129, 0.1)" }}>
                  <td><strong>VaaniShield Fused Model</strong></td>
                  <td style={{ color: "#10b981", fontWeight: 700 }}>14.0%</td>
                  <td style={{ color: "#10b981", fontWeight: 700 }}>2.0%</td>
                  <td>Fusing boundary transition dynamics with speaker-nulled residuals yields lowest overall EER.</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3>3. Zero Transcript Storage & DPDPA 2023 Compliance</h3>
          <p style={{ fontSize: "12px", color: "var(--text-secondary)", lineHeight: "1.6" }}>
            In compliance with India's Digital Personal Data Protection Act (DPDPA 2023), VaaniShield performs strictly non-semantic acoustic analysis. Audio is evaluated strictly for physical vocal tract and vocoder anomalies; speech content is never recorded, transcribed, or stored in persistent databases.
          </p>
        </article>
      )}

      {/* Footer */}
      <footer className="site-footer">
        <div>
          &copy; 2026 VaaniShield Research &bull; IIT Delhi Research Park
        </div>
        <div>
          Privacy-Preserving Acoustic Voice Clone Detection &bull; Zero Transcript Storage
        </div>
      </footer>
    </div>
  );
}
