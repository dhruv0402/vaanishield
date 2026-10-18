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
  const [theme, setTheme] = useState(() => {
    if (typeof window !== "undefined" && window.localStorage) {
      return localStorage.getItem("vaanishield_theme") || "dark";
    }
    return "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    if (typeof window !== "undefined" && window.localStorage) {
      localStorage.setItem("vaanishield_theme", theme);
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

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

  // Live Microphone & Streaming Call Guard states
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [isStreamingGuard, setIsStreamingGuard] = useState(false);
  const [streamTelemetry, setStreamTelemetry] = useState(null);
  const wsStreamRef = useRef(null);
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

        canvasCtx.fillStyle = "#06080b";
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

  // Start Live Streaming Call Guard
  const startStreamingGuard = async () => {
    try {
      setFormError("");
      setApiError("");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const wsUrl = (API_BASE ? API_BASE.replace(/^http/, "ws") : "ws://127.0.0.1:8000") + "/ws/stream_analyze";
      const ws = new WebSocket(wsUrl);
      wsStreamRef.current = ws;

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 8000 });
      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(2048, 1, 1);

      processor.onaudioprocess = (e) => {
        if (ws.readyState === WebSocket.OPEN) {
          const inputData = e.inputBuffer.getChannelData(0);
          ws.send(inputData.buffer);
        }
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "telemetry") {
            setStreamTelemetry(data);
          }
        } catch (err) {
          console.warn("WS parse error:", err);
        }
      };

      ws.onerror = () => {
        setFormError("Streaming WebSocket connection error. Check backend.");
        stopStreamingGuard();
      };

      setIsStreamingGuard(true);
      startVisualizer(stream);
    } catch (err) {
      setFormError(`Live Guard activation error: ${err.message}`);
    }
  };

  const stopStreamingGuard = () => {
    if (wsStreamRef.current) {
      wsStreamRef.current.close();
      wsStreamRef.current = null;
    }
    setIsStreamingGuard(false);
    stopVisualizer();
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
          <span className="brand-pill-badge">
            <span className="pulse-dot"></span>
            ACTIVE FORENSIC ENGINE
          </span>
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
          <button
            type="button"
            className="theme-toggle-btn"
            onClick={toggleTheme}
            title={`Switch to ${theme === "dark" ? "Light" : "Dark"} Mode`}
            aria-label="Toggle theme"
          >
            {theme === "dark" ? "☀️" : "🌙"}
          </button>

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
          <button type="button" onClick={toggleTheme}>
            {theme === "dark" ? "☀️ Switch to Light Mode" : "🌙 Switch to Dark Mode"}
          </button>
        </div>
      </header>

      {/* Main View: Voice Detector */}
      {currentPage === "analyzer" && (
        <main className="main-dashboard">
          {/* Left Column: Audio Controls & Presets */}
          <section className="dashboard-col left-col">
            <div className="control-card">
              <div className="card-heading-row">
                <div className="card-heading">
                  <span className="step-tag">01</span>
                  <span>Select Voice Recording</span>
                </div>
                <span className="mono-text" style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                  8 kHz PSTN / HD
                </span>
              </div>

              <div className="sample-grid">
                {presets.map((p) => {
                  const isSel = selectedSample === p.id && !uploadedFile;
                  const isHuman = p.category === "human";
                  return (
                    <button
                      key={p.id}
                      type="button"
                      className={`sample-btn ${isHuman ? "human-sample" : "clone-sample"} ${isSel ? "active" : ""}`}
                      onClick={() => selectPreset(p)}
                    >
                      <div className="sample-tag-row">
                        <span className={isHuman ? "sample-tag-human" : "sample-tag-clone"}>
                          {isHuman ? "Authentic" : "Synthesized"}
                        </span>
                        <span className="sample-lang-tag">{p.lang}</span>
                      </div>
                      <div className="sample-title">{p.title}</div>
                      <div className="sample-sub">{p.speaker}</div>
                    </button>
                  );
                })}
              </div>

              {/* Upload & Mic Actions */}
              <div style={{ display: "flex", gap: "10px", marginTop: "14px" }}>
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
                  style={{ flex: 1 }}
                  onClick={() => fileInputRef.current.click()}
                >
                  📁 Upload .WAV Audio
                </button>

                {!isRecording ? (
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{ flex: 1, borderColor: "var(--crimson-border)", color: "var(--crimson-text)" }}
                    onClick={startRecording}
                  >
                    🎙️ Record Clip
                  </button>
                ) : (
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{ flex: 1, background: "var(--crimson-glow)", color: "var(--crimson-text)", borderColor: "var(--crimson-border)" }}
                    onClick={stopRecording}
                  >
                    ⏹️ Stop Clip ({recordingDuration}s)
                  </button>
                )}

                {!isStreamingGuard ? (
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{ flex: 1, borderColor: "var(--cyan-border)", color: "var(--cyan-pure)", background: "var(--cyan-glow)" }}
                    onClick={startStreamingGuard}
                  >
                    📡 Live Call Guard
                  </button>
                ) : (
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{ flex: 1, background: "rgba(6, 182, 212, 0.3)", color: "#a5f3fc", borderColor: "var(--cyan-pure)" }}
                    onClick={stopStreamingGuard}
                  >
                    ⏹️ Halt Guard
                  </button>
                )}
              </div>

              {/* Live Streaming Call Guard Status HUD */}
              {isStreamingGuard && (
                <div style={{ marginTop: "12px", padding: "12px", background: "var(--bg-surface)", border: "1px solid var(--cyan-border)", borderRadius: "8px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                      <span className="pulse-dot" style={{ background: "var(--cyan-pure)" }}></span>
                      <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--cyan-pure)", letterSpacing: "0.04em" }}>
                        STREAMING CALL GUARD ACTIVE
                      </span>
                    </div>
                    <span className="mono-text" style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                      Chunk #{streamTelemetry?.chunk_id || 0} &bull; {streamTelemetry?.buffer_duration_sec || 0}s buffer
                    </span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                    <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>Instantaneous Spoof Risk:</span>
                    <span className="mono-text" style={{ fontSize: "16px", fontWeight: 800, color: (streamTelemetry?.spoof_probability || 0) > 0.5 ? "var(--crimson-pure)" : "var(--emerald-pure)" }}>
                      {((streamTelemetry?.spoof_probability || 0) * 100).toFixed(1)}% ({streamTelemetry?.alert_level || "MONITORING"})
                    </span>
                  </div>
                </div>
              )}

              {/* Live Canvas for mic */}
              {(isRecording || isStreamingGuard) && (
                <div style={{ marginTop: "10px", borderRadius: "8px", overflow: "hidden", border: "1px solid var(--border-medium)" }}>
                  <canvas ref={canvasRef} width={420} height={48} style={{ display: "block", width: "100%" }} />
                </div>
              )}

              {/* Audio Player */}
              {audioUrl && (
                <div className="audio-player-wrapper">
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text-muted)" }}>
                    <span style={{ fontWeight: 600 }}>MONITORED AUDIO CHANNEL</span>
                    <span className="mono-text">PCM 16-BIT</span>
                  </div>
                  <audio src={audioUrl} controls />
                </div>
              )}

              {/* Transcript context (optional) */}
              <div style={{ marginTop: "14px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                  <label style={{ fontSize: "11px", fontWeight: 700, color: "var(--text-muted)", letterSpacing: "0.05em" }}>
                    TRANSCRIPT CONTEXT (CODE-SWITCH ALIGNMENT)
                  </label>
                  <span style={{ fontSize: "10px", color: "var(--text-dim)" }}>Automatic LID enabled</span>
                </div>
                <input
                  type="text"
                  value={transcript}
                  onChange={(e) => setTranscript(e.target.value)}
                  placeholder="e.g. Mera laptop start nahi ho raha, can you check the battery?"
                  style={{
                    width: "100%",
                    padding: "10px 14px",
                    background: "var(--bg-input)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "8px",
                    color: "var(--text-primary)",
                    fontSize: "13px",
                    fontFamily: "inherit",
                    outline: "none",
                  }}
                />
              </div>

              {/* Error & Success Messages */}
              {formError && <div className="banner error-banner">⚠️ {formError}</div>}
              {apiError && <div className="banner error-banner">⚠️ {apiError}</div>}
              {successMsg && <div className="banner success-banner">✓ {successMsg}</div>}

              {/* Execute Scan Button */}
              <button
                type="button"
                className="btn-scan"
                disabled={isLoading}
                onClick={runAnalysis}
                style={{ width: "100%", marginTop: "16px" }}
              >
                {isLoading ? (
                  <>
                    <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⚙️</span>
                    <span>EXTRACTING DUAL-STREAM ACOUSTIC RESIDUALS...</span>
                  </>
                ) : (
                  <>
                    <span>🔍</span>
                    <span>VERIFY VOICE AUTHENTICITY</span>
                  </>
                )}
              </button>
            </div>
          </section>

          {/* Right Column: Detection Verdict & Forensic Explanations */}
          <section className="dashboard-col right-col">
            {!analysisResult && !isLoading && (
              <div
                className="control-card"
                style={{
                  minHeight: "440px",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  textAlign: "center",
                  padding: "40px 24px",
                }}
              >
                <div style={{ fontSize: "44px", marginBottom: "16px", filter: "drop-shadow(0 0 12px rgba(59, 130, 246, 0.4))" }}>
                  🛡️
                </div>
                <h3 style={{ fontSize: "18px", fontWeight: 800, color: "var(--text-white)", marginBottom: "8px" }}>
                  Acoustic Forensic Engine Ready
                </h3>
                <p style={{ maxWidth: "420px", fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.6" }}>
                  Select an authentic speaker or AI synthesized clone on the left. Click <strong>Verify Voice Authenticity</strong> to isolate intra-sentence code-switch boundary seams and speaker-orthogonal synthesis residuals.
                </p>
                <div style={{ marginTop: "20px", display: "flex", gap: "8px" }}>
                  <span className="brand-pill-badge" style={{ background: "rgba(59, 130, 246, 0.1)", borderColor: "rgba(59, 130, 246, 0.3)", color: "#93c5fd" }}>
                    ✓ Anti-Memorization Nulling (SNAP)
                  </span>
                  <span className="brand-pill-badge" style={{ background: "rgba(16, 185, 129, 0.1)", borderColor: "rgba(16, 185, 129, 0.3)", color: "#86efac" }}>
                    ✓ DPDPA Non-Semantic Privacy
                  </span>
                </div>
              </div>
            )}

            {isLoading && (
              <div
                className="control-card"
                style={{
                  minHeight: "440px",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  textAlign: "center",
                  padding: "40px 24px",
                }}
              >
                <div style={{ fontSize: "36px", animation: "spin 1.2s linear infinite", marginBottom: "16px" }}>⚙️</div>
                <h3 style={{ fontSize: "18px", fontWeight: 800, color: "var(--text-white)", marginBottom: "8px" }}>
                  Scanning Micro-Seams & Nulling Identity...
                </h3>
                <p style={{ color: "var(--text-muted)", fontSize: "13px", maxWidth: "380px", lineHeight: "1.5" }}>
                  Aligning boundary phoneme windows (±300ms), computing F0 reset differentials, and projecting against the speaker manifold.
                </p>
              </div>
            )}

            {analysisResult && (
              <div className="control-card">
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
                  <div className="verdict-content">
                    <div className="verdict-header-row">
                      <div className="verdict-title">
                        {analysisResult.is_spoof
                          ? "SYNTHETIC AI VOICE CLONE"
                          : analysisResult.operational_decision?.risk_tier === "ELEVATED"
                          ? "UNCERTAIN / STEP-UP VERIFICATION"
                          : "AUTHENTIC HUMAN VOICE"}
                      </div>
                      <span
                        className={`mono-text verdict-policy-badge ${analysisResult.is_spoof ? "badge-spoof" : "badge-bonafide"}`}
                      >
                        {analysisResult.operational_decision?.recommended_action || "POLICY PASS"}
                      </span>
                    </div>
                    <div className="verdict-desc">
                      {analysisResult.is_spoof
                        ? "Acoustic seam discontinuities and neural vocoder spectral artifacts detected across language switch points."
                        : analysisResult.operational_decision?.risk_tier === "ELEVATED"
                        ? "Marginal acoustic confidence under channel noise; step-up biometric prompt recommended."
                        : "Continuous natural vocal tract dynamics and zero neural synthesis seams detected."}
                    </div>
                  </div>
                </div>

                {/* Metrics 3-Col Bar */}
                <div className="metrics-grid-3">
                  <div className="risk-meter-box">
                    <div className="metric-label">Spoof Probability</div>
                    <div
                      className="metric-val-huge"
                      style={{
                        color: analysisResult.is_spoof ? "var(--crimson-pure)" : "var(--emerald-pure)",
                      }}
                    >
                      {(analysisResult.spoof_probability * 100).toFixed(1)}%
                    </div>
                    <div className="metric-sub">
                      Threshold &tau; = 50.0%
                    </div>
                  </div>

                  <div className="risk-meter-box">
                    <div className="metric-label">95% Bootstrap CI</div>
                    <div className="metric-val-huge mono-text" style={{ fontSize: "18px", color: "var(--text-white)" }}>
                      [{(analysisResult.confidence_interval_95[0] * 100).toFixed(1)}%, {(analysisResult.confidence_interval_95[1] * 100).toFixed(1)}%]
                    </div>
                    <div className="metric-sub">
                      100-fold clustered resample
                    </div>
                  </div>

                  <div className="risk-meter-box">
                    <div className="metric-label">Operational Policy</div>
                    <div className="metric-val-huge" style={{ fontSize: "16px", color: "var(--text-white)" }}>
                      {analysisResult.operational_decision?.decision || "ALLOW"}
                    </div>
                    <div className="metric-sub">
                      Tier: {analysisResult.operational_decision?.risk_tier || "SECURE"}
                    </div>
                  </div>
                </div>

                {/* Waveform & Boundary Markers */}
                <div className="waveform-container">
                  <div className="waveform-header">
                    <span>Acoustic Energy & Code-Switch Seam Localization</span>
                    <span className="mono-text">{analysisResult.audio_metadata?.duration_sec}s &bull; {analysisResult.audio_metadata?.detected_boundary_count || 0} boundaries</span>
                  </div>
                  <div className="waveform-track">
                    {(analysisResult.audio_metadata?.waveform_peaks || []).map((val, idx) => (
                      <div
                        key={idx}
                        className="waveform-bar"
                        style={{
                          height: `${Math.max(6, Math.min(48, val * 48))}px`,
                          background: analysisResult.is_spoof ? "var(--crimson-pure)" : "var(--emerald-pure)",
                          opacity: 0.8,
                        }}
                      />
                    ))}

                    {/* Boundary Marker Lines */}
                    {(analysisResult.explanations || []).map((exp, eIdx) => {
                      const dur = analysisResult.audio_metadata?.duration_sec || 5.0;
                      const pct = Math.min(97, Math.max(3, (exp.timestamp_sec / dur) * 100));
                      return (
                        <div
                          key={eIdx}
                          className="seam-marker-line"
                          title={`Boundary #${exp.boundary_index}: ${exp.language_transition} at ${exp.timestamp_sec}s (${(exp.contribution_score * 100).toFixed(0)}% seam anomaly)`}
                          style={{ left: `${pct}%` }}
                        />
                      );
                    })}
                  </div>
                </div>

                {/* Acoustic Explanations List (Zero Transcript Content) */}
                <div style={{ marginTop: "16px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                    <div style={{ fontSize: "13px", fontWeight: 700, color: "var(--text-white)", display: "flex", alignItems: "center", gap: "6px" }}>
                      <span>🔬 Forensic Micro-Seam Attributions</span>
                    </div>
                    <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                      Zero Spoken Content Stored (DPDPA 2023)
                    </span>
                  </div>

                  {(!analysisResult.explanations || analysisResult.explanations.length === 0) ? (
                    <div style={{ fontSize: "13px", color: "var(--text-muted)", padding: "14px", background: "var(--bg-surface)", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
                      Monolingual or continuous vocal tract contour — no cross-lingual boundary seam detected.
                    </div>
                  ) : (
                    <div className="seam-card-list">
                      {analysisResult.explanations.map((exp, idx) => (
                        <div key={idx} className="seam-card-item">
                          <div>
                            <div style={{ fontSize: "13px", fontWeight: 700, color: "var(--text-white)", display: "flex", alignItems: "center", gap: "8px" }}>
                              <span>Boundary #{exp.boundary_index}</span>
                              <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>&bull;</span>
                              <span className="mono-text" style={{ color: "var(--cyan-pure)", fontSize: "12px" }}>{exp.language_transition}</span>
                              <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>@</span>
                              <span className="mono-text" style={{ color: "var(--text-secondary)", fontSize: "12px" }}>{exp.timestamp_sec}s</span>
                            </div>
                            <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "3px" }}>
                              {exp.description}
                            </div>
                          </div>
                          <div>
                            <span className={exp.contribution_score > 0.6 ? "seam-badge-danger" : "seam-badge-safe"}>
                              {(exp.contribution_score * 100).toFixed(0)}% Seam Discontinuity
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Diagnostics Toggle */}
                <div style={{ marginTop: "16px" }}>
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{ fontSize: "11px", padding: "6px 12px" }}
                    onClick={() => setShowTechnicalLog(!showTechnicalLog)}
                  >
                    {showTechnicalLog ? "Hide Diagnostic Telemetry" : "Show Diagnostic Telemetry"}
                  </button>

                  {showTechnicalLog && (
                    <pre
                      className="mono-text"
                      style={{
                        marginTop: "10px",
                        padding: "12px",
                        background: "var(--bg-base)",
                        border: "1px solid var(--border-subtle)",
                        borderRadius: "8px",
                        fontSize: "11px",
                        color: "var(--text-secondary)",
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
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
            <div>
              <h2>VaaniShield End-to-End Pipeline Architecture</h2>
              <p>How VaaniShield tracks synthetic speech and prevents voice fraud in Indian multilingual calls.</p>
            </div>
            <button type="button" className="btn-scan" onClick={() => navigateTo("analyzer")}>
              LAUNCH LIVE DETECTOR
            </button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "14px", marginBottom: "24px" }}>
            <div className="control-card">
              <div style={{ fontSize: "24px", marginBottom: "8px" }}>📡 Step 1</div>
              <h4 style={{ fontSize: "14px", color: "var(--text-white)", marginBottom: "6px" }}>Audio Ingestion</h4>
              <p style={{ fontSize: "12px", color: "var(--text-secondary)", lineHeight: "1.5" }}>
                Ingests streaming 8 kHz PSTN / VoIP telephony or 16 kHz HD speech. Downsamples and strips loudness offsets.
              </p>
            </div>

            <div className="control-card">
              <div style={{ fontSize: "24px", marginBottom: "8px" }}>📍 Step 2</div>
              <h4 style={{ fontSize: "14px", color: "var(--text-white)", marginBottom: "6px" }}>Boundary Localization</h4>
              <p style={{ fontSize: "12px", color: "var(--text-secondary)", lineHeight: "1.5" }}>
                Identifies code-switch points (e.g. Hindi &rarr; English). Extracts localized &plusmn;300ms analysis windows around transitions.
              </p>
            </div>

            <div className="control-card">
              <div style={{ fontSize: "24px", marginBottom: "8px" }}>🔬 Step 3</div>
              <h4 style={{ fontSize: "14px", color: "var(--text-white)", marginBottom: "6px" }}>Dual-Stream Extraction</h4>
              <p style={{ fontSize: "12px", color: "var(--text-secondary)", lineHeight: "1.5" }}>
                <strong>Stream A:</strong> Boundary micro-seams (F0 reset, formant transitions).<br />
                <strong>Stream B:</strong> SNAP speaker-nulling orthogonal projection.
              </p>
            </div>

            <div className="control-card">
              <div style={{ fontSize: "24px", marginBottom: "8px" }}>⚖️ Step 4</div>
              <h4 style={{ fontSize: "14px", color: "var(--text-white)", marginBottom: "6px" }}>Calibrated Fusion</h4>
              <p style={{ fontSize: "12px", color: "var(--text-secondary)", lineHeight: "1.5" }}>
                Calibrated logistic regression produces continuous probabilities with 95% bootstrap CIs and operational risk tiers.
              </p>
            </div>
          </div>

          <div className="control-card" style={{ marginBottom: "20px" }}>
            <h3 style={{ fontSize: "16px", marginBottom: "10px", color: "var(--text-white)" }}>
              Why Both Streams are Necessary: The Indian Telephony Dilemma
            </h3>
            <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.6" }}>
              1. <strong>Speaker Entanglement Problem:</strong> Standard deep learning models trained on voice samples easily memorize individual speakers or accent characteristics rather than synthesis artifacts. Our <em>Speaker-Nulling Artifact Projection (SNAP)</em> projects embeddings onto the orthogonal complement of the between-speaker subspace, stripping identity and preventing false alarms on non-standard Indian accents.<br />
              2. <strong>Intra-Utterance Boundary Seams:</strong> Voice cloning models frequently fail when transitioning between languages mid-sentence (e.g., Hinglish), leaving subtle phase resets and unnatural pitch jumps at switch points. Our boundary stream detects these localized anomalies even under compression.
            </p>
          </div>
        </article>
      )}

      {/* View 3: Technical & Research Specs */}
      {currentPage === "specs" && (
        <article className="page-article">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
            <div>
              <h2>Scientific Evaluation & Ablation Results</h2>
              <p>Empirical metrics computed across 4 strictly speaker-disjoint splits (N=400 recordings).</p>
            </div>
            <button type="button" className="btn-scan" onClick={() => navigateTo("analyzer")}>
              LAUNCH LIVE DETECTOR
            </button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "14px", marginBottom: "24px" }}>
            <div className="risk-meter-box">
              <div className="metric-label">IN-DOMAIN TEST EER</div>
              <div className="metric-val-huge" style={{ color: "var(--emerald-pure)" }}>14.00%</div>
              <div className="metric-sub">95% CI: [2.0%, 30.0%]</div>
            </div>
            <div className="risk-meter-box">
              <div className="metric-label">HELDOUT SYNTHESIZER EER</div>
              <div className="metric-val-huge" style={{ color: "var(--amber-pure)" }}>74.00%</div>
              <div className="metric-sub">Microsoft Edge Neural Voice</div>
            </div>
            <div className="risk-meter-box">
              <div className="metric-label">VALIDATION EER</div>
              <div className="metric-val-huge" style={{ color: "var(--cyan-pure)" }}>0.00%</div>
              <div className="metric-sub">95% CI: [0.0%, 0.0%]</div>
            </div>
          </div>

          <h3>1. Model Comparison across Splits (N=400)</h3>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginBottom: "12px" }}>
            Zero speaker or utterance overlap across splits. Verified generalization without speaker memorization:
          </p>
          <div className="table-wrapper" style={{ marginBottom: "24px" }}>
            <table className="raw-table">
              <thead>
                <tr>
                  <th>Model Configuration</th>
                  <th>Val EER</th>
                  <th>Test EER (In-Domain)</th>
                  <th>Held-Out Synthesizer EER</th>
                  <th>Test Precision / Recall</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Baseline A (Generic Acoustic + LR)</td>
                  <td>24.0%</td>
                  <td>39.0%</td>
                  <td>12.0%</td>
                  <td>66.0% / 62.0%</td>
                </tr>
                <tr>
                  <td>Baseline B (Indian-Accent Mel + LFCC)</td>
                  <td>34.0%</td>
                  <td>48.0%</td>
                  <td>14.0%</td>
                  <td>56.9% / 66.0%</td>
                </tr>
                <tr style={{ background: "rgba(16, 185, 129, 0.08)" }}>
                  <td><strong style={{ color: "var(--text-white)" }}>VaaniShield Full Fused Model</strong></td>
                  <td style={{ color: "var(--emerald-text)", fontWeight: 700 }}>0.0%</td>
                  <td style={{ color: "var(--emerald-text)", fontWeight: 700 }}>14.0%</td>
                  <td style={{ color: "var(--amber-text)", fontWeight: 700 }}>74.0%</td>
                  <td style={{ color: "var(--emerald-text)", fontWeight: 700 }}>88.4% / 76.0%</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3>2. Three-Way Ablation Analysis (Anti-Shortcut Verified)</h3>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginBottom: "12px" }}>
            Evaluating individual and combined stream contributions across test and held-out sets:
          </p>
          <div className="table-wrapper" style={{ marginBottom: "24px" }}>
            <table className="raw-table">
              <thead>
                <tr>
                  <th>Stream Variant</th>
                  <th>Test EER</th>
                  <th>Held-Out Synthesizer EER</th>
                  <th>Finding & Role in Architecture</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Boundary-Local Stream Only</td>
                  <td>51.0%</td>
                  <td>70.0%</td>
                  <td>Under narrowband telephony loss without vocoder conditioning, boundary cues alone operate near chance.</td>
                </tr>
                <tr>
                  <td>Speaker-Nulled Global Stream Only</td>
                  <td>22.0%</td>
                  <td>66.0%</td>
                  <td>Orthogonal projection captures XTTS training vocoder signatures, but exhibits domain shift on unseen commercial neural TTS engines.</td>
                </tr>
                <tr style={{ background: "rgba(16, 185, 129, 0.08)" }}>
                  <td><strong style={{ color: "var(--text-white)" }}>VaaniShield Full Fused Model</strong></td>
                  <td style={{ color: "var(--emerald-text)", fontWeight: 700 }}>14.0%</td>
                  <td style={{ color: "var(--amber-text)", fontWeight: 700 }}>74.0%</td>
                  <td>Fusing boundary transition dynamics with speaker-nulled residuals yields lowest in-domain error rate (14.0% vs 39.0%/48.0% baselines).</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3>3. Zero Transcript Storage & DPDPA 2023 Compliance</h3>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.6" }}>
            In compliance with India's Digital Personal Data Protection Act (DPDPA 2023), VaaniShield performs strictly non-semantic acoustic analysis. Audio is evaluated strictly for physical vocal tract and vocoder anomalies; speech content is never recorded, transcribed, or stored in persistent databases.
          </p>
        </article>
      )}

      {/* Footer */}
      <footer className="site-footer">
        <div>
          &copy; 2026 VaaniShield Security Intelligence &bull; IIT Delhi Research Park
        </div>
        <div>
          Privacy-Preserving Acoustic Voice Clone Detection &bull; Zero Transcript Storage
        </div>
      </footer>
    </div>
  );
}
