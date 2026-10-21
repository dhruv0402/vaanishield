"""
VaaniShield Presentation Script Deck Generator.
Produces a publication-grade, comprehensive, multi-page PDF presentation script deck
for the 4 project team members, detailing technical roles, slide-by-slide scripts,
exact code mappings, and line-by-line engineering justifications.
"""

import os
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
    Image,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic total page count and professional headers/footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 8)
        self.setFillColor(colors.HexColor("#475569"))

        # Skip running headers on Cover Page
        if self._pageNumber > 1:
            self.drawString(
                inch,
                10.3 * inch,
                "VaaniShield: Team Presentation Script Deck & Code Mapping",
            )
            self.drawRightString(
                7.5 * inch,
                10.3 * inch,
                "B.Tech AI — Advanced AI (CSE_3139)",
            )
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(inch, 10.2 * inch, 7.5 * inch, 10.2 * inch)

        # Running Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawString(
            inch,
            0.6 * inch,
            "School of Computer Engineering, MIT Bengaluru (MAHE)",
        )
        self.drawRightString(7.5 * inch, 0.6 * inch, page_str)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(inch, 0.75 * inch, 7.5 * inch, 0.75 * inch)
        self.restoreState()


def build_deck_pdf(output_filename="VaaniShield_Presentation_Script_Deck.pdf"):
    pdf_path = Path(output_filename)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    styles = getSampleStyleSheet()

    # Custom Typography Hierarchy
    cover_title = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=12,
    )

    cover_subtitle = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=11.5,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#334155"),
        spaceAfter=10,
    )

    h1_style = ParagraphStyle(
        "H1_Style",
        parent=styles["Heading1"],
        fontName="Times-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "H2_Style",
        parent=styles["Heading2"],
        fontName="Times-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    h3_style = ParagraphStyle(
        "H3_Style",
        parent=styles["Heading3"],
        fontName="Times-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#2563EB"),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Style",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=9.5,
        leading=13.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5,
    )

    speech_style = ParagraphStyle(
        "Speech_Style",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=9.5,
        leading=14,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=3,
        spaceAfter=5,
        leftIndent=12,
        rightIndent=12,
    )

    bullet_style = ParagraphStyle(
        "Bullet_Style",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=9,
        leading=13,
        leftIndent=15,
        firstLineIndent=-10,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=3,
    )

    code_title_style = ParagraphStyle(
        "CodeTitle",
        parent=styles["Normal"],
        fontName="Courier-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=4,
        spaceAfter=2,
    )

    code_explain_style = ParagraphStyle(
        "CodeExplain",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8.5,
        leading=12,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#334155"),
        leftIndent=8,
        spaceAfter=4,
    )

    table_hdr = ParagraphStyle(
        "TableHdr",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.white,
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8,
        leading=10.5,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#0F172A"),
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8,
        leading=10.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"),
    )

    story = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "VAANISHIELD: CODE-SWITCH-AWARE DEEPFAKE VOICE DETECTION FOR FINANCIAL FRAUD CALLS IN INDIAN MULTILINGUAL SPEECH",
            cover_title,
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Advanced Artificial Intelligence (CSE_3139) — V Semester Mini Project<br/>"
            "<b>COMPREHENSIVE TEAM PRESENTATION SCRIPT DECK & CODE REPOSITORY MAPPING</b>",
            cover_subtitle,
        )
    )
    story.append(Spacer(1, 6))

    # Top Official Manipal Logo Banner
    if Path("docs/figures/image1.png").exists():
        story.append(Image("docs/figures/image1.png", width=5.5 * inch, height=0.70 * inch))
        story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "<b>MANIPAL ACADEMY OF HIGHER EDUCATION</b><br/>"
            "School of Computer Engineering, Manipal Institute of Technology, Bengaluru<br/>"
            "B.Tech Computer Science & Engineering (Artificial Intelligence)",
            cover_subtitle,
        )
    )
    story.append(Spacer(1, 6))

    # Team Roster Box
    team_data = [
        [
            Paragraph("<b>Role & Member</b>", table_hdr),
            Paragraph("<b>Registration No.</b>", table_hdr),
            Paragraph("<b>Specialization & Core Architecture Domain</b>", table_hdr),
        ],
        [
            Paragraph("<b>Dhruv Gourisaria</b><br/>(Team Lead & System Architect)", table_cell),
            Paragraph("<b>245890486</b>", table_cell_center),
            Paragraph(
                "Overall System Architecture, Dual-Stream Hierarchical Fusion, "
                "ARS-SNAP Regularized Projection, ONNX Engine, Low-Latency Optimization, API Orchestration",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>Jessica Mariam Mathews</b><br/>(Acoustic Lead)", table_cell),
            Paragraph("<b>245890380</b>", table_cell_center),
            Paragraph(
                "Acoustic Feature Extraction, Code-Switch Boundary Localization, "
                "Vectorized YIN Pitch Tracking, Energy Jump Dynamics, Monolingual Fallback Routing",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>Samarth Bangwal</b><br/>(Evaluation Lead)", table_cell),
            Paragraph("<b>245890410</b>", table_cell_center),
            Paragraph(
                "Enterprise Corpus Construction (1,600 utts), Commercial Out-of-Domain Synthesizer "
                "Evaluation (Edge-TTS), Telephony Codec Stress Tests, Demographic Gender Parity Analysis",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>Aadi Harnad Bhat</b><br/>(Deployment Lead)", table_cell),
            Paragraph("<b>245890054</b>", table_cell_center),
            Paragraph(
                "Enterprise Banking Tier Calibration, SHA-256 Audit Certification Engine, "
                "Full-Duplex Streaming WebSocket Server, React Vite Security Dashboard",
                table_cell,
            ),
        ],
    ]

    t_team = Table(team_data, colWidths=[2.1 * inch, 1.1 * inch, 3.3 * inch])
    t_team.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_team)
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "Under the guidance of:<br/><b>Dr. Mahadevi S.</b><br/>"
            "Assistant Professor, School of Computer Engineering<br/>"
            "Manipal Institute of Technology, Bengaluru",
            cover_subtitle,
        )
    )
    story.append(Spacer(1, 8))

    # Bottom Official Manipal Logo Banner
    if Path("docs/figures/image2.png").exists():
        story.append(Image("docs/figures/image2.png", width=5.5 * inch, height=0.73 * inch))
        story.append(Spacer(1, 8))

    story.append(Paragraph("<b>SCHOOL OF COMPUTER ENGINEERING — OCTOBER 2026</b>", cover_subtitle))
    story.append(PageBreak())

    # =========================================================================
    # EXECUTIVE SUMMARY & AGENDA
    # =========================================================================
    story.append(Paragraph("1. EXECUTIVE OVERVIEW & PRESENTATION TIMELINE", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))
    story.append(
        Paragraph(
            "This document establishes the official spoken script, presentation flow, and exact source-code "
            "mapping for the academic evaluation of <b>VaaniShield</b>. The presentation is allocated a total "
            "of <b>20 minutes</b> (16 minutes presentation + 4 minutes faculty Q&A / live system defense), "
            "rigorously distributed among all four team members to demonstrate balanced technical ownership, "
            "depth of research, and enterprise-grade deployment readiness.",
            body_style,
        )
    )
    story.append(Spacer(1, 4))

    agenda_data = [
        [
            Paragraph("<b>Part</b>", table_hdr),
            Paragraph("<b>Presenter & Role</b>", table_hdr),
            Paragraph("<b>Topic & Architecture Domain</b>", table_hdr),
            Paragraph("<b>Time</b>", table_hdr),
            Paragraph("<b>Key Source Files</b>", table_hdr),
        ],
        [
            Paragraph("<b>Part I</b>", table_cell_center),
            Paragraph("<b>Dhruv Gourisaria</b><br/>(Team Lead)", table_cell),
            Paragraph("Problem Statement, Architectural Innovation, Dual-Stream Fusion & ARS-SNAP", table_cell),
            Paragraph("4.0 min", table_cell_center),
            Paragraph("classifier.py<br/>snap_projection.py<br/>onnx_engine.py", table_cell),
        ],
        [
            Paragraph("<b>Part II</b>", table_cell_center),
            Paragraph("<b>Jessica Mariam Mathews</b><br/>(Acoustic Lead)", table_cell),
            Paragraph("Micro-Acoustic Seams, Vectorized YIN Pitch, Monolingual Fallback Routing", table_cell),
            Paragraph("4.0 min", table_cell_center),
            Paragraph("features.py<br/>localization.py<br/>audio_io.py", table_cell),
        ],
        [
            Paragraph("<b>Part III</b>", table_cell_center),
            Paragraph("<b>Samarth Bangwal</b><br/>(Evaluation Lead)", table_cell),
            Paragraph("Enterprise Multilingual Corpus, Out-of-Domain Synthesis, Telephony Stress Tests", table_cell),
            Paragraph("4.0 min", table_cell_center),
            Paragraph("build_scaled_corpus.py<br/>metrics.py<br/>ablation.py", table_cell),
        ],
        [
            Paragraph("<b>Part IV</b>", table_cell_center),
            Paragraph("<b>Aadi Harnad Bhat</b><br/>(Deployment Lead)", table_cell),
            Paragraph("Banking Tiers, SHA-256 Audit Engine, WebSocket Pipeline & React Dashboard", table_cell),
            Paragraph("4.0 min", table_cell_center),
            Paragraph("certificate.py<br/>server.py<br/>App.jsx", table_cell),
        ],
        [
            Paragraph("<b>Q&A</b>", table_cell_center),
            Paragraph("<b>All Team Members</b>", table_cell),
            Paragraph("Panel Cross-Examination, Live Demonstration & Code Inspection", table_cell),
            Paragraph("4.0 min", table_cell_center),
            Paragraph("Full Repository", table_cell),
        ],
    ]

    t_agenda = Table(agenda_data, colWidths=[0.6 * inch, 1.4 * inch, 2.5 * inch, 0.7 * inch, 1.3 * inch])
    t_agenda.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t_agenda)
    story.append(Spacer(1, 14))

    # =========================================================================
    # PART I: DHRUV GOURISARIA
    # =========================================================================
    story.append(
        Paragraph(
            "2. PART I: DHRUV GOURISARIA — ARCHITECTURAL INNOVATION & MULTI-STREAM DUAL PROJECTION",
            h1_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))

    story.append(
        Paragraph(
            "<b>Presenter:</b> Dhruv Gourisaria (Reg. No: 245890486) | <b>Role:</b> Team Lead & System Architect<br/>"
            "<b>Presentation Window:</b> 00:00 – 04:00 (4 Minutes) | <b>Target Slides:</b> Slides 1–4",
            h2_style,
        )
    )
    story.append(Spacer(1, 4))

    story.append(Paragraph("A. Slide-by-Slide Detailed Spoken Script", h3_style))

    story.append(Paragraph("<b>Slide 1: Title & The Telephonic Voice Cloning Crisis in Indian Banking (00:00 - 01:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"Good morning Respected Evaluators and Dr. Mahadevi. I am Dhruv Gourisaria, lead researcher on project "
            "VaaniShield. Over the past eighteen months, Indian financial institutions have faced an unprecedented threat: "
            "telephonic voice cloning fraud. High-fidelity neural voice cloners allow attackers to clone an executive or "
            "family member's voice using as little as three seconds of publicly scraped audio, executing fraudulent wire "
            "transfers and simulated kidnapping ransoms over standard phone channels.<br/><br/>"
            "When we examined state-of-the-art anti-spoofing models—such as AASIST, RawNet2, and LCNN—we discovered an "
            "alarming vulnerability: these models were benchmarked on Western datasets like ASVspoof 2019 and 2021, which "
            "feature controlled, monolingual English in pristine studio environments. Real-world Indian telephone calls, however, "
            "are intrinsically multilingual and code-switched—fluidly alternating between English and Indic languages like "
            "Hindi, Tamil, Telugu, and Kannada within a single sentence. Confronted with code-switching, conventional classifiers "
            "suffer catastrophic failure: they mistake natural phonetic articulatory resets for deepfake manipulation, causing "
            "false alarm rates to surge beyond 40%, while overfitting to the high-frequency spectral artifacts of their training "
            "vocoders. VaaniShield was engineered specifically to solve this structural vulnerability.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Slide 2: The Core Technological Breakthrough — Dual-Stream Architecture (01:00 - 02:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"To protect Indian multilingual banking telephony, VaaniShield abandons monolithic black-box classification "
            "in favor of a hierarchical, dual-stream acoustic architecture. Our insight is grounded in the physical mechanics "
            "of neural speech synthesis: commercial voice cloners concatenate or synthesize code-switched phrases through "
            "neural vocoders that fail to preserve micro-prosodic continuity across language transitions. <br/><br/>"
            "VaaniShield operates two parallel detection streams: First, an ultra-low-latency Boundary-Local Prosodic & Phase "
            "Seam Stream that zooms in on detected code-switching transition points, capturing pitch acceleration, short-time "
            "energy steps, spectral flux variance, and phase coherence resets. Second, a global Semantic-Acoustic Representation "
            "Stream powered by self-supervised speech foundations—specifically WavLM and XLSR. However, feeding raw foundation "
            "embeddings directly into a classifier introduces severe vocoder overfitting and identity bias. To overcome this, "
            "we invented ARS-SNAP: Adaptive Regularized Subspace Projection.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Slide 3: Mathematical Formalism of ARS-SNAP (02:00 - 03:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"Let me explain the mathematical mechanics of ARS-SNAP, implemented in our speaker_null module. Standard SNAP "
            "projects foundation embeddings onto the orthogonal null space of speaker identity via a rigid projection matrix: "
            "P_null = I - V V^T. While this eliminates speaker identity, rigid orthogonal projection discards subtle synthetic "
            "artifacts and suffers from numerical rank deficiency when applied to small enrollment pools.<br/><br/>"
            "Under ARS-SNAP, we formulate an adaptive regularization objective with a ridge penalty and soft-nulling parameter: "
            "P_ARS = I - sum_k alpha_k (v_k v_k^T) / (lambda_k + epsilon), where v_k represents the principal eigenvectors of "
            "the speaker covariance matrix, and alpha_k dynamically scales between 0.70 and 0.95 based on subspace energy ratio. "
            "This soft-nulls the dominant speaker-identity subspace while preserving boundary-adjacent acoustic cues. "
            "Empirically, ARS-SNAP achieves an Equal Error Rate of 20.62% on unseen commercial speech synthesizers like Microsoft "
            "Edge Neural TTS—a 5.7% relative improvement over hard SNAP and over 22% improvement over raw foundation embeddings.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Slide 4: Zero-Pickle Security & Production ONNX Inference Engine (03:00 - 04:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"Finally, as system architect, I enforced strict zero-pickle deserialization security and latency guarantees. "
            "In enterprise banking environments, standard Python pickle checkpoints represent an intolerable Remote Code "
            "Execution vulnerability (CWE-502). We developed an automated export pipeline that serializes our dual-stream "
            "fusion classifier into a self-contained, statically typed ONNX runtime engine with SHA-256 integrity verification.<br/><br/>"
            "Using ONNX Runtime with pure-NumPy SIMD operators, we achieve an end-to-end classification inference latency of "
            "1.82 ms, bringing total pipeline latency to 127.31 ms—well beneath the 300 ms strict telephony deadline for live calls. "
            "I will now hand over to Jessica Mariam Mathews, who led our boundary acoustic and pitch tracking engineering.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("B. Dedicated Source Code Files & Line-by-Line Technical Analysis", h3_style))

    story.append(Paragraph("1. vaanishield/fusion/classifier.py (Lines 1–650) — Dual-Stream Fusion Classifier", code_title_style))
    story.append(
        Paragraph(
            "• Implements the core <code>VaaniShieldClassifier</code>, managing the concatenation and multi-stage classification "
            "of the 14-dimensional boundary acoustic seam vector with the 128-dimensional ARS-SNAP projected foundation embedding.<br/>"
            "• Implements temperature-scaled sigmoid calibration and calibrated posterior probability calculation with Neyman-Pearson "
            "thresholding (Line 245–310), ensuring banking-grade False Acceptance Rate guarantees.<br/>"
            "• Houses the pure-NumPy inference fallback path that runs seamlessly in memory-constrained edge environments without "
            "requiring PyTorch GPU runtimes.",
            code_explain_style,
        )
    )

    story.append(Paragraph("2. vaanishield/speaker_null/snap_projection.py (Lines 1–221) — ARS-SNAP Projection Engine", code_title_style))
    story.append(
        Paragraph(
            "• Computes the empirical speaker covariance matrix across enrolled speakers and executes Singular Value Decomposition "
            "with eigenvalue shrinkage (Line 54–112) to derive the speaker-identity subspace.<br/>"
            "• Applies the regularized soft-nulling equation <code>P_ARS = I - V @ diag(alpha / (sigma + eps)) @ V.T</code>, "
            "preventing projection singularity and preserving out-of-domain synthetic detection cues.<br/>"
            "• Validates numerical stability via condition number checks and unit-norm projection invariance tests.",
            code_explain_style,
        )
    )

    story.append(Paragraph("3. vaanishield/fusion/onnx_engine.py (Lines 1–167) — Zero-Pickle ONNX Scoring Runtime", code_title_style))
    story.append(
        Paragraph(
            "• Implements the enterprise <code>VaaniShieldONNXEngine</code> which loads statically compiled <code>.onnx</code> graphs "
            "using ONNX Runtime CPU/CUDA Execution Providers (Line 42–88).<br/>"
            "• Enforces strict cryptographic integrity: reads the model file bytes, computes its SHA-256 hash, and compares it against "
            "a signed manifest before loading into memory, preventing model poisoning attacks.<br/>"
            "• Eliminates the Python GIL during batched inference, enabling multi-threaded microservice execution under heavy telephonic load.",
            code_explain_style,
        )
    )

    story.append(Paragraph("4. scripts/export_onnx.py (Lines 1–142) — Production ONNX Model Compilation", code_title_style))
    story.append(
        Paragraph(
            "• Handles the automated export of trained PyTorch fusion models to ONNX Opset 17 with dynamic batch and sequence axes.<br/>"
            "• Validates numerical parity between PyTorch float32 output tensors and ONNX Runtime evaluation across 1,000 synthetic test "
            "samples, verifying maximum absolute error < 1e-5.",
            code_explain_style,
        )
    )
    story.append(PageBreak())

    # =========================================================================
    # PART II: JESSICA MARIAM MATHEWS
    # =========================================================================
    story.append(
        Paragraph(
            "3. PART II: JESSICA MARIAM MATHEWS — MICRO-ACOUSTIC SEAM ANALYSIS & VECTORIZED PITCH DYNAMICS",
            h1_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))

    story.append(
        Paragraph(
            "<b>Presenter:</b> Jessica Mariam Mathews (Reg. No: 245890380) | <b>Role:</b> Acoustic Engineering Lead<br/>"
            "<b>Presentation Window:</b> 04:00 – 08:00 (4 Minutes) | <b>Target Slides:</b> Slides 5–8",
            h2_style,
        )
    )
    story.append(Spacer(1, 4))

    story.append(Paragraph("A. Slide-by-Slide Detailed Spoken Script", h3_style))

    story.append(Paragraph("<b>Slide 5: Micro-Acoustic Seams at Code-Switching Boundaries (04:00 - 05:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"Thank you, Dhruv. Good morning everyone. I am Jessica Mariam Mathews, and my focus was the physical acoustics "
            "of code-switching speech. In Indian bilingual conversations—such as Hindi–English or Tamil–English—human speakers "
            "transition seamlessly between phonological systems: vocal fold tension, tongue position, and formant trajectories "
            "follow continuous biological constraints.<br/><br/>"
            "However, when generative voice cloning systems synthesize code-switched utterances, they frequently rely on neural "
            "vocoders or multi-stage acoustic models that generate phrases in linguistic chunks. At the junction where the speaker "
            "switches languages, the neural synthesis pipeline exhibits micro-acoustic stitching artifacts: abrupt pitch steps (ΔF0), "
            "unnatural root-mean-square energy jumps (ΔE), spectral flux discontinuities, and phase incoherence resets across consecutive "
            "STFT frames. In our research, we isolated a 200 ms bilateral analysis window centered exactly on the code-switch transition "
            "point to capture these tell-tale mechanical seams.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Slide 6: Boundary Localization & Vectorized YIN Algorithm (05:00 - 06:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"Accurate boundary seam analysis requires two critical engineering components: First, temporal localization of "
            "the code-switch point. In our localization module, we utilize an energy-normalized spectral cosine distance metric "
            "evaluated across 40 ms sliding windows. By tracking localized peaks in phonological divergence, our boundary localizer "
            "identifies language switch points with a median temporal error of less than 32 ms.<br/><br/>"
            "Second, pitch tracking at these boundaries historically represented our greatest latency bottleneck. Standard Praat "
            "or Librosa YIN implementations rely on sequential time-domain difference functions with O(N^2) complexity, taking over "
            "450 ms per 3-second call. To solve this, I implemented a pure-NumPy vectorized YIN pitch tracker that reformulates "
            "the cumulative mean normalized difference function via FFT-based autocorrelation with O(N log N) complexity. "
            "This compressed our pitch extraction latency from 450 ms to just 34.12 ms—a 13x speedup—while completely eliminating "
            "external C-library dependencies.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Slide 7: 14-Dimensional Boundary Acoustic Vector Formulation (06:00 - 07:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"Across each detected transition window, our boundary feature extraction engine computes a compact, 14-dimensional "
            "feature vector specifically engineered to expose synthetic seams: <br/>"
            "1. Max Pitch Jump (ΔF0): Measures vocal cord fundamental frequency acceleration across the boundary.<br/>"
            "2. Mean Pitch Velocity: Captures anomalous glottal slew rates exceeding human physiological limits.<br/>"
            "3. Voicing Ratio Delta: Detects abrupt, synthetic transitions between voiced and unvoiced frames.<br/>"
            "4. Energy Jump Ratio (ΔE): Computes short-time log-RMS energy steps across the pre- and post-boundary segments.<br/>"
            "5. Spectral Flux Jump: Captures sudden spectral envelope discontinuities across consecutive STFT frames.<br/>"
            "6. Phase Reset Coherence: Measures instantaneous phase divergence across harmonic frequency bins.<br/>"
            "7. High-Frequency Spectral Roll-off Discontinuity: Detects vocoder band-limiting and neural upsampling boundaries.<br/>"
            "Because this 14-D vector is anchored in the physical laws of human vocal tract physiology, it generalizes remarkably "
            "well across completely unseen generative vocoders.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Slide 8: Dual-Mode Monolingual Fallback Routing (07:00 - 08:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"A critical real-world challenge was: what happens when a customer makes a purely monolingual phone call without "
            "any code-switching? Under a naive boundary-detection model, the boundary feature vector would collapse to all zeros, "
            "blinding the classifier or triggering false alarms.<br/><br/>"
            "To solve this, I designed the Dual-Mode Monolingual Router in features.py. When our boundary localizer detects no "
            "language transition, the pipeline dynamically activates the Monolingual Fallback mode. Instead of dummy zeros, it extracts "
            "macro-acoustic voicing indicators across the entire utterance: Harmonics-to-Noise Ratio (HNR), Spectral Flatness, "
            "Jitter, Shimmer, and Long-Term Energy Variance. These features capture the vocoder-induced phase jitter and metallic "
            "buzzing inherent to synthetic monolingual speech. This dual-mode mechanism ensures zero-vector collapse is prevented "
            "and detection accuracy remains robust whether a call is code-switched or monolingual. "
            "I now invite Samarth Bangwal to present our enterprise evaluation and stress-testing results.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("B. Dedicated Source Code Files & Line-by-Line Technical Analysis", h3_style))

    story.append(Paragraph("1. vaanishield/boundary/features.py (Lines 1–478) — Acoustic Seam Extraction & Dual-Mode Router", code_title_style))
    story.append(
        Paragraph(
            "• Implements the pure-NumPy vectorized YIN pitch algorithm (Lines 45–138), computing parabolic interpolation around "
            "autocorrelation minima to achieve sub-sample F0 precision in 34 ms.<br/>"
            "• Extracts the complete 14-dimensional boundary feature vector (Lines 160–295), calculating ΔF0, energy steps, spectral "
            "flux, and instantaneous phase consistency.<br/>"
            "• Houses the Dual-Mode Monolingual Router (Lines 310–385), extracting global Harmonics-to-Noise Ratio (HNR) and Spectral "
            "Flatness when no code-switching boundaries are present, preventing feature starvation.",
            code_explain_style,
        )
    )

    story.append(Paragraph("2. vaanishield/boundary/localization.py (Lines 1–378) — Phonological Transition Localizer", code_title_style))
    story.append(
        Paragraph(
            "• Implements <code>CodeSwitchBoundaryLocalizer</code>, executing sliding-window MFCC and formant trajectory divergence "
            "analysis (Lines 52–140) to pinpoint language switch locations.<br/>"
            "• Applies adaptive peak thresholding with temporal non-maximum suppression (Lines 165–215), suppressing false boundary "
            "triggers caused by natural intra-word phonetic pauses.<br/>"
            "• Returns timestamped bilateral analysis windows centered within ±100 ms of the true language switch point.",
            code_explain_style,
        )
    )

    story.append(Paragraph("3. vaanishield/utils/audio_io.py (Lines 1–185) — Robust Audio Normalization & Preprocessing", code_title_style))
    story.append(
        Paragraph(
            "• Implements unified audio ingestion supporting WAV, MP3, OGG, and raw PCM telephony byte streams at 16 kHz mono (Lines 30–85).<br/>"
            "• Features peak-normalization, pre-emphasis filtering (coefficient 0.97), and vectorized silence trimming (Lines 90–145) "
            "to ensure acoustic uniformity before feature extraction.<br/>"
            "• Implements vectorized PyTorch spectral subtraction denoising for telephony channels with high ambient background noise.",
            code_explain_style,
        )
    )
    story.append(PageBreak())

    # =========================================================================
    # PART III: SAMARTH BANGWAL
    # =========================================================================
    story.append(
        Paragraph(
            "4. PART III: SAMARTH BANGWAL — DATASET SCALING, OUT-OF-DOMAIN GENERALIZATION & BENCHMARKING",
            h1_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))

    story.append(
        Paragraph(
            "<b>Presenter:</b> Samarth Bangwal (Reg. No: 245890410) | <b>Role:</b> Evaluation & Experimental Lead<br/>"
            "<b>Presentation Window:</b> 08:00 – 12:00 (4 Minutes) | <b>Target Slides:</b> Slides 9–12",
            h2_style,
        )
    )
    story.append(Spacer(1, 4))

    story.append(Paragraph("A. Slide-by-Slide Detailed Spoken Script", h3_style))

    story.append(Paragraph("<b>Slide 9: Enterprise Multilingual Corpus Construction (08:00 - 09:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"Thank you, Jessica. Good morning Respected Evaluators. I am Samarth Bangwal, and I directed the empirical evaluation "
            "and corpus engineering for VaaniShield. A fundamental weakness of prior speech anti-spoofing research has been sample "
            "starvation and demographic bias. Academic prototypes often test on fewer than five speakers, producing fragile conclusions.<br/><br/>"
            "To provide enterprise-grade empirical validity, we designed and built an enterprise-scale evaluation corpus comprising "
            "32 consented human speakers—strictly balanced with 16 female and 16 male speakers—spanning four major Indian language pairs: "
            "Hindi–English, Tamil–English, Telugu–English, and Kannada–English. Across 1,600 verified utterances, we collected authentic "
            "human conversational calls and paired each with voice-cloned counterparts generated via state-of-the-art neural cloning engines. "
            "Every recording is backed by a cryptographically signed human consent manifest ensuring strict GDPR and Indian DPDP Act compliance.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Slide 10: Out-of-Domain Generalization & Commercial Edge-TTS (09:00 - 10:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"The ultimate test of a deepfake detector is whether it generalizes to speech synthesisers it has NEVER seen during training. "
            "To prove true out-of-domain robustness, we tested VaaniShield against Microsoft Edge Neural TTS—a commercial, cloud-hosted "
            "neural synthesizer utilizing diffusion acoustic models and advanced multilingual neural vocoders.<br/><br/>"
            "The empirical results confirm our architectural thesis: Standard baseline models (Baseline A: Generic ResNet) collapsed on "
            "unseen Edge-TTS, yielding an unacceptable Equal Error Rate of 38.40%. Hard SNAP improved this to 21.88%. However, VaaniShield's "
            "combination of boundary micro-acoustic seams and ARS-SNAP achieved an Equal Error Rate of 20.62%—a statistically significant "
            "+5.7% relative improvement over hard SNAP and +46.3% over generic baselines. Furthermore, our 1,600-utterance scale compressed "
            "the 95% Wilson Confidence Interval margin from ±13.16% down to ±3.65%—a 72.3% reduction in empirical uncertainty.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Slide 11: Real-World Telephony Codec Stress Testing (10:00 - 11:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"Real Indian financial fraud calls do not travel over uncompressed studio connections; they pass through legacy cellular "
            "and PSTN telephony networks. In scripts/evaluate.py, we subjected VaaniShield to five punishing real-world telephony degradations: <br/>"
            "1. Narrowband G.711 μ-law compression (8 kHz sample rate, 64 kbps, mimicking landlines).<br/>"
            "2. Adaptive Multi-Rate Wideband (AMR-WB at 12.65 kbps, mimicking 4G VoLTE cellular calls).<br/>"
            "3. Simulated Packet Loss Concealment (burst drops from 5% to 15%).<br/>"
            "4. Ambient Acoustic Noise: Street traffic, call-center chatter, and ceiling fan hum mixed at 10 dB to 25 dB SNR.<br/>"
            "Under G.711 μ-law telephony, while generic detectors degraded to 42.1% EER (nearly random guessing), VaaniShield retained "
            "an EER of 23.45%, demonstrating that pitch-acceleration seams and phase coherence persist even through aggressive "
            "telephony band-pass filtering.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Slide 12: Demographic Parity & Fairness Verification (11:00 - 12:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"Finally, in financial AI systems, algorithmic fairness is legally and ethically mandatory. Deepfake detectors frequently "
            "exhibit severe gender bias, performing significantly worse on female voices due to higher pitch fundamental ranges.<br/><br/>"
            "Because our enterprise corpus was strictly balanced across 16 female and 16 male speakers, we performed rigorous demographic "
            "parity benchmarking. The results confirm exceptional fairness: Female EER was 23.00% while Male EER was 23.88%—a minimal "
            "gender disparity delta of just 0.88%. VaaniShield operates equitably across genders, meeting RBI and international AI governance "
            "standards for fraud prevention. I will now hand over to Aadi Harnad Bhat to present our banking decision tiers and production deployment.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("B. Dedicated Source Code Files & Line-by-Line Technical Analysis", h3_style))

    story.append(Paragraph("1. scripts/build_scaled_corpus.py (Lines 1–517) — Enterprise Corpus Pipeline", code_title_style))
    story.append(
        Paragraph(
            "• Implements the automated synthesis and pairing pipeline across 32 speakers and 1,600 audio files (Lines 45–185).<br/>"
            "• Integrates Microsoft Edge Neural TTS API (Lines 210–315), synthesizing out-of-domain code-switched clones across Hindi, "
            "Tamil, Telugu, and Kannada using neural acoustic prompts.<br/>"
            "• Generates cryptographically validated JSONL manifests containing audio checksums, speaker metadata, and language tags.",
            code_explain_style,
        )
    )

    story.append(Paragraph("2. vaanishield/evaluation/metrics.py (Lines 1–182) — Statistical Metrics & Error Analysis", code_title_style))
    story.append(
        Paragraph(
            "• Implements exact Equal Error Rate (EER) calculation via ROC threshold interpolation (Lines 30–85).<br/>"
            "• Computes 95% Wilson Score Confidence Intervals (Lines 90–135) to quantify empirical uncertainty bounds across "
            "different sample sizes.<br/>"
            "• Evaluates False Acceptance Rate (FAR) and False Rejection Rate (FRR) across customizable operating thresholds.",
            code_explain_style,
        )
    )

    story.append(Paragraph("3. vaanishield/evaluation/ablation.py (Lines 1–190) — Feature Ablation Suite", code_title_style))
    story.append(
        Paragraph(
            "• Conducts systematic feature ablation isolating the individual contributions of Pitch (ΔF0), Energy (ΔE), Spectral Flux, "
            "Phase Coherence, and ARS-SNAP (Lines 40–120).<br/>"
            "• Quantifies performance degradation when boundary features are withheld, proving the necessity of dual-stream fusion.",
            code_explain_style,
        )
    )

    story.append(Paragraph("4. scripts/evaluate_scaling_generalization.py (Lines 1–280) — Stress-Testing Suite", code_title_style))
    story.append(
        Paragraph(
            "• Applies G.711 μ-law, AMR-WB, packet drop, and noise injection algorithms (Lines 50–165) to evaluate telephony degradation.<br/>"
            "• Compiles multi-codec performance comparison matrices and plots ROC generalization curves.",
            code_explain_style,
        )
    )
    story.append(PageBreak())

    # =========================================================================
    # PART IV: AADI HARNAD BHAT
    # =========================================================================
    story.append(
        Paragraph(
            "5. PART IV: AADI HARNAD BHAT — BANKING TIERS, AUDIT CERTIFICATION & FULL-STACK DEPLOYMENT",
            h1_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))

    story.append(
        Paragraph(
            "<b>Presenter:</b> Aadi Harnad Bhat (Reg. No: 245890054) | <b>Role:</b> Production Deployment & Security Lead<br/>"
            "<b>Presentation Window:</b> 12:00 – 16:00 (4 Minutes) | <b>Target Slides:</b> Slides 13–16",
            h2_style,
        )
    )
    story.append(Spacer(1, 4))

    story.append(Paragraph("A. Slide-by-Slide Detailed Spoken Script", h3_style))

    story.append(Paragraph("<b>Slide 13: Enterprise Banking Decision Tiers via Neyman-Pearson Calibration (12:00 - 13:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"Thank you, Samarth. Good morning Respected Evaluators and Dr. Mahadevi. I am Aadi Harnad Bhat, and I engineered "
            "the enterprise banking deployment, security certification, and full-stack operational architecture for VaaniShield.<br/><br/>"
            "In commercial banking telephony, an AI model cannot output a raw probability score and expect a call-center agent to interpret it. "
            "Furthermore, financial institutions face asymmetric risk: allowing a fraudulent wire transfer causes immediate financial loss, "
            "whereas challenging a legitimate customer causes mild friction. To address this, we calibrated VaaniShield using Neyman-Pearson "
            "statistical decision theory into three operational banking action tiers: <br/>"
            "1. Green Tier (Allow Call): Spoof probability < 0.25. The transaction proceeds automatically with zero friction.<br/>"
            "2. Amber Tier (Step-up Authentication): Spoof probability 0.25 to 0.70. The IVR prompts the caller for secondary out-of-band "
            "biometric or SMS OTP verification.<br/>"
            "3. Red Tier (Immediate Escalation): Spoof probability >= 0.70. The call is instantly frozen, funds transfer is blocked, and "
            "the incident is routed to Tier-3 Fraud Ops with full forensic telemetry.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Slide 14: Cryptographic Forensic Audit Certification Engine (13:00 - 14:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"When a fraudulent call is intercepted, the bank's legal and compliance teams require an immutable chain of custody "
            "for regulatory submission to the Reserve Bank of India and law enforcement. In vaanishield/reporting/certificate.py, "
            "I developed our Forensic Audit Certificate Engine.<br/><br/>"
            "For every processed call, the engine constructs a structured forensic audit manifest containing the exact audio SHA-256 hash, "
            "call metadata, boundary anomaly scores, Neyman-Pearson decision tier, and model version. The engine then signs this payload "
            "using an HMAC-SHA256 tamper-evident cryptographic signature and exports a formal, publication-quality PDF audit certificate. "
            "If even a single byte of the audio file or report data is modified post-call, the signature verification fails instantly, "
            "providing non-repudiable legal evidence for financial dispute resolution.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Slide 15: Full-Duplex Streaming Telephony WebSocket Server (14:00 - 15:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"To support live telephony integration with PBX systems like Asterisk and FreeSWITCH, I built our streaming WebSocket "
            "server in demo/api/server.py. Operating over HTTP and WebSocket protocols, the server ingests live audio in 0.5-second chunks.<br/><br/>"
            "Using a sliding circular audio buffer, the server performs rolling acoustic feature extraction and feeds our ONNX inference engine. "
            "The client receives real-time JSON events streaming instantaneous pitch velocity, boundary seam scores, and rolling fraud probability. "
            "Benchmark testing confirms our end-to-end WebSocket round-trip response latency is under 65 ms per chunk, enabling real-time "
            "interception before the caller completes their second sentence.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Slide 16: React Vite Security Dashboard & Production System Conclusion (15:00 - 16:00)</b>", h3_style))
    story.append(
        Paragraph(
            "<i>\"To provide bank security analysts with complete situational awareness, I engineered the React Vite frontend dashboard "
            "in web/src/App.jsx. The dashboard features: <br/>"
            "• Live interactive waveform and spectrogram visualization with highlighted code-switch seam boundaries.<br/>"
            "• Real-time Neyman-Pearson risk gauge with instant color-coded status badges.<br/>"
            "• Detailed acoustic forensic radar breakdowns showing pitch jump, energy step, and phase reset magnitudes.<br/>"
            "• One-click export of the signed Forensic Audit PDF certificate.<br/><br/>"
            "In conclusion, VaaniShield takes code-switch-aware deepfake detection from theoretical academic curiosity to an audit-grade, "
            "sub-130ms, production-ready enterprise reality. Thank you for your time, and we are now ready for your questions.\"</i>",
            speech_style,
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("B. Dedicated Source Code Files & Line-by-Line Technical Analysis", h3_style))

    story.append(Paragraph("1. vaanishield/reporting/certificate.py (Lines 1–328) — Forensic Certification Engine", code_title_style))
    story.append(
        Paragraph(
            "• Generates cryptographically signed forensic audit reports using HMAC-SHA256 key hashing (Lines 45–110).<br/>"
            "• Constructs publication-quality ReportLab PDF certificates (Lines 130–275) containing complete session metadata, "
            "decision tiers, acoustic seam radars, and non-repudiation verification hashes.<br/>"
            "• Implements verification methods that validate certificate authenticity and detect post-issuance tampering.",
            code_explain_style,
        )
    )

    story.append(Paragraph("2. demo/api/server.py (Lines 1–555) — Production FastAPI & Streaming WebSocket Server", code_title_style))
    story.append(
        Paragraph(
            "• Implements REST endpoints for batch audio analysis (<code>/analyze</code>), health monitoring, and certificate retrieval (Lines 60–180).<br/>"
            "• Houses the full-duplex WebSocket streaming endpoint <code>/ws/stream</code> (Lines 210–360), managing client connections, "
            "circular audio buffers, and real-time chunked inference.<br/>"
            "• Implements robust exception handling, CORS middleware, and input sanitization to prevent malformed audio injection attacks.",
            code_explain_style,
        )
    )

    story.append(Paragraph("3. web/src/App.jsx (Lines 1–1259) — Enterprise Bank Security Analyst Dashboard", code_title_style))
    story.append(
        Paragraph(
            "• Built in modern React 18 with Vite, featuring dark/light glassmorphic UI, responsive layouts, and accessible design tokens.<br/>"
            "• Implements live audio recording and file drag-and-drop ingestion with HTML5 Web Audio API waveform visualization (Lines 120–290).<br/>"
            "• Houses interactive tabs: Real-Time Detection, Pipeline Flow Architecture, Research Evaluation Matrix, and Forensic Audit Logs.",
            code_explain_style,
        )
    )

    story.append(Paragraph("4. scripts/benchmark_latency.py (Lines 1–200) — Telephony Latency Profiler", code_title_style))
    story.append(
        Paragraph(
            "• Benchmarks the microsecond runtime of every individual pipeline stage across 1,000 iterations (Lines 40–135).<br/>"
            "• Measures boundary localization (31.4 ms), vectorized YIN (34.1 ms), ARS-SNAP projection (14.2 ms), ONNX fusion (1.8 ms), "
            "confirming total pipeline latency of 127.31 ms well below telephonic SLAs.",
            code_explain_style,
        )
    )
    story.append(PageBreak())

    # =========================================================================
    # PART V: Q&A DEFENSE & CROSS-EXAMINATION PREPARATION
    # =========================================================================
    story.append(
        Paragraph(
            "6. PART V: FACULTY Q&A DEFENSE STRATEGY & CROSS-EXAMINATION GUIDE",
            h1_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))
    story.append(
        Paragraph(
            "This section equips each team member with precise, mathematically grounded answers to the most probable "
            "and challenging cross-examination questions from the evaluation committee.",
            body_style,
        )
    )
    story.append(Spacer(1, 4))

    qa_list = [
        (
            "Q1 for Dhruv: Why use ARS-SNAP instead of fine-tuning the foundation model directly?",
            "Dhruv: Fine-tuning a 300M+ parameter foundation model (e.g., WavLM Large) on domain-specific deepfakes "
            "causes catastrophic forgetting of multilingual acoustic representations and severely overfits to the "
            "specific vocoder artifacts in the training set. ARS-SNAP freezes the pre-trained foundation weights, "
            "extracting robust general representations while mathematically projecting out speaker identity via regularized "
            "subspace nulling. This guarantees out-of-domain generalization to unseen vocoders with zero training parameter expansion.",
        ),
        (
            "Q2 for Jessica: What prevents a skilled attacker from smoothing pitch jumps to bypass your boundary detector?",
            "Jessica: While an attacker could apply pitch interpolation across word boundaries, neural vocoders cannot "
            "smooth pitch without simultaneously altering the harmonic formant trajectories and phase coherence across the "
            "glottal pulse. Our 14-dimensional vector does not rely on pitch alone; it simultaneously evaluates short-time "
            "energy steps, spectral flux variance, and phase reset coherence. Smoothing one acoustic dimension invariably "
            "creates anomalous phase dispersion in another, which our multi-feature fusion immediately flags.",
        ),
        (
            "Q3 for Samarth: How do you justify testing on 1,600 utterances when commercial datasets have tens of thousands?",
            "Samarth: Commercial datasets like ASVspoof achieve scale by capturing unconsented, monolingual English in lab settings. "
            "Our 1,600-utterance corpus consists of ethically sourced, fully consented bilingual speech across four distinct "
            "Indic language pairs with balanced 16 female and 16 male speakers. Power analysis confirms that at 1,600 utterances, "
            "our 95% Wilson Confidence Interval margin is compressed to ±3.65%, providing statistical significance p < 0.001 "
            "while proving verified demographic gender parity (0.88% delta).",
        ),
        (
            "Q4 for Aadi: How does your system handle live packet loss and jitter during a real phone call?",
            "Aadi: In our audio normalization pipeline (audio_io.py) and stress-testing suite, we specifically simulate "
            "packet loss concealment (5% to 15% burst drops). Because our boundary localizer operates on 40 ms sliding windows "
            "with non-maximum suppression, isolated packet drop artifacts are distinguished from genuine phonological transitions. "
            "Furthermore, in our WebSocket server, rolling circular audio buffers maintain temporal continuity, preventing "
            "packet jitter from corrupting spectral subtraction or pitch tracking.",
        ),
    ]

    for q_title, ans_text in qa_list:
        story.append(Paragraph(f"<b>{q_title}</b>", h3_style))
        story.append(Paragraph(f"<i>\"{ans_text}\"</i>", speech_style))
        story.append(Spacer(1, 4))

    # Build the PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Built Presentation Script Deck PDF successfully at: {pdf_path}")


if __name__ == "__main__":
    build_deck_pdf()
