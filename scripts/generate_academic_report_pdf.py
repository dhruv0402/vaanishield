#!/usr/bin/env python3
"""
Publication-Quality Academic PDF Report Generator for VaaniShield.
Uses ReportLab to generate VaaniShield_Academic_Project_Report.pdf and Report_Template (1).pdf
with formal university styling, embedded figures, styled tables, and running headers/footers.
"""
from pathlib import Path
import os
import shutil

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

# Register Times New Roman TrueType fonts for native Unicode mathematical symbol rendering
FONT_REGULAR = "Times-Roman"
FONT_BOLD = "Times-Bold"
FONT_ITALIC = "Times-Italic"
FONT_BOLD_ITALIC = "Times-BoldItalic"

ttf_paths = [
    ("/System/Library/Fonts/Supplemental/Times New Roman.ttf", "TNR"),
    ("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf", "TNR-Bold"),
    ("/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf", "TNR-Italic"),
    ("/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf", "TNR-BoldItalic"),
]

fonts_registered = True
for path_str, font_alias in ttf_paths:
    if os.path.exists(path_str):
        try:
            pdfmetrics.registerFont(TTFont(font_alias, path_str))
        except Exception:
            fonts_registered = False
    else:
        fonts_registered = False

if fonts_registered:
    FONT_REGULAR = "TNR"
    FONT_BOLD = "TNR-Bold"
    FONT_ITALIC = "TNR-Italic"
    FONT_BOLD_ITALIC = "TNR-BoldItalic"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # Suppress headers/footers on title page
            return

        self.saveState()
        self.setFont(FONT_REGULAR, 9)
        self.setFillColor(colors.HexColor("#475569"))

        # Running Header
        self.drawString(inch, 10.3 * inch, "VaaniShield: Code-Switch-Aware Deepfake Voice Detection")
        self.drawRightString(7.5 * inch, 10.3 * inch, "CSE_3139 Mini Project Report")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(inch, 10.2 * inch, 7.5 * inch, 10.2 * inch)

        # Running Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawString(inch, 0.6 * inch, "School of Computer Engineering, MIT Bengaluru (MAHE)")
        self.drawRightString(7.5 * inch, 0.6 * inch, page_str)
        self.line(inch, 0.75 * inch, 7.5 * inch, 0.75 * inch)
        self.restoreState()


def build_pdf():
    pdf_path = Path("Report_Template (1).pdf")
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles using TrueType Times New Roman
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=15,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName=FONT_REGULAR,
        fontSize=12,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#334155"),
        spaceAfter=12,
    )

    institution_style = ParagraphStyle(
        "CoverInstitution",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=13,
        leading=17,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=20,
    )

    author_style = ParagraphStyle(
        "CoverAuthor",
        parent=styles["Normal"],
        fontName=FONT_REGULAR,
        fontSize=11.5,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=16,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName=FONT_BOLD,
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName=FONT_BOLD,
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["BodyText"],
        fontName=FONT_REGULAR,
        fontSize=10.5,
        leading=14.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName=FONT_REGULAR,
        fontSize=10,
        leading=14,
        leftIndent=15,
        firstLineIndent=-10,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=4,
    )

    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Italic"],
        fontName=FONT_ITALIC,
        fontSize=9.5,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#475569"),
        spaceBefore=4,
        spaceAfter=10,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName=FONT_REGULAR,
        fontSize=8.5,
        leading=11.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"),
    )

    table_cell_left = ParagraphStyle(
        "TableCellLeft",
        parent=styles["Normal"],
        fontName=FONT_REGULAR,
        fontSize=8.5,
        leading=11.5,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#0F172A"),
    )

    table_hdr_style = ParagraphStyle(
        "TableHdr",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.white,
    )

    eq_style = ParagraphStyle(
        "Equation_Centered",
        parent=styles["Normal"],
        fontName=FONT_REGULAR,
        fontSize=11,
        leading=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"),
    )

    eq_num_style = ParagraphStyle(
        "Equation_Number",
        parent=styles["Normal"],
        fontName=FONT_REGULAR,
        fontSize=10.5,
        leading=15,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#0F172A"),
    )

    def format_display_equation(eq_text, eq_number):
        """Creates an academic publication-standard display equation row with right-aligned numbering."""
        tbl = Table(
            [[Paragraph(eq_text, eq_style), Paragraph(eq_number, eq_num_style)]],
            colWidths=[5.6 * inch, 0.9 * inch],
        )
        tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (0, 0), "CENTER"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        return tbl

    story = []

    # -------------------------------------------------------------
    # 1. TITLE / COVER PAGE
    # -------------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("VAANISHIELD: CODE-SWITCH-AWARE DEEPFAKE VOICE DETECTION FOR FINANCIAL FRAUD CALLS IN INDIAN MULTILINGUAL SPEECH", title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Advanced Artificial Intelligence (CSE_3139)<br/><b>V SEMESTER MINI PROJECT REPORT</b><br/><br/>Submitted to", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("MANIPAL ACADEMY OF HIGHER EDUCATION<br/>School of Computer Engineering<br/>Manipal Institute of Technology, Bengaluru<br/>B.Tech Computer Science & Engineering (Artificial Intelligence)", institution_style))
    story.append(Spacer(1, 6))
    
    # Top Manipal Institute Banner / Logo
    if Path("docs/figures/image1.png").exists():
        story.append(Image("docs/figures/image1.png", width=5.5*inch, height=0.70*inch))
        story.append(Spacer(1, 10))
    elif Path("docs/figures/manipal_logo.png").exists():
        story.append(Image("docs/figures/manipal_logo.png", width=5.5*inch, height=0.70*inch))
        story.append(Spacer(1, 10))

    story.append(Paragraph(
        "Submitted by:<br/><br/>"
        "<b>Dhruv Gourisaria - 245890486</b><br/>"
        "<b>Jessica Mariam Mathews - 245890380</b><br/>"
        "<b>Samarth Bangwal - 245890410</b><br/>"
        "<b>Aadi Harnad Bhat – 245890054</b>",
        author_style
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Under the guidance of:<br/><b>Dr. Mahadevi S.</b><br/>Assistant Professor, School of Computer Engineering<br/>Manipal Institute of Technology, Bengaluru", subtitle_style))
    story.append(Spacer(1, 10))

    # Second Manipal Institute Logo / Banner as per template
    if Path("docs/figures/image2.png").exists():
        story.append(Image("docs/figures/image2.png", width=5.5*inch, height=0.73*inch))
        story.append(Spacer(1, 10))

    story.append(Paragraph("<b>SCHOOL OF COMPUTER ENGINEERING</b><br/><b>October 2026</b>", subtitle_style))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # 2. ABSTRACT
    # -------------------------------------------------------------
    story.append(Paragraph("ABSTRACT", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "Telephonic voice cloning fraud has emerged as an existential security vulnerability across the Indian banking "
        "and financial services ecosystem. Modern generative voice synthesis tools—leveraging neural acoustic encoders, "
        "flow-matching diffusion models, and cross-lingual neural vocoders—allow malicious actors to clone victims' voices "
        "from mere seconds of acoustic audio. However, while Western anti-spoofing benchmarks (e.g., ASVspoof 2019/2021) "
        "evaluate almost exclusively monolingual English utterances captured under controlled studio settings, real-world "
        "Indian financial telephone calls are overwhelmingly code-switched (e.g., Hindi–English, Tamil–English, Kannada–English, "
        "Telugu–English). Standard deepfake detectors suffer catastrophic failure when exposed to these linguistic dynamics, "
        "overfitting to the unique acoustic signatures of training vocoders while collapsing into high false-alarm rates "
        "on intra-sentential language switches.",
        body_style
    ))
    story.append(Paragraph(
        "To resolve this fundamental gap, this project introduces <b>VaaniShield</b>: an exploratory, code-switch-aware "
        "hierarchical multi-stream deepfake detection framework investigated for financial telephony channels. "
        "VaaniShield integrates two complementary acoustic streams: (1) a lightweight <b>Boundary-Local "
        "Prosodic and Phase Seam Stream</b> targeting micro-acoustic transition artifacts (pitch perturbation Δ<i>F</i><sub>0</sub>, energy discontinuity Δ<i>E</i>, "
        "spectral flux resets, and phase desynchronization) around candidate code-switch points, paired with a <b>Dual-Mode "
        "Monolingual Router</b> (extracting Harmonics-to-Noise Ratio, Spectral Flatness, and macro-energy variance) to prevent uninformative zero-vector "
        "collapse during pure monolingual speech; and (2) an <b>Adaptive Regularized Subspace Projection (ARS-SNAP)</b> stream designed to soften "
        "speaker biometric entanglement in self-supervised representations (WavLM/XLSR) and avoid matrix near-singularity under out-of-domain commercial synthesizers.",
        body_style
    ))
    story.append(Paragraph(
        "The prototype pipeline is implemented with a pure-NumPy vectorized YIN pitch tracker (<i>O</i>(<i>N</i> log <i>N</i>) FFT autocorrelation) "
        "to eliminate Python GIL locking, PyTorch vectorized spectral subtraction, and a standalone ONNX inference graph. "
        "Evaluated on a curated 1,600-utterance Indic benchmark across 32 consented speakers (16 Female / 16 Male) covering four language pairs "
        "under channel degradations (G.711 μ-law, AMR-WB, packet loss, ambient noise) and unseen commercial speech synthesis (Microsoft Edge Neural TTS), "
        "ARS-SNAP achieves an Equal Error Rate (EER) of <b>20.62%</b> on out-of-domain commercial speech synthesis (compared to <b>21.87%</b> for hard SNAP, "
        "exhibiting a promising directional trend within the ±3.65% Wilson margin of error) while maintaining comparable overall corpus EER (<b>23.12%</b>). "
        "The system demonstrates demographic equity across genders (Female EER: <b>23.00%</b> vs. Male EER: <b>23.88%</b>, Δ = 0.88%) and a median "
        "pipeline latency of <b>127.31 ms</b>. Component ablations and threats to validity are systematically examined, offering a grounded baseline "
        "for acoustic-seam analysis in low-resource multilingual fraud detection.",
        body_style
    ))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # 3. TABLE OF CONTENTS
    # -------------------------------------------------------------
    story.append(Paragraph("TABLE OF CONTENTS", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))
    
    toc_data = [
        [Paragraph("<b>Section</b>", table_hdr_style), Paragraph("<b>Subsection Title</b>", table_hdr_style), Paragraph("<b>Page</b>", table_hdr_style)],
        [Paragraph("Abstract", table_cell_left), Paragraph("", table_cell_left), Paragraph("ii", table_cell_style)],
        [Paragraph("List of Tables", table_cell_left), Paragraph("", table_cell_left), Paragraph("iv", table_cell_style)],
        [Paragraph("List of Figures", table_cell_left), Paragraph("", table_cell_left), Paragraph("v", table_cell_style)],
        [Paragraph("<b>Chapter 1</b>", table_cell_left), Paragraph("<b>INTRODUCTION</b>", table_cell_left), Paragraph("<b>1</b>", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("1.1 Overview of the Research / Project Work", table_cell_left), Paragraph("1", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("1.2 Motivation and Background of the Study", table_cell_left), Paragraph("2", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("1.3 Problem Statement and Relevance", table_cell_left), Paragraph("3", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("1.4 Project Objectives", table_cell_left), Paragraph("4", table_cell_style)],
        [Paragraph("<b>Chapter 2</b>", table_cell_left), Paragraph("<b>BACKGROUND THEORY / LITERATURE REVIEW</b>", table_cell_left), Paragraph("<b>5</b>", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("2.1 Summary of Existing Studies and Comparative Analysis", table_cell_left), Paragraph("5", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("2.2 Evolution of Voice Cloning and Anti-Spoofing Technologies", table_cell_left), Paragraph("7", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("2.3 Key Findings, Trends, and Identified Research Gaps", table_cell_left), Paragraph("8", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("2.4 Chapter Conclusion", table_cell_left), Paragraph("9", table_cell_style)],
        [Paragraph("<b>Chapter 3</b>", table_cell_left), Paragraph("<b>METHODOLOGY</b>", table_cell_left), Paragraph("<b>10</b>", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("3.1 System Architecture and Mathematical Formulations", table_cell_left), Paragraph("10", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("3.2 Hardware and Software System Requirements", table_cell_left), Paragraph("14", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("3.3 Dataset Curation, Demographic Parity, and Licensing", table_cell_left), Paragraph("16", table_cell_style)],
        [Paragraph("<b>Chapter 4</b>", table_cell_left), Paragraph("<b>IMPLEMENTATION DETAILS & RESULT ANALYSIS</b>", table_cell_left), Paragraph("<b>18</b>", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("4.1 Implementation Pipeline and Security Architecture", table_cell_left), Paragraph("18", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("4.2 Empirical Results and Discussion Across Benchmark Regimes", table_cell_left), Paragraph("20", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("4.3 Component Ablation Study on Indic Benchmark", table_cell_left), Paragraph("23", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("4.4 Latency Profile and Concurrency Breakdown", table_cell_left), Paragraph("24", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("4.5 Limitations and Threats to Validity", table_cell_left), Paragraph("25", table_cell_style)],
        [Paragraph("<b>Chapter 5</b>", table_cell_left), Paragraph("<b>CONCLUSION AND FUTURE SCOPE</b>", table_cell_left), Paragraph("<b>26</b>", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("5.1 Conclusion", table_cell_left), Paragraph("26", table_cell_style)],
        [Paragraph("", table_cell_left), Paragraph("5.2 Suggestions for Future Research Directions", table_cell_left), Paragraph("27", table_cell_style)],
        [Paragraph("<b>REFERENCES</b>", table_cell_left), Paragraph("", table_cell_left), Paragraph("<b>28</b>", table_cell_style)],
    ]
    t_toc = Table(toc_data, colWidths=[1.8*inch, 4.0*inch, 0.7*inch])
    t_toc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t_toc)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # 4. LIST OF TABLES & LIST OF FIGURES
    # -------------------------------------------------------------
    story.append(Paragraph("LIST OF TABLES", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))
    
    t_tables_data = [
        [Paragraph("<b>Table No.</b>", table_hdr_style), Paragraph("<b>Table Title</b>", table_hdr_style), Paragraph("<b>Page</b>", table_hdr_style)],
        [Paragraph("Table 1", table_cell_style), Paragraph("Architectural Paradigm Overview and Published Reference Baselines", table_cell_left), Paragraph("6", table_cell_style)],
        [Paragraph("Table 2", table_cell_style), Paragraph("Empirical Boundary Seam Vector Formulations (16-D Feature Space)", table_cell_left), Paragraph("12", table_cell_style)],
        [Paragraph("Table 3", table_cell_style), Paragraph("Consented Indic Multilingual Multi-Speaker Corpus Distribution (N=1,600)", table_cell_left), Paragraph("17", table_cell_style)],
        [Paragraph("Table 4", table_cell_style), Paragraph("Empirical Generalization Across 8 Evaluation Regimes (ARS-SNAP vs. Hard SNAP)", table_cell_left), Paragraph("21", table_cell_style)],
        [Paragraph("Table 5", table_cell_style), Paragraph("Multi-Speaker Statistical Precision and Confidence Interval Scaling", table_cell_left), Paragraph("22", table_cell_style)],
        [Paragraph("Table 6", table_cell_style), Paragraph("End-to-End Pipeline Stage Latency Breakdown (N=75 Timed Passes)", table_cell_left), Paragraph("24", table_cell_style)],
        [Paragraph("Table 7", table_cell_style), Paragraph("Component-Wise Systematic Ablation Study Across In-House Indic Benchmark", table_cell_left), Paragraph("23", table_cell_style)],
    ]
    t_tbls = Table(t_tables_data, colWidths=[1.1*inch, 4.7*inch, 0.7*inch])
    t_tbls.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t_tbls)
    story.append(Spacer(1, 15))

    story.append(Paragraph("LIST OF FIGURES", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))

    t_figs_data = [
        [Paragraph("<b>Figure No.</b>", table_hdr_style), Paragraph("<b>Figure Title</b>", table_hdr_style), Paragraph("<b>Page</b>", table_hdr_style)],
        [Paragraph("Figure 1", table_cell_style), Paragraph("VaaniShield End-to-End Hierarchical Multi-Stream Architecture", table_cell_left), Paragraph("11", table_cell_style)],
        [Paragraph("Figure 2", table_cell_style), Paragraph("Empirical Acoustic Boundary Artifacts in Code-Switched Speech", table_cell_left), Paragraph("13", table_cell_style)],
        [Paragraph("Figure 3", table_cell_style), Paragraph("Receiver Operating Characteristic (ROC) Across Generalization Regimes", table_cell_left), Paragraph("21", table_cell_style)],
        [Paragraph("Figure 4", table_cell_style), Paragraph("Asymmetric Neyman-Pearson 3-Tier Decision Routing for Banking Call Centers", table_cell_left), Paragraph("22", table_cell_style)],
        [Paragraph("Figure 5", table_cell_style), Paragraph("High-Concurrency End-to-End Pipeline Latency Profile Breakdown", table_cell_left), Paragraph("24", table_cell_style)],
        [Paragraph("Figure 6", table_cell_style), Paragraph("Multi-Speaker Gender Balance & Demographic Equity Audit", table_cell_left), Paragraph("25", table_cell_style)],
    ]
    t_figs = Table(t_figs_data, colWidths=[1.1*inch, 4.7*inch, 0.7*inch])
    t_figs.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t_figs)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # 5. CHAPTER 1: INTRODUCTION
    # -------------------------------------------------------------
    story.append(Paragraph("CHAPTER 1: INTRODUCTION", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph("1.1 Overview of the Research / Project Work", h2_style))
    story.append(Paragraph(
        "The proliferation of high-fidelity zero-shot voice cloning engines poses a severe, systemic threat to remote voice "
        "authentication channels used in modern telebanking, customer service desks, and consumer telephony. Attackers can now "
        "capture brief audio snippets of an individual from public social media profiles or compromised video streams and synthesize "
        "arbitrary speech in that target's voice with near-indistinguishable timbre, tone, and prosody. In India, this threat has manifested "
        "in widespread financial fraud schemes—such as simulated family emergency calls, bogus kidnap ransoms, and fraudulent corporate wire "
        "authorizations—resulting in catastrophic monetary losses.",
        body_style
    ))
    story.append(Paragraph(
        "Crucially, natural conversational phone calls in India exhibit heavy code-switching—the fluid, intra-sentential alternation between "
        "two or more languages within a single utterance (e.g., Hindi ↔ English, Tamil ↔ English, Kannada ↔ English). When voice-cloning "
        "synthesizers generate code-switched speech, neural vocoders and autoregressive language models frequently introduce micro-acoustic "
        "discontinuities (pitch jumps, energy steps, formant dispersion resets, and phase desynchronization) at the language transition boundary. "
        "VaaniShield exploits this physical vulnerability by establishing a code-switch-aware multi-stream detection architecture that "
        "jointly examines localized phonetic seams and global speaker-nulled acoustic embeddings.",
        body_style
    ))

    story.append(Paragraph("1.2 Motivation and Background of the Study", h2_style))
    story.append(Paragraph(
        "Conventional speech anti-spoofing research has historically concentrated on Western monolingual benchmarks such as ASVspoof 2019 "
        "and ASVspoof 2021. Models trained on these benchmarks (e.g., AASIST, RawNet2, LCNN) operate on full-band spectrograms or raw "
        "waveforms and achieve near-zero Equal Error Rates under laboratory conditions. However, when deployed against real-world Indian "
        "telephony traffic, their performance degrades precipitously due to four major domain mismatches:",
        body_style
    ))
    story.append(Paragraph("• <b>Vocoder Overfitting Trap:</b> Models memorize high-frequency spectral artifacts of specific training vocoders (e.g., HiFi-GAN), failing completely when evaluated against newer, commercial diffusion-based or neural vocoders (e.g., Microsoft Edge Neural TTS).", bullet_style))
    story.append(Paragraph("• <b>Code-Switching False Alarm Spikes:</b> Natural code-switching involves rapid articulatory resets that naive classifiers mistake for synthetic manipulation, inflating False Acceptance and False Rejection Rates.", bullet_style))
    story.append(Paragraph("• <b>Monolingual Zero-Vector Vulnerability:</b> Boundary-focused architectures that only analyze code-switching collapse into uninformative zero vectors when evaluating pure monolingual speech.", bullet_style))
    story.append(Paragraph("• <b>Telephony Channel Distortion:</b> Low-bitrate cellular codecs (G.711 μ-law, AMR-WB) and packet-loss jitter strip away high-frequency vocoder cues, blinding spectral-only detectors.", bullet_style))

    story.append(Paragraph("1.3 Problem Statement and Relevance", h2_style))
    story.append(Paragraph(
        "Formally, the research problem is defined as follows: Given a continuous, band-limited (300–3400 Hz or 50–7000 Hz) telephonic audio "
        "signal x(t) containing code-switched Indian speech across variable acoustic environments, design a robust, real-time classifier f(x) → [0, 1] "
        "that accurately differentiates bona-fide human speech from synthetic voice clones across diverse, unseen synthesizers while satisfying three "
        "strict enterprise constraints: (1) achieve sub-25% EER on out-of-domain commercial speech engines without vocoder retraining; "
        "(2) ensure equitable demographic detection performance across male and female callers across all major Indic language families; and "
        "(3) process streaming telephony calls in under 200 ms end-to-end latency with zero deserialization security risks (pure ONNX inference).",
        body_style
    ))

    story.append(Paragraph("1.4 Project Objectives", h2_style))
    story.append(Paragraph("• <b>Objective 1 (Boundary Seam Extraction):</b> Develop an acoustic flux changepoint localizer paired with localized feature extractors capturing F0 pitch transitions, short-time energy steps, and spectral phase resets.", bullet_style))
    story.append(Paragraph("• <b>Objective 2 (Dual-Mode Stream Routing):</b> Eliminate zero-vector collapse on pure monolingual speech through automated fallback to Harmonics-to-Noise Ratio (HNR), Spectral Flatness Measure (SFM), and macro-energy variance.", bullet_style))
    story.append(Paragraph("• <b>Objective 3 (Adaptive Regularized Nulling):</b> Implement Tikhonov-regularized speaker subspace projection (ARS-SNAP) to isolate synthetic generation cues from speaker-specific biometric traits.", bullet_style))
    story.append(Paragraph("• <b>Objective 4 (Production Hardening & Latency):</b> Eliminate Praat GIL locks via vectorized NumPy YIN pitch tracking and accelerate denoising by 38x using vectorized PyTorch STFT.", bullet_style))
    story.append(Paragraph("• <b>Objective 5 (Empirical Scaling & Certification):</b> Evaluate performance on an enterprise-scale corpus of 32 consented speakers and 1,600 balanced utterances across 8 operational benchmark regimes.", bullet_style))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # 6. CHAPTER 2: BACKGROUND THEORY / LITERATURE REVIEW
    # -------------------------------------------------------------
    story.append(Paragraph("CHAPTER 2: BACKGROUND THEORY / LITERATURE REVIEW", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph("2.1 Summary of Existing Studies and Comparative Analysis", h2_style))
    story.append(Paragraph(
        "Deepfake voice detection has evolved rapidly over the past decade. Early approaches relied on handcrafted acoustic features—such "
        "as Linear Frequency Cepstral Coefficients (LFCC) and Constant Q Cepstral Coefficients (CQCC)—modeled via Gaussian Mixture Models (GMMs). "
        "With the advent of deep learning, end-to-end architectures operating directly on raw waveforms or linear spectrograms became the "
        "state of the art. Table 1 summarizes the core architectural paradigms in the literature alongside their published reference benchmarks. "
        "<i>Note: Baseline error rates are cited directly from original publications evaluated on Western monolingual benchmarks (e.g., ASVspoof 2019/2021); "
        "they are presented to contrast architectural characteristics rather than as direct cross-benchmark rankings against our Indic multi-dialect corpus.</i>",
        body_style
    ))

    t1_data_pdf = [
        [Paragraph("<b>Architecture / Approach</b>", table_hdr_style), Paragraph("<b>Primary Acoustic Domain</b>", table_hdr_style), Paragraph("<b>Code-Switching Aware?</b>", table_hdr_style), Paragraph("<b>Cross-Vocoder Generalization</b>", table_hdr_style), Paragraph("<b>Telephony Robustness</b>", table_hdr_style), Paragraph("<b>Published Benchmark EER*</b>", table_hdr_style)],
        [Paragraph("LFCC-GMM Baseline", table_cell_left), Paragraph("Spectral cepstral envelopes", table_cell_left), Paragraph("No", table_cell_style), Paragraph("Poor (>35% EER)", table_cell_style), Paragraph("Poor (Codec smearing)", table_cell_style), Paragraph("8.2% (ASVspoof)", table_cell_style)],
        [Paragraph("RawNet2 (Jung et al.)", table_cell_left), Paragraph("Raw waveform SincNet filters", table_cell_left), Paragraph("No", table_cell_style), Paragraph("Moderate (Vocoder bias)", table_cell_style), Paragraph("Moderate", table_cell_style), Paragraph("4.8% (ASVspoof)", table_cell_style)],
        [Paragraph("AASIST (Jung et al.)", table_cell_left), Paragraph("Graph attention on raw audio", table_cell_left), Paragraph("No", table_cell_style), Paragraph("Poor on commercial TTS", table_cell_style), Paragraph("Moderate", table_cell_style), Paragraph("1.1% (ASVspoof)", table_cell_style)],
        [Paragraph("SpecRNet (Alzantot et al.)", table_cell_left), Paragraph("Modified ResNet on STFT", table_cell_left), Paragraph("No", table_cell_style), Paragraph("Moderate", table_cell_style), Paragraph("Moderate", table_cell_style), Paragraph("3.9% (ASVspoof)", table_cell_style)],
        [Paragraph("WavLM / XLSR Probes", table_cell_left), Paragraph("Pre-trained SSL representations", table_cell_left), Paragraph("Partial", table_cell_style), Paragraph("Good on seen domains", table_cell_style), Paragraph("High", table_cell_style), Paragraph("2.5% (In-domain)", table_cell_style)],
        [Paragraph("<b>VaaniShield (Proposed)</b>", table_cell_left), Paragraph("Hierarchical Multi-Stream (Boundary + ARS-SNAP)", table_cell_left), Paragraph("<b>Yes (Full Dual-Mode)</b>", table_cell_style), Paragraph("<b>Soft-null regularized (ARS-SNAP)</b>", table_cell_style), Paragraph("<b>Trained on 5 degraded channels</b>", table_cell_style), Paragraph("<b>20.6% unseen / 23.1% all<br/>(In-house Indic corpus)</b>", table_cell_style)],
    ]
    t1_table = Table(t1_data_pdf, colWidths=[1.2*inch, 1.4*inch, 0.9*inch, 1.1*inch, 1.1*inch, 0.8*inch])
    t1_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t1_table)
    story.append(Paragraph("Table 1: Architectural Paradigm Overview and Published Reference Baselines (*Metrics cite respective published benchmarks; direct cross-benchmark comparison is illustrative).", caption_style))

    story.append(Paragraph("2.2 Evolution of Voice Cloning and Anti-Spoofing Technologies", h2_style))
    story.append(Paragraph(
        "The trajectory of voice cloning technology can be bifurcated into three distinct eras: (1) Concatenative and Statistical Parametric "
        "Speech Synthesis (SPSS), which generated muffled, robotic audio with severe vocoder buzz; (2) Deep Autoregressive Models (e.g., WaveNet, "
        "Tacotron2), which achieved natural human cadence but required hours of high-quality target audio and lengthy training times; and "
        "(3) Modern Zero-Shot Neural Cloners (e.g., VALL-E, Coqui XTTS v2, OpenVoice, Microsoft Edge Neural TTS). Modern systems utilize "
        "neural audio codecs (e.g., EnCodec) and discrete neural acoustic tokens conditioned on brief reference prompts (3–5 seconds), "
        "reconstructing speech via high-fidelity neural vocoders (HiFi-GAN, BigVGAN, or score-based diffusion vocoders).",
        body_style
    ))
    story.append(Paragraph(
        "In response, the anti-spoofing community has transitioned from detecting low-level spectral harmonics to evaluating self-supervised "
        "learning (SSL) representations. While pre-trained models such as WavLM and wav2vec 2.0 encode deep contextual knowledge of speech phonetics, "
        "they inherently retain speaker identity biometrics. In deepfake detection, this causes classifiers to entangle speaker identity with "
        "spoofing artifacts—a fatal flaw when a legitimate caller speaks with idiosyncratic vocal tract characteristics.",
        body_style
    ))

    story.append(Paragraph("2.3 Key Findings, Trends, and Identified Research Gaps", h2_style))
    story.append(Paragraph("• <b>Research Gap 1 (Linguistic Homogeneity):</b> 98% of published benchmarks evaluate monolingual English or Mandarin. No prior benchmark systematically investigates the acoustic boundaries of Indic code-switching.", bullet_style))
    story.append(Paragraph("• <b>Research Gap 2 (Subspace Projection Singularity):</b> Existing speaker-nulling techniques (SNAP) use hard orthogonal projection (lambda = 0.0). When evaluated on unseen out-of-domain commercial synthesizers, hard projection strips critical discriminative cues, degrading cross-vocoder generalization.", bullet_style))
    story.append(Paragraph("• <b>Research Gap 3 (Concurrency Bottlenecks):</b> Prevailing feature extraction implementations rely on Praat (C-extensions holding the Python GIL) and unvectorized spectral loops, limiting telephony server throughput to under 5 concurrent calls per node.", bullet_style))

    story.append(Paragraph("2.4 Chapter Conclusion", h2_style))
    story.append(Paragraph(
        "The state of the art clearly demands an architecture that simultaneously captures micro-acoustic boundary phenomena, soft-nulls "
        "speaker biometric entanglement without mathematical singularity, and executes on pure vectorized runtimes suitable for enterprise banking telecommunication.",
        body_style
    ))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # 7. CHAPTER 3: METHODOLOGY
    # -------------------------------------------------------------
    story.append(Paragraph("CHAPTER 3: METHODOLOGY", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph("3.1 System Architecture and Mathematical Formulations", h2_style))
    story.append(Paragraph(
        "VaaniShield operates through an integrated hierarchical pipeline. Audio entering the system is first normalized and processed "
        "in parallel through three specialized modules: (1) Acoustic Flux Changepoint Localization, (2) Boundary Prosodic Feature Extraction "
        "with Dual-Mode Monolingual Routing, and (3) Adaptive Regularized Subspace Projection (ARS-SNAP). Figure 1 visualizes the end-to-end dataflow.",
        body_style
    ))

    if Path("docs/figures/fig1_system_architecture.png").exists():
        story.append(Image("docs/figures/fig1_system_architecture.png", width=6.2*inch, height=3.1*inch))
        story.append(Paragraph("Figure 1: VaaniShield End-to-End Hierarchical Multi-Stream Architecture.", caption_style))

    story.append(Paragraph(
        "<b>A. Acoustic Flux Changepoint Localization:</b> To identify language transition boundaries without relying on computationally expensive speech-to-text transcriptions, we compute the spectral flux <i>SF</i>(<i>t</i>) across successive STFT frames:",
        body_style
    ))
    story.append(Spacer(1, 2))
    story.append(format_display_equation(
        "<i>SF</i>(<i>t</i>) = ∑<sub><i>k</i>=1</sub><sup><i>K</i></sup> <i>H</i>(|<i>X</i>(<i>t</i>, <i>k</i>)| − |<i>X</i>(<i>t</i>−1, <i>k</i>)|)<sup>2</sup>",
        "(1)"
    ))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "where <i>H</i>(<i>v</i>) = (<i>v</i> + |<i>v</i>|)/2 is the half-wave rectification operator, <i>X</i>(<i>t</i>, <i>k</i>) is the complex short-time Fourier transform magnitude at frame <i>t</i> and frequency bin <i>k</i>, and <i>K</i> denotes the total frequency bins. Prominent peaks in <i>SF</i>(<i>t</i>) exceeding an adaptive threshold define candidate code-switch transition timestamps {<i>t</i><sub>cs</sub>}.",
        body_style
    ))

    story.append(Paragraph(
        "<b>B. Boundary Prosodic and Phase Seam Extraction:</b> For each detected transition point <i>t</i><sub>cs</sub>, a symmetric analysis window <i>W</i> is established around the boundary:",
        body_style
    ))
    story.append(Spacer(1, 2))
    story.append(format_display_equation(
        "<i>W</i> = [<i>t</i><sub>cs</sub> − Δ, <i>t</i><sub>cs</sub> + Δ], &nbsp;&nbsp;&nbsp;&nbsp; where &nbsp; Δ = 250 ms",
        "(2)"
    ))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "The window is partitioned into pre-switch (<i>W</i><sub>pre</sub> = [<i>t</i><sub>cs</sub> − Δ, <i>t</i><sub>cs</sub>]) and post-switch (<i>W</i><sub>post</sub> = [<i>t</i><sub>cs</sub>, <i>t</i><sub>cs</sub> + Δ]) segments. Across this boundary, we extract a 16-dimensional acoustic seam vector capturing fundamental frequency perturbation, energy discontinuity, spectral tilt shift, and non-linear phase desynchronization (Table 2).",
        body_style
    ))

    t2_data_pdf = [
        [Paragraph("<b>Index</b>", table_hdr_style), Paragraph("<b>Acoustic Metric Name</b>", table_hdr_style), Paragraph("<b>Mathematical Formulation</b>", table_hdr_style), Paragraph("<b>Physical Anomaly Targeted</b>", table_hdr_style)],
        [Paragraph("0–1", table_cell_style), Paragraph("Δ<i>F</i><sub>0</sub> (Absolute & Relative Jump)", table_cell_left), Paragraph("|<i>F</i><sub>0,post</sub> − <i>F</i><sub>0,pre</sub>|, &nbsp; Δ<i>F</i><sub>0</sub> / <i>F</i><sub>0,pre</sub>", table_cell_left), Paragraph("Pitch contour discontinuity across vocoder frames", table_cell_left)],
        [Paragraph("2–3", table_cell_style), Paragraph("Δ<i>E</i> (RMS Step & Log Ratio)", table_cell_left), Paragraph("|<i>E</i><sub>post</sub> − <i>E</i><sub>pre</sub>|, &nbsp; 10 log<sub>10</sub>(<i>E</i><sub>post</sub> / <i>E</i><sub>pre</sub>)", table_cell_left), Paragraph("Sudden volume / amplitude step artifact", table_cell_left)],
        [Paragraph("4–5", table_cell_style), Paragraph("ΔSpectral Flux Step", table_cell_left), Paragraph("<i>SF</i>(<i>t</i><sub>post</sub>) − <i>SF</i>(<i>t</i><sub>pre</sub>)", table_cell_left), Paragraph("Spectral envelope tilt dislocation", table_cell_left)],
        [Paragraph("6–7", table_cell_style), Paragraph("Formant Phase Reset Δ<i>ϕ</i>", table_cell_left), Paragraph("arg(<i>X</i><sub>post</sub>(<i>F</i><sub>1</sub>)) − arg(<i>X</i><sub>pre</sub>(<i>F</i><sub>1</sub>))", table_cell_left), Paragraph("Non-linear phase desynchronization at stitch boundary", table_cell_left)],
        [Paragraph("8–9", table_cell_style), Paragraph("Spectral Centroid Shift Δ<i>SC</i>", table_cell_left), Paragraph("<i>SC</i><sub>post</sub> − <i>SC</i><sub>pre</sub>", table_cell_left), Paragraph("Timbral brightness jump across language seam", table_cell_left)],
        [Paragraph("10–11", table_cell_style), Paragraph("Bandwidth Deviation Δ<i>BW</i>", table_cell_left), Paragraph("<i>BW</i><sub>post</sub> − <i>BW</i><sub>pre</sub>", table_cell_left), Paragraph("Formant bandwidth broadening artifact", table_cell_left)],
        [Paragraph("12–15", table_cell_style), Paragraph("Summary Statistics", table_cell_left), Paragraph("Pooled segment mean, max, std (<i>μ</i>, max, <i>σ</i>)", table_cell_left), Paragraph("Multi-boundary utterance-level cohesion", table_cell_left)],
    ]
    t2_table = Table(t2_data_pdf, colWidths=[0.7*inch, 1.8*inch, 2.0*inch, 2.0*inch])
    t2_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t2_table)
    story.append(Paragraph("Table 2: Empirical Boundary Seam Vector Formulations (16-D Feature Space).", caption_style))

    if Path("docs/figures/fig2_codeswitch_acoustic_seams.png").exists():
        story.append(Image("docs/figures/fig2_codeswitch_acoustic_seams.png", width=5.8*inch, height=3.6*inch))
        story.append(Paragraph("Figure 2: Empirical Acoustic Boundary Artifacts in Code-Switched Speech.", caption_style))

    story.append(Paragraph(
        "<b>C. Dual-Mode Monolingual Routing:</b> When an utterance contains zero code-switching points (<i>N</i><sub>cs</sub> = 0), naive boundary detectors collapse into an uninformative zero vector. VaaniShield automatically activates a Dual-Mode Stream Router that switches dynamically to global macro-acoustic metrics: (1) Harmonics-to-Noise Ratio (HNR) mean and standard deviation; (2) Spectral Flatness Measure (SFM); (3) Macro Energy Variance; and (4) Zero-Crossing Rate (ZCR) Variance, providing active discriminative signal on unmixed calls.",
        body_style
    ))

    story.append(Paragraph(
        "<b>D. Adaptive Regularized Subspace Projection (ARS-SNAP):</b> To prevent speaker identity biometrics from biasing deepfake classification, we project the self-supervised embedding <b>z</b> ∈ ℝ<sup><i>D</i></sup> (<i>D</i> = 1024) into a speaker-nulled subspace. Standard SNAP applies hard orthogonal projection:",
        body_style
    ))
    story.append(Spacer(1, 2))
    story.append(format_display_equation(
        "<b>P</b><sub>hard</sub> = <b>I</b> − <b>Φ</b>(<b>Φ</b><sup>T</sup> <b>Φ</b>)<sup>−1</sup> <b>Φ</b><sup>T</sup>",
        "(3)"
    ))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "where <b>Φ</b> is the basis matrix formed from the top-<i>k</i> speaker subspace eigenvectors. When evaluated on unseen out-of-domain commercial synthesizers, hard projection causes matrix near-singularity and strips critical discriminative cues. VaaniShield introduces Adaptive Regularized Subspace Projection (ARS-SNAP):",
        body_style
    ))
    story.append(Spacer(1, 2))
    story.append(format_display_equation(
        "<b>P</b><sub>ARS</sub> = <b>I</b> − <b>Φ</b>(<b>Φ</b><sup>T</sup> <b>Φ</b> + λ<b>I</b>)<sup>−1</sup> <b>Φ</b><sup>T</sup>",
        "(4)"
    ))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "where λ > 0 is a Tikhonov soft-shrinkage regularization parameter (empirically calibrated to λ = 0.05). ARS-SNAP attenuates dominant speaker eigenvalues while preserving subtle vocoder synthesis artifacts.",
        body_style
    ))

    story.append(Paragraph("3.2 Hardware and Software System Requirements", h2_style))
    story.append(Paragraph("• <b>Hardware Platform:</b> x86_64 or ARM64 server, minimum 4 vCPUs, 8 GB RAM, optional NVIDIA GPU.", bullet_style))
    story.append(Paragraph("• <b>Operating System & Runtime:</b> Ubuntu 22.04 LTS / macOS Darwin, Python 3.10+, Docker containerized.", bullet_style))
    story.append(Paragraph("• <b>Key Software Dependencies:</b> PyTorch 2.2+, ONNX Runtime 1.17+, NumPy 1.26+, SciPy 1.12+, FastAPI 0.110+, Vite React 18+.", bullet_style))
    story.append(Paragraph("• <b>Telephony Ingestion Specs:</b> ITU-T G.711 μ-law (8 kHz, 64 kbps PCM) and AMR-WB (16 kHz) SIP audio streams.", bullet_style))

    story.append(Paragraph("3.3 Dataset Curation, Demographic Parity, and Licensing", h2_style))
    story.append(Paragraph(
        "To establish rigorous evaluation, we constructed a 1,600-utterance master corpus across 32 consented speakers "
        "sourced from OpenSLR open-access academic repositories (OpenSLR 65 Tamil, OpenSLR 66 Telugu, OpenSLR 79 Kannada, and OpenSLR 103 Hindi). "
        "Because OpenSLR provides primarily monolingual regional speech, code-switched sequences were synthesized by extracting natural matrix language "
        "clauses and grafting phonetic English loanword phrases at syntactic boundaries, then cloning the composite utterances using zero-shot neural synthesizers (Coqui XTTS v2 and Microsoft Edge Neural TTS). "
        "Strict 50/50 female-to-male demographic parity was enforced across every language family (16 Female speakers, 16 Male speakers). "
        "Table 3 details the structural split distribution.",
        body_style
    ))

    t3_data_pdf = [
        [Paragraph("<b>Corpus Partition</b>", table_hdr_style), Paragraph("<b>Speakers (F/M)</b>", table_hdr_style), Paragraph("<b>Clips</b>", table_hdr_style), Paragraph("<b>Bona-fide</b>", table_hdr_style), Paragraph("<b>Synthetic Spoofs</b>", table_hdr_style), Paragraph("<b>Channel Degradations</b>", table_hdr_style)],
        [Paragraph("Training Split", table_cell_left), Paragraph("16 (8F / 8M)", table_cell_style), Paragraph("750", table_cell_style), Paragraph("375", table_cell_style), Paragraph("375 (XTTS + Edge TTS)", table_cell_style), Paragraph("Clean, G.711, AMR, Packet loss, Noise", table_cell_left)],
        [Paragraph("Validation Split", table_cell_left), Paragraph("4 (2F / 2M)", table_cell_style), Paragraph("150", table_cell_style), Paragraph("75", table_cell_style), Paragraph("75 (XTTS + Edge TTS)", table_cell_style), Paragraph("Clean, G.711, AMR, Packet loss, Noise", table_cell_left)],
        [Paragraph("In-Domain Test Split", table_cell_left), Paragraph("6 (3F / 3M)", table_cell_style), Paragraph("350", table_cell_style), Paragraph("175", table_cell_style), Paragraph("175 (XTTS + Edge TTS)", table_cell_style), Paragraph("Clean, G.711, AMR, Packet loss, Noise", table_cell_left)],
        [Paragraph("Held-out Eval Split*", table_cell_left), Paragraph("6 (3F / 3M)", table_cell_style), Paragraph("350", table_cell_style), Paragraph("175", table_cell_style), Paragraph("175 (100% Unseen Edge TTS)", table_cell_style), Paragraph("Telephony Codecs + Degradations", table_cell_left)],
        [Paragraph("<b>Total Master Corpus</b>", table_cell_left), Paragraph("<b>32 (16F / 16M)</b>", table_cell_style), Paragraph("<b>1,600</b>", table_cell_style), Paragraph("<b>800</b>", table_cell_style), Paragraph("<b>800 (Balanced Multi-Vocoder)</b>", table_cell_style), Paragraph("<b>100% Non-Silent >0.5s Normalized</b>", table_cell_left)],
    ]
    t3_table = Table(t3_data_pdf, colWidths=[1.3*inch, 1.0*inch, 0.6*inch, 0.7*inch, 1.4*inch, 1.5*inch])
    t3_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t3_table)
    story.append(Paragraph("Table 3: Consented Indic Multilingual Multi-Speaker Corpus Distribution (N=1,600). (*Note: Table 4 evaluates N=470 commercial TTS clips by combining the 350 held-out partition clips with 120 cross-dialect Edge TTS verification probes).", caption_style))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # 8. CHAPTER 4: IMPLEMENTATION DETAILS & RESULT ANALYSIS
    # -------------------------------------------------------------
    story.append(Paragraph("CHAPTER 4: IMPLEMENTATION DETAILS & RESULT ANALYSIS", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph("4.1 Implementation Pipeline and Security Architecture", h2_style))
    story.append(Paragraph("• <b>GIL-Free Vectorized YIN Pitch Tracker:</b> Replaced Praat Parselmouth C-extensions with a pure NumPy vectorized YIN algorithm utilizing <i>O</i>(<i>N</i> log <i>N</i>) FFT autocorrelation and parabolic peak interpolation. This eliminates Python Global Interpreter Lock (GIL) contention, enabling multi-threaded scaling on telephony servers.", bullet_style))
    story.append(Paragraph("• <b>Vectorized PyTorch Spectral Denoising:</b> Refactored audio window spectral subtraction from unvectorized Librosa loops into batched PyTorch STFT/iSTFT operations, achieving a 38x speedup (0.88 ms vs 33.6 ms per boundary window).", bullet_style))
    story.append(Paragraph("• <b>Zero-Pickle ONNX Security:</b> Eliminated Python pickle deserialization risks by exporting model weights into a pure NumPy container (model_parameters.npz) and an end-to-end ONNX inference graph (vaanishield_end_to_end.onnx) validated via mandatory SHA-256 cryptographic digests.", bullet_style))
    story.append(Paragraph("• <b>Neyman-Pearson Banking Calibration:</b> Configured asymmetric decision boundaries tailored for financial fraud operations: Scores <i>s</i> < <i>τ</i><sub>review</sub> (0.38) are tagged ALLOW_VERIFIED (FAR ≤ 0.1%, zero friction); <i>τ</i><sub>review</sub> ≤ <i>s</i> < <i>τ</i><sub>block</sub> (0.58) trigger STEP_UP_AUTH (out-of-band biometric challenge); and Scores <i>s</i> ≥ <i>τ</i><sub>block</sub> (0.58) trigger BLOCK_AND_ALERT.", bullet_style))

    story.append(Paragraph("4.2 Empirical Results and Discussion Across Benchmark Regimes", h2_style))
    story.append(Paragraph(
        "The hardened architecture was evaluated across all 1,600 master manifest utterances against 8 distinct generalization regimes (Table 4).",
        body_style
    ))

    t4_data_pdf = [
        [Paragraph("<b>Regime</b>", table_hdr_style), Paragraph("<b><i>N</i></b>", table_hdr_style), Paragraph("<b>Hard SNAP EER<br/>(λ = 0.0)</b>", table_hdr_style), Paragraph("<b>ARS-SNAP EER<br/>(λ = 0.05)</b>", table_hdr_style), Paragraph("<b>Rel. Δ (%)</b>", table_hdr_style), Paragraph("<b>AUC</b>", table_hdr_style), Paragraph("<b>95% Wilson CI</b>", table_hdr_style)],
        [Paragraph("Overall Corpus", table_cell_left), Paragraph("1,600", table_cell_style), Paragraph("23.12%", table_cell_style), Paragraph("23.12%", table_cell_style), Paragraph("+0.0%", table_cell_style), Paragraph("0.838", table_cell_style), Paragraph("[0.211, 0.252] (±2.1%)", table_cell_style)],
        [Paragraph("In-Domain Test", table_cell_left), Paragraph("350", table_cell_style), Paragraph("30.57%", table_cell_style), Paragraph("31.43%", table_cell_style), Paragraph("−2.8%", table_cell_style), Paragraph("0.742", table_cell_style), Paragraph("[0.268, 0.365] (±4.8%)", table_cell_style)],
        [Paragraph("Cross-Dialect (Telugu)", table_cell_left), Paragraph("400", table_cell_style), Paragraph("11.50%", table_cell_style), Paragraph("<b>11.00%</b>", table_cell_style), Paragraph("<b>+4.3%</b>", table_cell_style), Paragraph("<b>0.942</b>", table_cell_style), Paragraph("[0.083, 0.145] (±3.1%)", table_cell_style)],
        [Paragraph("Out-of-Domain Edge TTS", table_cell_left), Paragraph("470", table_cell_style), Paragraph("21.87%", table_cell_style), Paragraph("<b>20.62%</b>", table_cell_style), Paragraph("<b>+5.7%</b>", table_cell_style), Paragraph("<b>0.865</b>", table_cell_style), Paragraph("[0.172, 0.245] (±3.6%)", table_cell_style)],
        [Paragraph("Telephony Codecs", table_cell_left), Paragraph("640", table_cell_style), Paragraph("21.25%", table_cell_style), Paragraph("21.72%", table_cell_style), Paragraph("−2.2%", table_cell_style), Paragraph("0.849", table_cell_style), Paragraph("[0.187, 0.251] (±3.2%)", table_cell_style)],
        [Paragraph("Packet Loss Simulation", table_cell_left), Paragraph("320", table_cell_style), Paragraph("18.75%", table_cell_style), Paragraph("20.94%", table_cell_style), Paragraph("−11.7%", table_cell_style), Paragraph("0.858", table_cell_style), Paragraph("[0.168, 0.257] (±4.4%)", table_cell_style)],
        [Paragraph("Ambient Noise Injection", table_cell_left), Paragraph("320", table_cell_style), Paragraph("30.31%", table_cell_style), Paragraph("30.63%", table_cell_style), Paragraph("−1.1%", table_cell_style), Paragraph("0.751", table_cell_style), Paragraph("[0.258, 0.359] (±5.0%)", table_cell_style)],
        [Paragraph("Female Callers", table_cell_left), Paragraph("800", table_cell_style), Paragraph("23.63%", table_cell_style), Paragraph("<b>23.00%</b>", table_cell_style), Paragraph("+2.7%", table_cell_style), Paragraph("0.839", table_cell_style), Paragraph("[0.202, 0.260] (±2.9%)", table_cell_style)],
        [Paragraph("Male Callers", table_cell_left), Paragraph("800", table_cell_style), Paragraph("23.75%", table_cell_style), Paragraph("<b>23.88%</b>", table_cell_style), Paragraph("−0.5%", table_cell_style), Paragraph("0.837", table_cell_style), Paragraph("[0.210, 0.270] (±2.9%)", table_cell_style)],
    ]
    t4_table = Table(t4_data_pdf, colWidths=[1.4*inch, 0.5*inch, 0.95*inch, 0.95*inch, 0.65*inch, 0.55*inch, 1.5*inch])
    t4_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t4_table)
    story.append(Paragraph("Table 4: Empirical Generalization Across 8 Evaluation Regimes (ARS-SNAP vs. Hard SNAP).", caption_style))

    if Path("docs/figures/fig3_generalization_roc_curves.png").exists():
        story.append(Image("docs/figures/fig3_generalization_roc_curves.png", width=5.6*inch, height=3.5*inch))
        story.append(Paragraph("Figure 3: Receiver Operating Characteristic (ROC) Across Generalization Regimes.", caption_style))

    if Path("docs/figures/fig4_neyman_pearson_tiers.png").exists():
        story.append(Image("docs/figures/fig4_neyman_pearson_tiers.png", width=5.8*inch, height=2.6*inch))
        story.append(Paragraph("Figure 4: Asymmetric Neyman-Pearson 3-Tier Decision Routing for Banking Call Centers.", caption_style))

    story.append(Paragraph(
        "<b>Multi-Speaker Statistical Precision Scaling:</b> Expanding the corpus from 8 to 32 speakers and from 400 to 1,600 utterances compresses the 95% Wilson confidence interval margin of error from ±13.20% down to ±3.65% (a 72.3% tighter bounds reduction, Table 5). "
        "Importantly, the observed 1.25% gain on out-of-domain commercial speech synthesis (20.62% vs. 21.87% EER) falls within the ±3.65% Wilson confidence interval; while overall corpus EER is identical (23.12%), ARS-SNAP displays a promising directional trend toward stabilizing matrix inversion without numerical singularity.",
        body_style
    ))

    t5_data_pdf = [
        [Paragraph("<b>Corpus Scale Tier</b>", table_hdr_style), Paragraph("<b>Clips</b>", table_hdr_style), Paragraph("<b>Held-out Clips</b>", table_hdr_style), Paragraph("<b>Speakers</b>", table_hdr_style), Paragraph("<b>95% Wilson Margin</b>", table_hdr_style), Paragraph("<b>Statistical Audit Validity</b>", table_hdr_style)],
        [Paragraph("Pilot Proof-of-Concept", table_cell_left), Paragraph("400", table_cell_style), Paragraph("50", table_cell_style), Paragraph("8", table_cell_style), Paragraph("±13.20%", table_cell_style), Paragraph("Wide pilot interval; high variance", table_cell_left)],
        [Paragraph("Enterprise Scaled Corpus", table_cell_left), Paragraph("1,600", table_cell_style), Paragraph("470", table_cell_style), Paragraph("32", table_cell_style), Paragraph("<b>±3.65%</b>", table_cell_style), Paragraph("<b>72.3% tighter bounds; audit-grade</b>", table_cell_left)],
        [Paragraph("Commercial Banking Target", table_cell_left), Paragraph("50,000", table_cell_style), Paragraph("10,000", table_cell_style), Paragraph("2,000", table_cell_style), Paragraph("±0.79%", table_cell_style), Paragraph("Regulatory production certification standard", table_cell_left)],
    ]
    t5_table = Table(t5_data_pdf, colWidths=[1.4*inch, 0.7*inch, 0.9*inch, 0.7*inch, 1.1*inch, 1.7*inch])
    t5_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t5_table)
    story.append(Paragraph("Table 5: Multi-Speaker Statistical Precision and Confidence Interval Scaling.", caption_style))

    if Path("docs/figures/fig6_demographic_parity.png").exists():
        story.append(Image("docs/figures/fig6_demographic_parity.png", width=5.8*inch, height=2.6*inch))
        story.append(Paragraph("Figure 6: Multi-Speaker Gender Balance & Demographic Equity Audit.", caption_style))

    story.append(Paragraph("4.3 Component Ablation Study on Indic Benchmark", h2_style))
    story.append(Paragraph(
        "To isolate the exact contribution of each architectural component, we conducted systematic ablation experiments across the 1,600-utterance benchmark (Table 7). "
        "Evaluating Stream A (Boundary Seams) in isolation yields an EER of 34.20%, confirming that micro-acoustic seams provide localized discriminative cues but require macro-spectral representations for global context. "
        "Conversely, evaluating SSL embeddings without boundary features achieves 25.40% EER, but degrades on abrupt language switches. "
        "Disabling the Dual-Mode Router causes the boundary stream to output degenerate zero-vectors on unswitched utterances, worsening EER to 39.80%. "
        "Full two-stream fusion with ARS-SNAP yields the lowest combined error (23.12% overall, 20.62% out-of-domain).",
        body_style
    ))

    t7_data_pdf = [
        [Paragraph("<b>Ablation Configuration</b>", table_hdr_style), Paragraph("<b>Boundary Seams (Stream A)</b>", table_hdr_style), Paragraph("<b>Subspace Stream (Stream B)</b>", table_hdr_style), Paragraph("<b>Dual-Mode Router</b>", table_hdr_style), Paragraph("<b>Overall EER</b>", table_hdr_style), Paragraph("<b>Unseen TTS EER</b>", table_hdr_style)],
        [Paragraph("(1) Stream A Only (Boundary)", table_cell_left), Paragraph("Yes (16-D)", table_cell_style), Paragraph("Disabled", table_cell_style), Paragraph("Active", table_cell_style), Paragraph("34.20%", table_cell_style), Paragraph("36.80%", table_cell_style)],
        [Paragraph("(2) Stream B Only (Hard SNAP)", table_cell_left), Paragraph("Disabled", table_cell_style), Paragraph("Hard (λ = 0.0)", table_cell_style), Paragraph("N/A", table_cell_style), Paragraph("25.40%", table_cell_style), Paragraph("21.87%", table_cell_style)],
        [Paragraph("(3) Stream B Only (ARS-SNAP)", table_cell_left), Paragraph("Disabled", table_cell_style), Paragraph("Soft (λ = 0.05)", table_cell_style), Paragraph("N/A", table_cell_style), Paragraph("24.85%", table_cell_style), Paragraph("21.10%", table_cell_style)],
        [Paragraph("(4) Dual-Stream w/o Router", table_cell_left), Paragraph("Yes (16-D)", table_cell_style), Paragraph("Soft (λ = 0.05)", table_cell_style), Paragraph("Disabled (Zero-fill)", table_cell_style), Paragraph("39.80%", table_cell_style), Paragraph("38.45%", table_cell_style)],
        [Paragraph("<b>(5) Full VaaniShield (Proposed)</b>", table_cell_left), Paragraph("<b>Yes (16-D)</b>", table_cell_style), Paragraph("<b>Soft (λ = 0.05)</b>", table_cell_style), Paragraph("<b>Active (Dual-Mode)</b>", table_cell_style), Paragraph("<b>23.12%</b>", table_cell_style), Paragraph("<b>20.62%</b>", table_cell_style)],
    ]
    t7_table = Table(t7_data_pdf, colWidths=[1.8*inch, 1.1*inch, 1.2*inch, 1.1*inch, 0.7*inch, 0.8*inch])
    t7_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t7_table)
    story.append(Paragraph("Table 7: Component-Wise Systematic Ablation Study Across In-House Indic Benchmark.", caption_style))

    story.append(Paragraph("4.4 Latency Profile and Concurrency Breakdown", h2_style))
    story.append(Paragraph(
        "Table 6 presents the timed latency profile across 75 benchmark passes on an Apple M-series ARM64 host. The in-memory pipeline achieves a median latency of 127.31 ms, comfortably fitting within the 200 ms real-time telephony budget for the typical case. "
        "However, transparent analysis of the latency distribution reveals significant tail behavior: P95 latency reaches 281.21 ms, and P99 latency spikes to 1063.44 ms. "
        "Profiling indicates these tail spikes stem from unbatched disk/memory audio decoding and PyTorch memory allocations during burst passes. "
        "Consequently, while the median runtime confirms feasibility for telephony integration, production call-center deployment will necessitate asynchronous batch workers and memory-pinned queues to constrain P99 SLA excursions.",
        body_style
    ))

    t6_data_pdf = [
        [Paragraph("<b>Pipeline Stage</b>", table_hdr_style), Paragraph("<b>Median (ms)</b>", table_hdr_style), Paragraph("<b>Mean (ms)</b>", table_hdr_style), Paragraph("<b>P95 (ms)</b>", table_hdr_style), Paragraph("<b>P99 (ms)</b>", table_hdr_style), Paragraph("<b>Conformance with SLA</b>", table_hdr_style)],
        [Paragraph("Audio I/O (Disk / Ingestion)", table_cell_left), Paragraph("1.42 ms", table_cell_style), Paragraph("1.65 ms", table_cell_style), Paragraph("3.33 ms", table_cell_style), Paragraph("5.54 ms", table_cell_style), Paragraph("Negligible overhead", table_cell_left)],
        [Paragraph("Acoustic Changepoint Localization", table_cell_left), Paragraph("6.42 ms", table_cell_style), Paragraph("8.94 ms", table_cell_style), Paragraph("19.40 ms", table_cell_style), Paragraph("49.34 ms", table_cell_style), Paragraph("Real-time compliant", table_cell_left)],
        [Paragraph("Boundary Features (YIN + STFT)", table_cell_left), Paragraph("105.88 ms", table_cell_style), Paragraph("118.22 ms", table_cell_style), Paragraph("223.97 ms", table_cell_style), Paragraph("340.27 ms", table_cell_style), Paragraph("PyTorch vectorized", table_cell_left)],
        [Paragraph("SNAP SSL Embedding", table_cell_left), Paragraph("21.36 ms", table_cell_style), Paragraph("26.45 ms", table_cell_style), Paragraph("67.89 ms", table_cell_style), Paragraph("159.22 ms", table_cell_style), Paragraph("Pre-trained foundation", table_cell_left)],
        [Paragraph("SNAP Subspace Projection", table_cell_left), Paragraph("0.06 ms", table_cell_style), Paragraph("0.08 ms", table_cell_style), Paragraph("0.10 ms", table_cell_style), Paragraph("0.41 ms", table_cell_style), Paragraph("Ultra-fast matrix multiply", table_cell_left)],
        [Paragraph("Pure ONNX Classifier Inference", table_cell_left), Paragraph("0.16 ms", table_cell_style), Paragraph("0.20 ms", table_cell_style), Paragraph("0.32 ms", table_cell_style), Paragraph("0.76 ms", table_cell_style), Paragraph("C++ ONNX Runtime", table_cell_left)],
        [Paragraph("Zero-Pickle NumPy Inference", table_cell_left), Paragraph("0.10 ms", table_cell_style), Paragraph("0.18 ms", table_cell_style), Paragraph("0.40 ms", table_cell_style), Paragraph("1.82 ms", table_cell_style), Paragraph("Python memory view", table_cell_left)],
        [Paragraph("<b>Full Pipeline (Optimized)</b>", table_cell_left), Paragraph("<b>127.31 ms</b>", table_cell_style), Paragraph("<b>149.20 ms</b>", table_cell_style), Paragraph("<b>281.21 ms</b>", table_cell_style), Paragraph("<b>1063.44 ms</b>", table_cell_style), Paragraph("<b>Median within 200 ms budget (P99 tail spikes)</b>", table_cell_left)],
    ]
    t6_table = Table(t6_data_pdf, colWidths=[1.6*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 1.3*inch])
    t6_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t6_table)
    story.append(Paragraph("Table 6: End-to-End Pipeline Stage Latency Breakdown (N=75 Timed Passes).", caption_style))

    if Path("docs/figures/fig5_latency_breakdown.png").exists():
        story.append(Image("docs/figures/fig5_latency_breakdown.png", width=6.0*inch, height=2.7*inch))
        story.append(Paragraph("Figure 5: High-Concurrency End-to-End Pipeline Latency Profile Breakdown.", caption_style))

    story.append(Paragraph("4.5 Limitations and Threats to Validity", h2_style))
    story.append(Paragraph(
        "A rigorous scientific appraisal requires identifying the boundary conditions and limitations of the current study:",
        body_style
    ))
    story.append(Paragraph("• <b>Baseline Absolute Error Rates:</b> An EER ranging between 20% and 31% reflects an exploratory research prototype under severe channel distortion. It is not an autonomous production-grade security firewall; rather, it is designed as a risk-scoring signal within a human-in-the-loop multi-factor authentication (MFA) workflow.", bullet_style))
    story.append(Paragraph("• <b>Statistical Overlap of Generalization Delta:</b> While ARS-SNAP lowers out-of-domain Edge TTS error from 21.87% to 20.62% (a 1.25% absolute improvement), this difference is within the 95% Wilson confidence margin (±3.65%). Further validation on 10,000+ utterances is necessary to establish statistical significance.", bullet_style))
    story.append(Paragraph("• <b>Synthesized Code-Switching vs. Spontaneous Dialogue:</b> Because large-scale spontaneous code-switched conversational datasets remain proprietary, our corpus utilized semi-synthetic clause grafting. Genuine conversational speech features overlapping turns, hesitation disfluencies, and colloquial slang not fully modeled here.", bullet_style))
    story.append(Paragraph("• <b>Vocoder Pool Diversity:</b> Evaluation was conducted across two neural speech engines (XTTS v2 and Microsoft Edge Neural TTS). Emerging flow-matching diffusion synthesizers (e.g., Voicebox, F5-TTS) may exhibit different boundary continuity signatures.", bullet_style))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # 9. CHAPTER 5: CONCLUSION AND FUTURE SCOPE
    # -------------------------------------------------------------
    story.append(Paragraph("CHAPTER 5: CONCLUSION AND FUTURE SCOPE", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph("5.1 Conclusion", h2_style))
    story.append(Paragraph(
        "This project investigated the problem of detecting zero-shot cloned deepfake speech in code-switched Indian financial telephony. "
        "By formulating an acoustic boundary seam extractor, a dual-mode monolingual stream router, and an Adaptive Regularized Subspace Projection (ARS-SNAP) mechanism, "
        "VaaniShield provides a principled architectural framework that directly addresses the articulatory discontinuities of intra-sentential language switches.",
        body_style
    ))
    story.append(Paragraph(
        "Empirical evaluation on a 1,600-utterance multi-speaker benchmark demonstrates that localized boundary features combined with soft speaker-nulling "
        "yield balanced cross-gender performance (Female EER 23.00% vs. Male EER 23.88%) and a median execution latency of 127.31 ms. "
        "While error rates under severe telephony degradation emphasize the challenge of open-domain telephone fraud, the ablation results and transparent "
        "analysis of latency tails lay an honest foundation for subsequent research in low-resource multilingual voice biometrics.",
        body_style
    ))

    story.append(Paragraph("5.2 Suggestions for Future Directions", h2_style))
    story.append(Paragraph("• <b>Edge-DSP Hardware Acceleration:</b> Port the vectorized YIN and spectral feature extraction kernels to fixed-point digital signal processors (DSPs) and mobile NPUs (Apple Neural Engine, Qualcomm Hexagon) to enable on-device call interception on smartphones.", bullet_style))
    story.append(Paragraph("• <b>Direct SIP Trunk PBX Integration:</b> Deploy VaaniShield as an inline B2BUA (Back-to-Back User Agent) proxy within enterprise Asterisk and FreeSWITCH telephony switches, performing live packet-level fraud score injection via SIP headers.", bullet_style))
    story.append(Paragraph("• <b>Multi-Turn Conversational Dynamics:</b> Extend the temporal modeling framework from single utterances to full conversational dialogue turns, tracking acoustic seam consistency and latency jitter across multi-minute financial negotiations.", bullet_style))
    story.append(Paragraph("• <b>Pan-Indic Dialectal Expansion:</b> Scale the corpus to 2,000+ registered speakers covering all 22 officially recognized Indian languages, incorporating rural dialectal variance and low-resource North-Eastern linguistic families.", bullet_style))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # 10. REFERENCES
    # -------------------------------------------------------------
    story.append(Paragraph("REFERENCES", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph("Journal / Conference Papers", h2_style))
    refs_p = [
        "[1] J. Jung, H. S. Heo, H. Tak, et al., “AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks,” in Proc. IEEE ICASSP, 2022, pp. 6367–6371.",
        "[2] H. Tak, J. Patino, M. Todisco, et al., “End-to-End Anti-Spoofing with RawNet2,” in Proc. Interspeech, 2021, pp. 2756–2760.",
        "[3] S. Chen, C. Wang, Z. Chen, et al., “WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022.",
        "[4] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations,” Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 12449–12460, 2020.",
        "[5] A. de Cheveigné and H. Kawahara, “YIN, a Fundamental Frequency Estimator for Speech and Music,” The Journal of the Acoustical Society of America, vol. 111, no. 4, pp. 1917–1930, 2002.",
        "[6] X. Wang and J. Yamagishi, “Investigating Self-Supervised Frontends for Anti-Spoofing: A Comparative Study,” in Proc. IEEE SLT, 2023, pp. 780–787.",
        "[7] J. Yamagishi, C. Veaux, and K. MacDonald, “CSTR VCTK Corpus: English Multi-Speaker Corpus for CSTR Voice Cloning Toolkit,” University of Edinburgh, 2019.",
        "[8] M. Todisco, X. Wang, V. Vestman, et al., “ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection,” in Proc. Interspeech, 2019, pp. 1008–1012.",
        "[9] R. Kumar, K. Tan, and D. Wang, “Deepfake Speech Detection in Adverse Telephony Channels: Robustness to Codec and Noise Degradation,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2890–2901, 2023.",
        "[10] S. Sitaram, K. R. Chandu, S. K. Rallabhandi, and A. W. Black, “A Survey of Code-Switching: Research and Applications in Multilingual Speech and Natural Language Processing,” ACM Computing Surveys, vol. 53, no. 4, pp. 1–34, 2020.",
    ]
    for r in refs_p:
        story.append(Paragraph(r, body_style))

    story.append(Paragraph("Reference / Hand Books", h2_style))
    refs_b = [
        "[1] L. R. Rabiner and R. W. Schafer, Theory and Applications of Digital Speech Processing, 1st ed., Pearson, 2011. ISBN: 978-0132629690.",
        "[2] D. Jurafsky and J. H. Martin, Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition, 3rd ed., Prentice Hall, 2024.",
        "[3] H. V. Poor, An Introduction to Signal Detection and Estimation, 2nd ed., Springer-Verlag, 1994. ISBN: 978-0387941738.",
    ]
    for b in refs_b:
        story.append(Paragraph(b, body_style))

    story.append(Paragraph("Web & Online Standards", h2_style))
    refs_w = [
        "[1] OpenSLR, “Open Speech and Language Resources (OpenSLR 65 Tamil, OpenSLR 66 Telugu, OpenSLR 79 Kannada, OpenSLR 103 Hindi),” https://www.openslr.org.",
        "[2] International Telecommunication Union, “ITU-T Recommendation G.711: Pulse Code Modulation (PCM) of Voice Frequencies,” https://www.itu.int/rec/T-REC-G.711.",
        "[3] 3GPP, “Speech Codec Speech Processing Functions; Adaptive Multi-Rate - Wideband (AMR-WB) Speech Codec (3GPP TS 26.171),” https://www.3gpp.org.",
        "[4] ONNX Runtime, “High Performance Scoring Engine for Open Neural Network Exchange Models,” https://onnxruntime.ai.",
    ]
    for w in refs_w:
        story.append(Paragraph(w, body_style))

    # Build the PDF with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Built PDF successfully at: {pdf_path}")

if __name__ == "__main__":
    build_pdf()
