"""
VaaniShield Forensic Audit Certificate & Incident Report Generator.
Compliant with RBI Cyber Security Framework, CERT-In advisory directives,
and DPDPA 2023 Non-Semantic Privacy principles (Zero transcript leakage).
"""

import io
import os
import hashlib
import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def compute_sha256(audio_path: str) -> str:
    """Computes SHA-256 fingerprint of the original audio file."""
    if not os.path.isfile(audio_path):
        return "UNKNOWN_HASH_FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(audio_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_forensic_certificate_pdf(
    analysis_result: Dict[str, Any],
    audio_path: Optional[str] = None,
    call_id: Optional[str] = None
) -> bytes:
    """
    Renders an official forensic incident audit report as PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#2563EB'),
        spaceAfter=12
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#334155')
    )

    body_bold = ParagraphStyle(
        'DocBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    disclaimer_style = ParagraphStyle(
        'DocDisclaimer',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#64748B')
    )

    elements = []

    # 1. Header Banner & Verification Seal
    elements.append(Paragraph("VAANISHIELD FORENSIC INCIDENT & AUDIT REPORT", title_style))
    elements.append(Paragraph("AUTOMATED ACOUSTIC CHANGE-POINT & BIOMETRIC VERIFICATION REPORT", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=10))

    # Timestamp and Meta
    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    file_hash = analysis_result.get("audio_sha256")
    if not file_hash and audio_path:
        file_hash = compute_sha256(audio_path)
    if not file_hash:
        file_hash = "NOT_PROVIDED"

    assigned_call_id = call_id or (f"VS-AUDIT-{file_hash[:10].upper()}" if file_hash != "NOT_PROVIDED" else f"VS-AUDIT-{int(datetime.datetime.now().timestamp())}")

    decision = analysis_result.get("decision", "ALLOW_VERIFIED")
    p_spoof = float(analysis_result.get("calibrated_p_spoof", analysis_result.get("calibrated_probability", 0.0)))
    fusion_score = float(analysis_result.get("fusion_score", 0.0))
    tier = analysis_result.get("tier", "SECURE")

    # Decision Badge Colors
    if decision == "BLOCK_AND_ALERT" or p_spoof >= 0.52:
        badge_bg = colors.HexColor('#FEE2E2')
        badge_fg = colors.HexColor('#991B1B')
        verdict_str = f"RISK TIER: {tier} - SPOOF DETECTED [ACTION: {decision}]"
    elif decision == "STEP_UP_AUTH" or p_spoof >= 0.48:
        badge_bg = colors.HexColor('#FEF3C7')
        badge_fg = colors.HexColor('#92400E')
        verdict_str = f"RISK TIER: {tier} - SUSPICIOUS / UNCERTAIN [ACTION: {decision}]"
    else:
        badge_bg = colors.HexColor('#DCFCE7')
        badge_fg = colors.HexColor('#166534')
        verdict_str = f"RISK TIER: {tier} - VERIFIED GENUINE [ACTION: {decision}]"

    # Verdict Box
    verdict_table = Table(
        [[Paragraph(f"<b>OPERATIONAL ACTION: {verdict_str}</b>", ParagraphStyle('Verdict', parent=body_bold, fontSize=10, textColor=badge_fg, alignment=1))]],
        colWidths=[540]
    )
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), badge_bg),
        ('BOX', (0,0), (-1,-1), 1.5, badge_fg),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 10))

    # 2. Call Integrity & Session Identification Table
    elements.append(Paragraph("1. Session Integrity & Cryptographic Identity", section_heading))
    
    meta_data = [
        [Paragraph("<b>Audit Report ID:</b>", body_style), Paragraph(assigned_call_id, body_bold),
         Paragraph("<b>Timestamp (UTC):</b>", body_style), Paragraph(generated_at, body_style)],
        [Paragraph("<b>Audio SHA-256 Digest:</b>", body_style), Paragraph(f"<font name='Courier'>{file_hash[:32]}...<br/>{file_hash[32:] if len(file_hash) > 32 else ''}</font>", body_style),
         Paragraph("<b>Sampling Spec:</b>", body_style), Paragraph("8,000 Hz Narrowband Standardized", body_style)],
        [Paragraph("<b>Pipeline Engine:</b>", body_style), Paragraph("VaaniShield Research & Detection Engine", body_style),
         Paragraph("<b>Privacy Framework:</b>", body_style), Paragraph("Non-Semantic Acoustic Processing", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[120, 170, 110, 140])
    meta_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10))

    # 3. Biometric & Calibration Risk Assessment
    elements.append(Paragraph("2. Biometric Authenticity & Risk Metrics", section_heading))
    ci_lower = analysis_result.get("ci_lower")
    ci_upper = analysis_result.get("ci_upper")
    ci_display = f"[{ci_lower * 100:.1f}%, {ci_upper * 100:.1f}%]" if (ci_lower is not None and ci_upper is not None) else "N/A"

    risk_data = [
        [
            Paragraph("<b>Metric</b>", body_bold),
            Paragraph("<b>Measured Value</b>", body_bold),
            Paragraph("<b>Operating Threshold / Reference</b>", body_bold),
            Paragraph("<b>Status</b>", body_bold)
        ],
        [
            Paragraph("Calibrated Spoof Probability", body_style),
            Paragraph(f"<b>{p_spoof * 100:.1f}%</b>", body_style),
            Paragraph("Decision Threshold &tau;<sub>block</sub> = 52.0%", body_style),
            Paragraph(tier, body_bold)
        ],
        [
            Paragraph("Calibrated Risk Tier", body_style),
            Paragraph(f"Tier: {tier}", body_style),
            Paragraph("SECURE (<48%) | ELEVATED (48-52%) | CRITICAL (&ge;52%)", body_style),
            Paragraph(decision, body_bold)
        ],
        [
            Paragraph("95% Bootstrap Confidence Interval", body_style),
            Paragraph(ci_display, body_style),
            Paragraph("N=100 bootstrap resamples", body_style),
            Paragraph("COMPUTED" if ci_lower is not None else "NOT_COMPUTED", body_style)
        ],
    ]

    if fusion_score > 0.0:
        risk_data.append([
            Paragraph("Fusion Decision Score", body_style),
            Paragraph(f"{fusion_score:.4f}", body_style),
            Paragraph("Raw ensemble logits / distance", body_style),
            Paragraph("EVALUATED", body_style)
        ])

    boundary_res = analysis_result.get("boundary_analysis", {})
    if boundary_res:
        seams_detected = boundary_res.get("seams_detected", 0)
        risk_data.append([
            Paragraph("Acoustic Change-Points", body_style),
            Paragraph(f"<b>{seams_detected}</b> seam point(s)", body_style),
            Paragraph("Spectral flux & continuity window analysis", body_style),
            Paragraph("SEAMS DETECTED" if seams_detected > 0 else "HOMOGENEOUS", body_bold)
        ])

    risk_table = Table(risk_data, colWidths=[160, 120, 160, 100])
    risk_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 10))

    # 4. Forensic Micro-Seam Breakdown (Zero transcript leakage)
    elements.append(Paragraph("3. Micro-Seam Attribution & Acoustic Transition Signatures", section_heading))
    elements.append(Paragraph(
        "Notice: Acoustic change-points and spectral continuity are evaluated without speech-to-text "
        "transcription or text logging to preserve acoustic non-semantic privacy.",
        disclaimer_style
    ))
    elements.append(Spacer(1, 4))

    transitions = analysis_result.get("language_transitions", [])
    boundary_points = boundary_res.get("seam_timestamps", [])

    seam_data = [
        [
            Paragraph("<b>Timestamp (s)</b>", body_bold),
            Paragraph("<b>Transition Type / Boundary</b>", body_bold),
            Paragraph("<b>Acoustic Artifact Cue</b>", body_bold),
            Paragraph("<b>Contribution Score</b>", body_bold)
        ]
    ]

    if transitions:
        for t in transitions[:6]:
            t_time = t.get("timestamp_sec", t.get("timestamp_seconds", 0.0))
            trans_label = t.get("language_transition", "Acoustic Boundary")
            desc = t.get("description", t.get("cue", "Acoustic discontinuity detected"))
            c_score = t.get("contribution_score", t.get("anomaly_score", 0.0))
            score_str = f"{c_score * 100:.1f}%" if isinstance(c_score, float) and c_score <= 1.0 else f"{c_score:.2f}"

            seam_data.append([
                Paragraph(f"{float(t_time):.2f}s", body_style),
                Paragraph(str(trans_label), body_style),
                Paragraph(str(desc), body_style),
                Paragraph(score_str, body_bold)
            ])
    elif boundary_points:
        for b in boundary_points[:6]:
            b_time = b.get("time_seconds", 0.0) if isinstance(b, dict) else float(b)
            score_val = b.get("anomaly_score", 0.0) if isinstance(b, dict) else 0.0
            score_str = f"{score_val*100:.1f}%" if score_val > 0.0 else "N/A"
            seam_data.append([
                Paragraph(f"{b_time:.2f}s", body_style),
                Paragraph("Acoustic Spectral Flux Point", body_style),
                Paragraph("Spectral flux & envelope discontinuity point", body_style),
                Paragraph(score_str, body_bold)
            ])
    else:
        seam_data.append([
            Paragraph("N/A", body_style),
            Paragraph("Continuous Audio - No Boundary Splices Localized", body_style),
            Paragraph("Consistent vocal tract dynamics across inspected duration", body_style),
            Paragraph("0.0%", body_style)
        ])

    seam_table = Table(seam_data, colWidths=[90, 160, 200, 90])
    seam_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(seam_table)
    elements.append(Spacer(1, 14))

    # 5. Technical Disclaimer Box
    legal_text = (
        "<b>TECHNICAL ASSESSMENT NOTICE:</b> This document contains the algorithmic output produced by "
        "the VaaniShield automated voice authenticity analysis system. Metrics represent statistical inferences "
        "derived from acoustic boundary features, speaker-subspace projection, and calibrated probabilistic thresholds. "
        "This automated report is intended to assist technical evaluation and should be reviewed alongside institutional "
        "operational guidelines. Authenticity of the evaluated sample can be referenced against the cryptographic SHA-256 digest."
    )
    legal_table = Table(
        [[Paragraph(legal_text, disclaimer_style)]],
        colWidths=[540]
    )
    legal_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#94A3B8')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(KeepTogether(legal_table))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
