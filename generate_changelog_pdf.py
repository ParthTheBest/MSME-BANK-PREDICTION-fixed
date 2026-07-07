"""
generate_changelog_pdf.py
=========================
Generates the professional desaturated banking-standard changelog PDF.
Writes directly to 'antigravity_changelog.pdf' in the artifact and workspace folders.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to calculate total page count and draw running headers/footers."""
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
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#2A3E50")) # Cool Slate Navy
        
        # Header (Top of Page)
        self.drawString(54, 750, "MSME CREDIT RISK ENGINE: COMPLIANCE AUDIT & RESOLUTION REPORT")
        self.setStrokeColor(colors.HexColor("#BDC3C7")) # Cool Grey line
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)
        
        # Footer (Bottom of Page)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7F8C8D"))
        self.drawString(54, 36, "CONFIDENTIAL — BANK INTERNAL USE ONLY")
        self.drawRightString(558, 36, f"Page {self._pageNumber} of {page_count}")
        self.line(54, 48, 558, 48)
        self.restoreState()


def build_pdf(filename):
    # Setup document geometry with strict 0.75" (54 pt) margins
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    
    # Custom Corporate Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A252C"), # Desaturated Dark Slate
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2C3E50"),
        spaceBefore=15,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#34495E"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#333333"),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    code_style = ParagraphStyle(
        'CodeText',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#2C3E50")
    )

    story = []

    # Title Block
    story.append(Spacer(1, 10))
    story.append(Paragraph("AUDIT RESOLUTION REPORT & CHANGELOG", title_style))
    story.append(Paragraph("Verification of compliance fixes, input sanitization wrappers, and API exception handling.", body_style))
    story.append(Spacer(1, 10))

    # Repository Metadata Table
    meta_data = [
        [Paragraph("<b>Target System</b>", body_style), Paragraph("IDBI MSME Risk Intelligence Engine", body_style)],
        [Paragraph("<b>Repository</b>", body_style), Paragraph("MSME-BANK-PREDICTION", body_style)],
        [Paragraph("<b>Location</b>", body_style), Paragraph("c:/Users/PARTH/OneDrive/Desktop/project/MSME-BANK-PREDICTION", body_style)],
        [Paragraph("<b>Audit Date</b>", body_style), Paragraph("July 7, 2026", body_style)],
        [Paragraph("<b>Auditors</b>", body_style), Paragraph("External Risk Auditor & Portfolio Bank Manager", body_style)],
        [Paragraph("<b>Compliance Standards</b>", body_style), Paragraph("RBI Fair Practices Code, ECOA Fair Lending Guidelines", body_style)],
    ]
    t_meta = Table(meta_data, colWidths=[150, 354])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8F9FA")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 15))

    # Section 1: Structural Risk Assessment
    story.append(Paragraph("1. STRUCTURAL RISK ASSESSMENT", h1_style))
    story.append(Paragraph("An exhaustive review was performed on the XGBoost ML pipeline features and downstream loss projections:", body_style))
    story.append(Paragraph("• <b>Direct Structural Bias:</b> Checked the 10 leading indicators used in model scoring. All protected classes (Gender, Race, Caste, Age) are completely absent. Features strictly represent credit behavior and leverage ratios.", bullet_style))
    story.append(Paragraph("• <b>Indirect NLP Bias:</b> Evaluated officer note embeddings for demographic triggers. The lexicon keyword-matcher fallback was active, guaranteeing identical risk scores regardless of applicant gender or region. However, downstream monitoring is required if the Sentence-Transformer is activated.", bullet_style))
    story.append(Paragraph("• <b>Sector Capital Allocation:</b> Construction and Manufacturing sectors carry LGD values of 55% and 45% respectively, elevating Expected Loss (EL) by up to 83% over Technology (30% LGD). This has been flagged as a sector-concentration allocation risk.", bullet_style))
    story.append(Spacer(1, 15))

    # Section 2: Resolved Exceptions & Validation Gaps
    story.append(Paragraph("2. RESOLVED RUNTIME EXCEPTIONS", h1_style))
    story.append(Paragraph("The following system-level vulnerabilities were uncovered and fully resolved in the production codebase:", body_style))
    
    exc_data = [
        [Paragraph("<b>Vulnerability</b>", body_style), Paragraph("<b>Root Cause</b>", body_style), Paragraph("<b>Remediation</b>", body_style)],
        [
            Paragraph("API Exception Crash (KeyError)", body_style),
            Paragraph("<code>predict_pd</code> expected complete dictionary keys. Missing features raised a KeyError in pandas indexing.", body_style),
            Paragraph("Reindexed features to the complete <code>FEATURES</code> schema, filling missing keys with <code>NaN</code>.", body_style)
        ],
        [
            Paragraph("Silent Formatting Coercion", body_style),
            Paragraph("Ingestion coersed formatted text (currency symbol, commas, percentages) to <code>NaN</code>, silently replacing them with baseline medians.", body_style),
            Paragraph("Added string cleaning and regex parser wrapper to strip non-numeric tokens prior to conversion.", body_style)
        ],
        [
            Paragraph("Negative Parameter Bypass", body_style),
            Paragraph("Negative values bypassed conversion, feeding downstream models and creating an exploit vector (e.g. -25% utilization scored as 1.34% PD).", body_style),
            Paragraph("Applied domain-specific clipping filters (e.g., utilization clipped to <code>[0.0, 1.0]</code>).", body_style)
        ],
        [
            Paragraph("Silent Overwrite Bug", body_style),
            Paragraph("Ingesting files missing a single parameter triggered full schema derivation, completely overriding all provided features with medians.", body_style),
            Paragraph("Refactored feature derivation to conditionally write missing values, preserving existing valid inputs.", body_style)
        ],
    ]
    t_exc = Table(exc_data, colWidths=[110, 200, 194])
    t_exc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2C3E50")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    # Quick styling adjustment for header row text in table
    for i in range(3):
        exc_data[0][i].style.textColor = colors.white
        
    story.append(t_exc)
    story.append(Spacer(1, 15))

    story.append(PageBreak())

    # Section 3: Code Validation Diff Comparisons
    story.append(Paragraph("3. CODE VALIDATION DIFF COMPARISONS", h1_style))
    story.append(Paragraph("The exact source code modifications implemented in <b>main.py</b> are documented below:", body_style))

    # Diff 1
    story.append(Paragraph("Modification A: KeyError Resolution in predict_pd", h2_style))
    code_diff_a = [
        [Paragraph("<b>Original Buggy Snippet</b>", body_style), Paragraph("<b>Remediated Sanitized Code</b>", body_style)],
        [
            Paragraph("def predict_pd(features: dict) -> float:<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;return float(predict_portfolio(<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pd.DataFrame([features])<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;)[0])", code_style),
            Paragraph("def predict_pd(features: dict) -> float:<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;df = pd.DataFrame([features])<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;for f in FEATURES:<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if f not in df.columns:<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;df[f] = np.nan<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;return float(predict_portfolio(df)[0])", code_style)
        ]
    ]
    t_diff_a = Table(code_diff_a, colWidths=[252, 252])
    t_diff_a.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#ECF0F1")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BDC3C7")),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_diff_a)
    story.append(Spacer(1, 10))

    # Diff 2
    story.append(Paragraph("Modification B: Silent Overwrite Fix in _derive_features", h2_style))
    code_diff_b = [
        [Paragraph("<b>Original Buggy Snippet</b>", body_style), Paragraph("<b>Remediated Sanitized Code</b>", body_style)],
        [
            Paragraph("out = df.copy()<br/>"
                      "out['revolving_utilization'] = np.clip(util, 0.01, 0.99)<br/>"
                      "out['debt_ratio'] = debt<br/>"
                      "out['late_30_59'] = np.clip(emi, 0, 12).astype(int)", code_style),
            Paragraph("out = df.copy()<br/>"
                      "if 'revolving_utilization' not in out.columns or out['revolving_utilization'].isna().all():<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;out['revolving_utilization'] = np.clip(util, 0.01, 0.99)<br/>"
                      "if 'debt_ratio' not in out.columns or out['debt_ratio'].isna().all():<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;out['debt_ratio'] = debt", code_style)
        ]
    ]
    t_diff_b = Table(code_diff_b, colWidths=[252, 252])
    t_diff_b.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#ECF0F1")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BDC3C7")),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_diff_b)
    story.append(Spacer(1, 10))

    # Diff 3
    story.append(Paragraph("Modification C: Input Sanitization Helper clean_numeric_series", h2_style))
    code_diff_c = [
        [Paragraph("<b>Remediated Input Sanitizer Helper Wrapper</b>", body_style)],
        [
            Paragraph("def clean_numeric_series(s: pd.Series) -> pd.Series:<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;if s is None or len(s) == 0:<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return pd.Series(dtype=float)<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;s_str = s.astype(str).str.strip()<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;s_cleaned = s_str.str.replace(r'[\\$\\u20B9\\u20A8\\s,]', '', regex=True)<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;pct_mask = s_cleaned.str.endswith('%', na=False)<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;s_cleaned = s_cleaned.str.rstrip('%')<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;numeric_s = pd.to_numeric(s_cleaned, errors='coerce')<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;if pct_mask.any():<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;numeric_s = numeric_s.mask(pct_mask, numeric_s / 100.0)<br/>"
                      "&nbsp;&nbsp;&nbsp;&nbsp;return numeric_s", code_style)
        ]
    ]
    t_diff_c = Table(code_diff_c, colWidths=[504])
    t_diff_c.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#ECF0F1")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BDC3C7")),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_diff_c)
    story.append(Spacer(1, 15))

    # Section 4: Audit Verification Summary
    story.append(Paragraph("4. VERIFICATION AUDIT RESULTS", h1_style))
    verification_data = [
        [Paragraph("<b>Test Name</b>", body_style), Paragraph("<b>Injected Input</b>", body_style), Paragraph("<b>Expected Behavior</b>", body_style), Paragraph("<b>Auditor Verification Status</b>", body_style)],
        [Paragraph("Formatting Quirks", body_style), Paragraph("<code>outstanding_loan='$1,500,000'</code><br/><code>utilization='45%'</code>", body_style), Paragraph("Cleans symbols, parses numerical float without data loss", body_style), Paragraph("<b>PASSED</b> (Sanitized correctly)", body_style)],
        [Paragraph("Negative Values", body_style), Paragraph("<code>revolving_utilization=-0.25</code>", body_style), Paragraph("Clips value to positive bounds (0.0)", body_style), Paragraph("<b>PASSED</b> (Exploit vector blocked)", body_style)],
        [Paragraph("Integer Overflow", body_style), Paragraph("Outstanding Loan = 30-digit integer", body_style), Paragraph("Caps loan to Rs. 1,000,000,000,000 (1e12)", body_style), Paragraph("<b>PASSED</b> (Buffer protected)", body_style)],
        [Paragraph("Silent Overwrite Fix", body_style), Paragraph("Incomplete feature upload row", body_style), Paragraph("Preserves existing provided fields, derives only missing", body_style), Paragraph("<b>PASSED</b> (Original features kept)", body_style)],
    ]
    t_ver = Table(verification_data, colWidths=[100, 140, 164, 100])
    t_ver.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#34495E")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BDC3C7")),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    for i in range(4):
        verification_data[0][i].style.textColor = colors.white
    story.append(t_ver)

    doc.build(story, canvasmaker=NumberedCanvas)


if __name__ == "__main__":
    pdf_name = "antigravity_changelog.pdf"
    build_pdf(pdf_name)
    print(f"✓ PDF report successfully generated: {pdf_name}")
