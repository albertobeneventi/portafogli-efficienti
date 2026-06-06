"""
exporter.py — export Excel e PDF per portafogli.
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph,
        Spacer, Image,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

NAVY = "#1A2C54"
LIGHT_GRAY = "#F5F7FA"


# ---------------------------------------------------------------------------
# EXCEL EXPORT
# ---------------------------------------------------------------------------

def export_portfolio_excel(
    weights: dict,
    metrics: dict,
    fund_df: Optional[pd.DataFrame] = None,
    corr_df: Optional[pd.DataFrame] = None,
    price_dict: Optional[dict] = None,
    title: str = "Portafoglio",
) -> bytes:
    """
    Genera file xlsx in memoria.
    Ritorna bytes pronti per st.download_button.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Foglio 1: Pesi
        weights_df = pd.DataFrame([
            {"ISIN": isin, "Peso (%)": round(w * 100, 2)}
            for isin, w in weights.items() if w > 0
        ])
        if fund_df is not None:
            fund_info = fund_df[["isin", "nome", "classificazione",
                                  "perf_1y", "perf_3y", "volatilita",
                                  "rating_fida", "retrocessione"]].copy()
            fund_info.columns = ["ISIN", "Nome", "Classificazione",
                                  "Perf 1Y (%)", "Perf 3Y (%)", "Volatilità (%)",
                                  "Rating FIDA", "Retrocessione (%)"]
            weights_df = weights_df.merge(fund_info, on="ISIN", how="left")
        weights_df.to_excel(writer, sheet_name="Pesi", index=False)

        # Foglio 2: Metriche
        metrics_df = pd.DataFrame([{
            "Metrica": k, "Valore": v
        } for k, v in metrics.items()])
        metrics_df.to_excel(writer, sheet_name="Metriche", index=False)

        # Foglio 3: Correlazioni
        if corr_df is not None:
            corr_df.to_excel(writer, sheet_name="Correlazioni")

        # Foglio 4: Serie storiche
        if price_dict:
            prices_df = pd.DataFrame(price_dict)
            prices_df.index.name = "Data"
            prices_df.to_excel(writer, sheet_name="Serie Storiche")

        # Styling base
        wb = writer.book
        from openpyxl.styles import Font, PatternFill, Alignment
        navy_fill = PatternFill(start_color="1A2C54", end_color="1A2C54", fill_type="solid")
        white_font = Font(color="FFFFFF", bold=True)
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for cell in ws[1]:
                cell.fill = navy_fill
                cell.font = white_font
                cell.alignment = Alignment(horizontal="center")
            for col in ws.columns:
                max_len = max((len(str(cell.value or "")) for cell in col), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# PDF EXPORT
# ---------------------------------------------------------------------------

def export_portfolio_pdf(
    weights: dict,
    metrics: dict,
    title: str = "Report Portafoglio",
    chart_bytes: Optional[bytes] = None,
) -> bytes:
    """
    Genera PDF in memoria.
    Ritorna bytes pronti per st.download_button.
    """
    if not HAS_REPORTLAB:
        return b""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        fontSize=18, textColor=colors.HexColor(NAVY),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor(NAVY),
        spaceBefore=12, spaceAfter=6,
    )
    normal = styles["Normal"]

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(f"Generato il {datetime.now().strftime('%d/%m/%Y %H:%M')}", normal))
    story.append(Spacer(1, 0.5 * cm))

    # Grafico (se fornito)
    if chart_bytes:
        img_buf = io.BytesIO(chart_bytes)
        img = Image(img_buf, width=16 * cm, height=9 * cm)
        story.append(img)
        story.append(Spacer(1, 0.5 * cm))

    # Tabella pesi
    story.append(Paragraph("Composizione Portafoglio", heading_style))
    table_data = [["ISIN", "Peso (%)"]]
    for isin, w in sorted(weights.items(), key=lambda x: -x[1]):
        if w > 0:
            table_data.append([isin, f"{w * 100:.2f}%"])
    t = Table(table_data, colWidths=[8 * cm, 4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    # Tabella metriche
    story.append(Paragraph("Metriche di Portafoglio", heading_style))
    metrics_data = [["Metrica", "Valore"]]
    for k, v in metrics.items():
        metrics_data.append([str(k), str(v)])
    mt = Table(metrics_data, colWidths=[8 * cm, 4 * cm])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
    ]))
    story.append(mt)

    doc.build(story)
    return buf.getvalue()
