"""
exporter.py — export Excel e PDF per portafogli.
Supporta: grafico frontiera efficiente + stelle FIDA nel PDF.
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph,
        Spacer, Image, HRFlowable, KeepTogether,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

NAVY      = "#1A2C54"
NAVY_RL   = colors.HexColor(NAVY)
GRAY_LIGHT = colors.HexColor("#F5F7FA")
GRAY_MID   = colors.HexColor("#E2E8F0")


# ---------------------------------------------------------------------------
# HELPER: converti figura Plotly → PNG bytes
# ---------------------------------------------------------------------------

def plotly_to_png(fig, width: int = 1100, height: int = 550) -> Optional[bytes]:
    """Converte un oggetto go.Figure in bytes PNG via kaleido."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        pass
    try:
        import plotly.io as pio
        return pio.to_image(fig, format="png", width=width, height=height, scale=2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HELPER: stelle FIDA come stringa
# ---------------------------------------------------------------------------

def _stars(n) -> str:
    if n is None:
        return "—"
    try:
        n = int(float(str(n)))
        return "★" * n + "☆" * (5 - n) if 1 <= n <= 5 else "—"
    except Exception:
        return "—"


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
    """Genera xlsx in memoria con pesi, metriche, correlazioni, serie storiche."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Foglio 1: Pesi
        weights_df = pd.DataFrame([
            {"ISIN": isin, "Peso (%)": round(w * 100, 2)}
            for isin, w in weights.items() if w > 0
        ])
        if fund_df is not None and not fund_df.empty:
            try:
                _cols = [c for c in ["isin","nome","classificazione","casa",
                                      "perf_1y","perf_3y","volatilita",
                                      "rating_fida","retrocessione"] if c in fund_df.columns]
                fund_info = fund_df[_cols].copy()
                fund_info.columns = [c.replace("_"," ").title() for c in _cols]
                fund_info = fund_info.rename(columns={"Isin": "ISIN"})
                weights_df = weights_df.merge(fund_info, on="ISIN", how="left")
            except Exception:
                pass
        weights_df.to_excel(writer, sheet_name="Pesi", index=False)

        # Foglio 2: Metriche
        metrics_df = pd.DataFrame([{"Metrica": k, "Valore": v} for k, v in metrics.items()])
        metrics_df.to_excel(writer, sheet_name="Metriche", index=False)

        # Foglio 3: Correlazioni
        if corr_df is not None and not corr_df.empty:
            corr_df.to_excel(writer, sheet_name="Correlazioni")

        # Foglio 4: Serie storiche
        if price_dict:
            try:
                prices_df = pd.DataFrame(price_dict)
                prices_df.index.name = "Data"
                prices_df.to_excel(writer, sheet_name="Serie Storiche")
            except Exception:
                pass

        # Styling header navy
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
    chart_bytes: Optional[bytes] = None,          # PNG del grafico frontiera
    frontier_fig=None,                             # go.Figure → converti al volo
    fund_df: Optional[pd.DataFrame] = None,       # per stelle FIDA + nomi
    fund_pool: Optional[dict] = None,             # pool ISIN→info dall'app
    include_stars: bool = True,
) -> bytes:
    """
    Genera PDF in memoria.
    Ritorna bytes pronti per st.download_button.
    """
    if not HAS_REPORTLAB:
        return b""

    # Converti figura Plotly se non già disponibile come bytes
    if chart_bytes is None and frontier_fig is not None:
        chart_bytes = plotly_to_png(frontier_fig)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Heading1"],
        fontSize=20, textColor=NAVY_RL,
        spaceAfter=4, spaceBefore=0,
    )
    sub_style = ParagraphStyle(
        "SubCustom", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#666666"),
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "HeadingCustom", parent=styles["Heading2"],
        fontSize=12, textColor=NAVY_RL,
        spaceBefore=14, spaceAfter=5,
        borderPad=0,
    )
    small = ParagraphStyle(
        "Small", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#555555"),
        spaceAfter=4,
    )
    normal = styles["Normal"]

    story = []

    # ── Intestazione ─────────────────────────────────────────────────────────
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(
        f"Generato il {datetime.now().strftime('%d/%m/%Y alle %H:%M')}  ·  "
        "Portafogli Efficienti",
        sub_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY_RL, spaceAfter=8))

    # ── Grafico Frontiera Efficiente ─────────────────────────────────────────
    if chart_bytes:
        story.append(Paragraph("Frontiera Efficiente", heading_style))
        try:
            img_buf = io.BytesIO(chart_bytes)
            # Larghezza piena A4 meno margini ≈ 17.4cm
            img = Image(img_buf, width=17.4*cm, height=9.5*cm)
            story.append(img)
            story.append(Paragraph(
                "Il grafico mostra la frontiera efficiente (curva blu) e i portafogli ottimali. "
                "Max Sharpe (stella rossa) massimizza il rendimento per unità di rischio. "
                "Min Varianza (diamante blu) minimizza la volatilità.",
                small,
            ))
        except Exception as _e:
            story.append(Paragraph(f"Grafico non disponibile ({_e})", small))
        story.append(Spacer(1, 0.4*cm))

    # ── Metriche portafoglio ─────────────────────────────────────────────────
    story.append(Paragraph("Metriche di Portafoglio", heading_style))
    m_data = [["Metrica", "Valore"]]
    for k, v in metrics.items():
        m_data.append([str(k), str(v)])
    mt = Table(m_data, colWidths=[9*cm, 7*cm])
    mt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  NAVY_RL),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, GRAY_LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.4, GRAY_MID),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(KeepTogether([mt]))
    story.append(Spacer(1, 0.4*cm))

    # ── Composizione portafoglio con nomi e stelle FIDA ──────────────────────
    story.append(Paragraph("Composizione Portafoglio", heading_style))

    # Costruisce dizionario isin→info da tutte le sorgenti disponibili
    _info_map: dict = {}
    if fund_pool:
        _info_map.update(fund_pool)
    if fund_df is not None and not fund_df.empty and "isin" in fund_df.columns:
        for _, r in fund_df.iterrows():
            isin = str(r.get("isin",""))
            if isin and isin not in _info_map:
                _info_map[isin] = r.to_dict()

    has_stars = include_stars and any(
        _info_map.get(isin, {}).get("rating_fida") is not None
        for isin in weights if weights[isin] > 0
    )

    # Header tabella
    if has_stars:
        hdr = ["ISIN", "Nome / Fondo", "Classificazione", "Peso %", "★ FIDA", "Perf 3Y %"]
        col_w = [2.8*cm, 6.5*cm, 3.5*cm, 1.6*cm, 1.6*cm, 1.8*cm]
    else:
        hdr = ["ISIN", "Nome / Fondo", "Classificazione", "Peso %", "Perf 3Y %"]
        col_w = [2.8*cm, 7.5*cm, 4.0*cm, 1.8*cm, 2.0*cm]

    w_data = [hdr]
    for isin, w in sorted(weights.items(), key=lambda x: -x[1]):
        if w <= 0:
            continue
        info = _info_map.get(isin, {})
        nome = str(info.get("nome", info.get("FONDO AZIMUT", isin)))[:55]
        cl   = str(info.get("classificazione", info.get("categoria","—")))[:30]
        perf = info.get("perf_3y")
        perf_str = f"{float(perf):.1f}%" if perf is not None else "—"
        row = [isin, nome, cl, f"{w*100:.1f}%", perf_str]
        if has_stars:
            stars = _stars(info.get("rating_fida"))
            row = [isin, nome, cl, f"{w*100:.1f}%", stars, perf_str]
        w_data.append(row)

    wt = Table(w_data, colWidths=col_w, repeatRows=1)
    _ts = [
        ("BACKGROUND",    (0,0), (-1,0),  NAVY_RL),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, GRAY_LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.35, GRAY_MID),
        ("LEFTPADDING",   (0,0), (-1,-1), 5),
        ("RIGHTPADDING",  (0,0), (-1,-1), 5),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("ALIGN",         (3,0), (-1,-1), "CENTER"),
    ]
    # Colori celle stelle FIDA
    if has_stars:
        star_col_idx = 4
        for row_i in range(1, len(w_data)):
            s_str = w_data[row_i][star_col_idx]
            if "★★★★★" in s_str:
                _ts.append(("TEXTCOLOR",(star_col_idx,row_i),(star_col_idx,row_i), colors.HexColor("#D97706")))
            elif "★★★★" in s_str:
                _ts.append(("TEXTCOLOR",(star_col_idx,row_i),(star_col_idx,row_i), colors.HexColor("#B45309")))
            elif "★★★" in s_str:
                _ts.append(("TEXTCOLOR",(star_col_idx,row_i),(star_col_idx,row_i), colors.HexColor("#92400E")))

    wt.setStyle(TableStyle(_ts))
    story.append(wt)

    if has_stars:
        story.append(Paragraph(
            "★ Rating FIDA: ★★★★★ = 5 stelle (top), ★★★ = 3 stelle (nella media)",
            small,
        ))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_MID))
    story.append(Paragraph(
        "Documento generato da Portafogli Efficienti — solo uso interno. "
        "Non costituisce consulenza finanziaria.",
        small,
    ))

    doc.build(story)
    return buf.getvalue()
