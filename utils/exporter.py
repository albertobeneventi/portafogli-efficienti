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
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import os as _os

    # Registra font Unicode — supporta stelle e accenti italiani
    _FONT_NAME = "Helvetica"
    _FONT_BOLD = "Helvetica-Bold"
    _unicode_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      # Linux (Streamlit Cloud)
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/calibri.ttf",                          # Windows
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",                   # macOS
    ]
    for _fp in _unicode_paths:
        if _os.path.exists(_fp):
            try:
                pdfmetrics.registerFont(TTFont("UniFont", _fp))
                _FONT_NAME = "UniFont"
                _FONT_BOLD = "UniFont"
                break
            except Exception:
                pass

    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    _FONT_NAME = "Helvetica"
    _FONT_BOLD = "Helvetica-Bold"

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

NAVY     = "#1A2C54"
NAVY_RL  = colors.HexColor(NAVY) if HAS_REPORTLAB else None
GRAY_LIGHT = colors.HexColor("#F5F7FA") if HAS_REPORTLAB else None
GRAY_MID   = colors.HexColor("#E2E8F0") if HAS_REPORTLAB else None


# ---------------------------------------------------------------------------
# HELPER: converti figura Plotly in PNG bytes
# ---------------------------------------------------------------------------

def plotly_to_png(fig, width: int = 1100, height: int = 550) -> Optional[bytes]:
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
# HELPER: pulizia testo per PDF Helvetica (non-Unicode)
# ---------------------------------------------------------------------------

def _safe_str(text: str) -> str:
    """Converte stringa in formato sicuro per Helvetica (ASCII + Latin-1)."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    if _FONT_NAME != "Helvetica":
        return text          # font Unicode → nessuna conversione
    _MAP = {
        "★": "*", "☆": "o",   # ★ ☆
        "€": "EUR",                 # €
        "–": "-", "—": "-",   # en-dash, em-dash
        "‘": "'", "’": "'",   # curly quotes
        "“": '"', "”": '"',
        "\xe0": "a", "\xe8": "e", "\xe9": "e",
        "\xec": "i", "\xf2": "o", "\xf9": "u",
        "\xc0": "A", "\xc8": "E", "\xc9": "E",
        "\xcc": "I", "\xd2": "O", "\xd9": "U",
        "\xe4": "a", "\xf6": "o", "\xfc": "u",
    }
    for src, dst in _MAP.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ---------------------------------------------------------------------------
# HELPER: stelle FIDA
# ---------------------------------------------------------------------------

def _stars(n) -> str:
    if n is None:
        return "-"
    try:
        n = int(float(str(n)))
        if not (1 <= n <= 5):
            return "-"
        s = "*" if _FONT_NAME == "Helvetica" else "★"   # ★
        e = "o" if _FONT_NAME == "Helvetica" else "☆"   # ☆
        return s * n + e * (5 - n)
    except Exception:
        return "-"


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
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
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

        metrics_df = pd.DataFrame([{"Metrica": k, "Valore": v} for k, v in metrics.items()])
        metrics_df.to_excel(writer, sheet_name="Metriche", index=False)

        if corr_df is not None and not corr_df.empty:
            corr_df.to_excel(writer, sheet_name="Correlazioni")

        if price_dict:
            try:
                prices_df = pd.DataFrame(price_dict)
                prices_df.index.name = "Data"
                prices_df.to_excel(writer, sheet_name="Serie Storiche")
            except Exception:
                pass

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
    frontier_fig=None,
    fund_df: Optional[pd.DataFrame] = None,
    fund_pool: Optional[dict] = None,
    include_stars: bool = True,
) -> bytes:
    if not HAS_REPORTLAB:
        return b""

    if chart_bytes is None and frontier_fig is not None:
        chart_bytes = plotly_to_png(frontier_fig)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()

    def _p(style_name, **kw):
        return ParagraphStyle(style_name, parent=styles["Normal"],
                              fontName=_FONT_NAME, **kw)

    title_style   = _p("T",  fontSize=20, textColor=NAVY_RL, spaceAfter=4)
    sub_style     = _p("S",  fontSize=9,  textColor=colors.HexColor("#666666"), spaceAfter=10)
    heading_style = _p("H",  fontSize=12, textColor=NAVY_RL, spaceBefore=14, spaceAfter=5)
    small         = _p("Sm", fontSize=8,  textColor=colors.HexColor("#555555"), spaceAfter=4)

    story = []

    # ── Intestazione ──────────────────────────────────────────────────────────
    story.append(Paragraph(_safe_str(title), title_style))
    story.append(Paragraph(
        _safe_str(f"Generato il {datetime.now().strftime('%d/%m/%Y alle %H:%M')}  -  Portafogli Efficienti"),
        sub_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY_RL, spaceAfter=8))

    # ── Grafico Frontiera Efficiente ──────────────────────────────────────────
    if chart_bytes:
        story.append(Paragraph("Frontiera Efficiente", heading_style))
        try:
            img = Image(io.BytesIO(chart_bytes), width=17.4*cm, height=9.5*cm)
            story.append(img)
            story.append(Paragraph(
                _safe_str("Curva blu = frontiera efficiente. Stella rossa = Max Sharpe. "
                          "Diamante blu = Min Varianza."),
                small,
            ))
        except Exception as _e:
            story.append(Paragraph(f"Grafico non disponibile ({_e})", small))
        story.append(Spacer(1, 0.4*cm))

    # ── Metriche ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Metriche di Portafoglio", heading_style))
    m_data = [["Metrica", "Valore"]]
    for k, v in metrics.items():
        m_data.append([_safe_str(str(k)), _safe_str(str(v))])
    mt = Table(m_data, colWidths=[9*cm, 7*cm])
    mt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), NAVY_RL),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTNAME",      (0,0),(-1,-1), _FONT_NAME),
        ("FONTNAME",      (0,0),(-1,0),  _FONT_BOLD),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, GRAY_LIGHT]),
        ("GRID",          (0,0),(-1,-1), 0.4, GRAY_MID),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ]))
    story.append(KeepTogether([mt]))
    story.append(Spacer(1, 0.4*cm))

    # ── Composizione portafoglio ──────────────────────────────────────────────
    story.append(Paragraph("Composizione Portafoglio", heading_style))

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
        for isin in weights if weights.get(isin, 0) > 0
    )

    star_hdr = "Rating" if _FONT_NAME == "Helvetica" else "★ FIDA"
    if has_stars:
        hdr    = ["ISIN", "Nome / Fondo", "Classificazione", "Peso %", star_hdr, "Perf 3Y %"]
        col_w  = [2.8*cm, 6.2*cm, 3.5*cm, 1.6*cm, 1.6*cm, 1.7*cm]
    else:
        hdr    = ["ISIN", "Nome / Fondo", "Classificazione", "Peso %", "Perf 3Y %"]
        col_w  = [2.8*cm, 7.5*cm, 4.0*cm, 1.8*cm, 2.0*cm]

    w_data = [hdr]
    for isin, w in sorted(weights.items(), key=lambda x: -x[1]):
        if w <= 0:
            continue
        info = _info_map.get(isin, {})
        nome = _safe_str(str(info.get("nome", info.get("FONDO AZIMUT", isin)))[:55])
        cl   = _safe_str(str(info.get("classificazione", info.get("categoria", "-")))[:30])
        perf = info.get("perf_3y")
        perf_str = f"{float(perf):.1f}%" if perf is not None else "-"
        if has_stars:
            w_data.append([isin, nome, cl, f"{w*100:.1f}%",
                           _stars(info.get("rating_fida")), perf_str])
        else:
            w_data.append([isin, nome, cl, f"{w*100:.1f}%", perf_str])

    _ts = [
        ("BACKGROUND",    (0,0),(-1,0), NAVY_RL),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTNAME",      (0,0),(-1,-1), _FONT_NAME),
        ("FONTNAME",      (0,0),(-1,0),  _FONT_BOLD),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, GRAY_LIGHT]),
        ("GRID",          (0,0),(-1,-1), 0.35, GRAY_MID),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ("TOPPADDING",    (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
        ("ALIGN",         (3,0),(-1,-1), "CENTER"),
    ]
    wt = Table(w_data, colWidths=col_w, repeatRows=1)
    wt.setStyle(TableStyle(_ts))
    story.append(wt)

    if has_stars:
        legend = ("Rating FIDA: ***** = 5 stelle (top), *** = 3 stelle"
                  if _FONT_NAME == "Helvetica"
                  else "Rating FIDA: ★★★★★ = 5 stelle (top), ★★★ = 3 stelle")
        story.append(Paragraph(legend, small))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_MID))
    story.append(Paragraph(
        _safe_str("Documento generato da Portafogli Efficienti - solo uso interno. "
                  "Non costituisce consulenza finanziaria."),
        small,
    ))

    doc.build(story)
    return buf.getvalue()
