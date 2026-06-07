"""
market_views_loader.py — Lettura file "View di Mercato" fornito dall'utente.

Formato atteso: file Excel (.xlsx) con fino a 3 fogli:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Foglio 1 — "Pesi"  (Allocazione target per sottocategoria / macro)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Colonne obbligatorie:
  Sottocategoria   : nome libero (es. "Azionari Europa", "High Yield",
                     "Obbligazionari Governativi", "Materie Prime")
  Peso %           : peso target (0-100, somma non deve essere 100 — verrà
                     usato come suggerimento di range per i vincoli)
Colonne opzionali:
  Min %            : peso minimo (default = max(0, Peso-5))
  Max %            : peso massimo (default = Peso+5)
  Segnale          : "+", "=", "-"  (sovrappeso / neutro / sottopeso)
  Note             : testo libero, mostrato nell'app

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Foglio 2 — "View"  (View specifiche per Black-Litterman)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Colonne obbligatorie:
  ISIN o Categoria : ISIN specifico (es. "IE00B4L5Y983") oppure nome
                     sottocategoria (es. "Azionari Emergenti") — in quel
                     caso la view viene applicata a tutti i fondi della
                     sottocategoria
  Rendimento %     : rendimento atteso annuo (es. 9.5)
  Confidenza       : da 0.1 a 1.0 (es. 0.7)
Colonne opzionali:
  Note             : motivazione della view

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Foglio 3 — "Preferiti"  (Asset da forzare in portafoglio)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Colonne obbligatorie:
  ISIN             : ISIN dell'asset
Colonne opzionali:
  Nome             : nome descrittivo
  Note             : motivazione

Tutti i nomi di foglio e colonna sono case-insensitive e tolleranti
agli spazi (vengono normalizzati prima del parsing).
"""

from __future__ import annotations
import io
import re
import numpy as np
import pandas as pd
from typing import Any


# ---------------------------------------------------------------------------
# STRUTTURA DEL RISULTATO
# ---------------------------------------------------------------------------

class MarketViews:
    """
    Contenitore delle view di mercato caricate dal file.

    Attributi:
      weights     : dict[str, dict]  — {sottocategoria: {peso, min, max, segnale, note}}
      bl_views    : dict[str, float] — {isin_o_categ: rendimento_%}
      bl_confs    : dict[str, float] — {isin_o_categ: confidenza 0-1}
      bl_notes    : dict[str, str]   — {isin_o_categ: nota}
      preferiti   : list[str]        — ISIN da forzare
      raw_weights : pd.DataFrame
      raw_views   : pd.DataFrame
      raw_prefs   : pd.DataFrame
      errors      : list[str]        — avvisi non bloccanti
    """
    def __init__(self):
        self.weights:     dict[str, dict]  = {}
        self.bl_views:    dict[str, float] = {}
        self.bl_confs:    dict[str, float] = {}
        self.bl_notes:    dict[str, str]   = {}
        self.preferiti:   list[str]        = []
        self.raw_weights  = pd.DataFrame()
        self.raw_views    = pd.DataFrame()
        self.raw_prefs    = pd.DataFrame()
        self.errors:      list[str]        = []

    def is_empty(self) -> bool:
        return (not self.weights and not self.bl_views and not self.preferiti)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _norm_col(df: pd.DataFrame) -> dict[str, str]:
    """Mappa nome colonna normalizzato → nome originale."""
    return {re.sub(r"\s+", " ", c).strip().lower(): c for c in df.columns}


def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Trova il nome della prima colonna che matcha uno dei candidati (normalizzati)."""
    nc = _norm_col(df)
    for cand in candidates:
        key = cand.strip().lower()
        if key in nc:
            return nc[key]
    return None


def _safe_float(v, default=None) -> float | None:
    try:
        f = float(str(v).replace(",", ".").replace("%", "").strip())
        return f if not np.isnan(f) else default
    except (ValueError, TypeError):
        return default


def _norm_sheet_name(name: str) -> str:
    return re.sub(r"\s+", "", name).lower()


# ---------------------------------------------------------------------------
# PARSER FOGLIO "PESI"
# ---------------------------------------------------------------------------

def _parse_weights(df: pd.DataFrame, errors: list) -> dict[str, dict]:
    """Legge il foglio Pesi e restituisce {sottocategoria: {peso, min, max, segnale, note}}."""
    result: dict[str, dict] = {}

    col_cat  = _find_col(df, "sottocategoria", "categoria", "asset class",
                          "asset", "subcategory", "categoria asset")
    col_peso = _find_col(df, "peso %", "peso", "weight %", "weight",
                          "target %", "target", "allocazione %", "allocazione")
    if not col_cat or not col_peso:
        errors.append("Foglio 'Pesi': colonne 'Sottocategoria' e 'Peso %' obbligatorie.")
        return result

    col_min  = _find_col(df, "min %", "min", "minimo %", "minimo")
    col_max  = _find_col(df, "max %", "max", "massimo %", "massimo")
    col_seg  = _find_col(df, "segnale", "signal", "view", "bias")
    col_note = _find_col(df, "note", "notes", "commento")

    for _, row in df.iterrows():
        cat  = str(row.get(col_cat, "") or "").strip()
        peso = _safe_float(row.get(col_peso))
        if not cat or peso is None:
            continue

        # Default min/max: ±5 punti percentuali intorno al target
        min_v  = _safe_float(row.get(col_min))  if col_min  else None
        max_v  = _safe_float(row.get(col_max))  if col_max  else None
        seg    = str(row.get(col_seg, "") or "").strip() if col_seg else ""
        note   = str(row.get(col_note, "") or "").strip() if col_note else ""

        if min_v is None: min_v = max(0.0, peso - 5.0)
        if max_v is None: max_v = peso + 5.0

        result[cat] = {
            "peso":    round(peso, 1),
            "min":     round(min_v, 1),
            "max":     round(max_v, 1),
            "segnale": seg,
            "note":    note,
        }
    return result


# ---------------------------------------------------------------------------
# PARSER FOGLIO "VIEW"
# ---------------------------------------------------------------------------

def _parse_views(df: pd.DataFrame, errors: list
                 ) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """Legge il foglio View. Restituisce (bl_views, bl_confs, bl_notes)."""
    views: dict[str, float] = {}
    confs: dict[str, float] = {}
    notes: dict[str, str]   = {}

    col_key  = _find_col(df, "isin o categoria", "isin", "categoria",
                          "asset", "fondo", "sottocategoria", "key")
    col_ret  = _find_col(df, "rendimento %", "rendimento", "return %",
                          "return", "expected return", "rendimento atteso %",
                          "rendimento atteso")
    col_conf = _find_col(df, "confidenza", "confidence", "conf", "peso view")
    col_note = _find_col(df, "note", "notes", "commento", "motivazione")

    if not col_key or not col_ret:
        errors.append("Foglio 'View': colonne 'ISIN o Categoria' e 'Rendimento %' obbligatorie.")
        return views, confs, notes

    for _, row in df.iterrows():
        key = str(row.get(col_key, "") or "").strip()
        ret = _safe_float(row.get(col_ret))
        if not key or ret is None:
            continue
        conf = _safe_float(row.get(col_conf)) if col_conf else 0.5
        if conf is None or not (0 < conf <= 1):
            conf = 0.5
        note = str(row.get(col_note, "") or "").strip() if col_note else ""

        views[key] = round(ret, 2)
        confs[key] = round(conf, 2)
        notes[key] = note

    return views, confs, notes


# ---------------------------------------------------------------------------
# PARSER FOGLIO "PREFERITI"
# ---------------------------------------------------------------------------

def _parse_preferiti(df: pd.DataFrame, errors: list) -> list[str]:
    col_isin = _find_col(df, "isin", "ticker", "codice")
    if not col_isin:
        errors.append("Foglio 'Preferiti': colonna 'ISIN' obbligatoria.")
        return []
    return [
        str(row[col_isin]).strip()
        for _, row in df.iterrows()
        if str(row.get(col_isin, "") or "").strip()
    ]


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def load_market_views(file_source: Any) -> MarketViews:
    """
    Carica le view di mercato da un file Excel (.xlsx).

    file_source: path stringa, bytes, o file-like object (UploadedFile Streamlit).
    Ritorna un oggetto MarketViews (vuoto se il file non è valido).
    """
    mv = MarketViews()

    try:
        if hasattr(file_source, "read"):
            data = io.BytesIO(file_source.read())
        elif isinstance(file_source, (bytes, bytearray)):
            data = io.BytesIO(file_source)
        else:
            data = file_source   # path stringa

        xl = pd.ExcelFile(data)
    except Exception as e:
        mv.errors.append(f"Impossibile aprire il file: {e}")
        return mv

    sheet_map: dict[str, str] = {
        _norm_sheet_name(s): s for s in xl.sheet_names
    }

    # ── Foglio Pesi ──────────────────────────────────────────────────────
    _pesi_keys = ["pesi", "peso", "allocazione", "macro", "weights", "allocation"]
    pesi_sheet = next((sheet_map[k] for k in _pesi_keys if k in sheet_map), None)
    if pesi_sheet:
        try:
            df_p = xl.parse(pesi_sheet, header=0)
            df_p.columns = [str(c).strip() for c in df_p.columns]
            df_p = df_p.dropna(how="all")
            mv.raw_weights = df_p
            mv.weights = _parse_weights(df_p, mv.errors)
        except Exception as e:
            mv.errors.append(f"Errore lettura foglio Pesi: {e}")
    else:
        mv.errors.append("Foglio 'Pesi' non trovato — nomi accettati: Pesi, Peso, Allocazione, Macro.")

    # ── Foglio View ───────────────────────────────────────────────────────
    _view_keys = ["view", "views", "rendimenti", "bl", "black-litterman",
                  "rendimentiattesi", "viewmercato", "viewdimercato"]
    view_sheet = next((sheet_map[k] for k in _view_keys if k in sheet_map), None)
    if view_sheet:
        try:
            df_v = xl.parse(view_sheet, header=0)
            df_v.columns = [str(c).strip() for c in df_v.columns]
            df_v = df_v.dropna(how="all")
            mv.raw_views = df_v
            mv.bl_views, mv.bl_confs, mv.bl_notes = _parse_views(df_v, mv.errors)
        except Exception as e:
            mv.errors.append(f"Errore lettura foglio View: {e}")
    else:
        mv.errors.append("Foglio 'View' non trovato — nomi accettati: View, Rendimenti, BL.")

    # ── Foglio Preferiti ──────────────────────────────────────────────────
    _pref_keys = ["preferiti", "preferito", "forzati", "forced", "forceinc",
                  "force", "include"]
    pref_sheet = next((sheet_map[k] for k in _pref_keys if k in sheet_map), None)
    if pref_sheet:
        try:
            df_pr = xl.parse(pref_sheet, header=0)
            df_pr.columns = [str(c).strip() for c in df_pr.columns]
            df_pr = df_pr.dropna(how="all")
            mv.raw_prefs = df_pr
            mv.preferiti = _parse_preferiti(df_pr, mv.errors)
        except Exception as e:
            mv.errors.append(f"Errore lettura foglio Preferiti: {e}")

    return mv


# ---------------------------------------------------------------------------
# HELPER: espande view per categoria → ISINs presenti nel pool
# ---------------------------------------------------------------------------

def expand_category_views(
    mv: MarketViews,
    all_fund_pool: dict[str, dict],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Espande le view che sono sottocategorie (es. "Azionari Emergenti")
    a tutti gli ISIN del pool che hanno quella classificazione.
    Le view su ISIN specifici restano invariate.

    Ritorna (views_dict, confs_dict) pronti per compute_black_litterman.
    """
    expanded_views: dict[str, float] = {}
    expanded_confs: dict[str, float] = {}

    for key, ret in mv.bl_views.items():
        conf = mv.bl_confs.get(key, 0.5)

        # ISIN valido (12 caratteri alfanumerici)?
        if re.match(r"^[A-Z]{2}[A-Z0-9]{10}$", key.upper()):
            expanded_views[key.upper()] = ret
            expanded_confs[key.upper()] = conf
        else:
            # Categoria: applica a tutti i fondi con quella classificazione
            key_low = key.lower()
            matched = 0
            for isin, info in all_fund_pool.items():
                cl = str(info.get("classificazione", "") or "").lower()
                if key_low in cl or cl in key_low:
                    expanded_views[isin] = ret
                    expanded_confs[isin] = conf
                    matched += 1
            if matched == 0:
                # Nessun match esatto: cerca per parole chiave
                kws = key_low.split()
                for isin, info in all_fund_pool.items():
                    cl = str(info.get("classificazione", "") or "").lower()
                    if all(kw in cl for kw in kws):
                        expanded_views[isin] = ret
                        expanded_confs[isin] = conf

    return expanded_views, expanded_confs
