"""
data_loader.py — carica e normalizza i file Excel fondi terzi e Azimut.
"""

import os
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# COSTANTI NOMI COLONNE
# ---------------------------------------------------------------------------

TERZI_COLS = {
    "isin": "ISIN",
    "nome": "DESCRIZIONE FONDO",
    "casa": "ASSET MANAGEMENT HOUSE",
    "classificazione": "CLASSIFICAZIONE (FIDA)",
    "commissioni": "COMMISSIONI DI GESTIONE",
    "retrocessione": "RETROCESSIONE BANCA",
    "perf_1y": "PERF. 1 ANNO",
    "perf_3y": "PERF. 3 ANNI",
    "perf_ytd": "PERF. YTD",
    "perf_2024": "PERF. 2024",
    "perf_2023": "PERF. 2023",
    "perf_2022": "PERF. 2022",
    "volatilita": "VOLATILITA' (1 anno)",
    "rating_fida": "RATING (stelle FIDA)",
    "acc_distr": "ACC./DISTR.",
}

AZIMUT_COLS = {
    "nome": "FONDO AZIMUT",
    "isin": "ISIN",
    "share_class": "SHARE CLASS",
    "valuta": "VALUTA",
    "acc_dis": "ACC/DIS",
    "hedged": "HEDGED",
    "stelle_fida": "STELLE FIDA",
    "stelle_ms": "STELLE MORNINGSTAR",
    "classificazione": "CLASSIFICAZIONE FIDA",
    "cat_ms": "CAT. MORNINGSTAR",
    "sottocategoria": "SOTTOCATEGORIA",
    "perf_ytd": "PERF YTD",
    "perf_1y": "PERF 1Y",
    "perf_3y": "PERF 3Y",
    "perf_5y": "PERF 5Y",
    "perf_2024": "PERF 2024",
    "perf_2023": "PERF 2023",
    "perf_2022": "PERF 2022",
    "ongoing_charges": "ONGOING CHARGES",
    "volatilita": "VOLATILITA'",   # colonna volatilità se presente nel file Azimut
}

KEYWORD_GENERALISTE = [
    "Globali", "Globale", "Bilanciati", "Bilanciato", "Flessibili",
    "Flessibile", "Ritorno Assoluto", "Multi-Asset", "Allocation",
    "Azionari Globali", "Obbligazionari Globali",
]

MACRO_AREA_MAP = {
    "US": ["stati uniti", "usa", "america", "s&p", "nasdaq", "north america"],
    "Europe": ["europa", "european", "euro", "stoxx", "dax", "europe"],
    "Emerging": ["emergenti", "emerging", "em ", "bric", "asia ex", "frontier"],
    "Japan": ["giappone", "japan", "japanese"],
    "Asia": ["asia", "pacifico", "pacific", "cina", "china", "india", "asia pac"],
    "Global": ["globali", "globale", "global", "world", "mondo", "acwi", "msci w",
               "international", "internazionale", "worldwide"],
    "Italy": ["italia", "italian", "btp", "ftse mib"],
}

# Regole per inferire classificazione FIDA dal nome del fondo
_CLASS_RULES = [
    # Ordine: più specifico prima
    ("Azionari Emergenti",        ["emerging market", "em equity", "emerging equity", "frontier market"]),
    ("Azionari USA",              ["s&p 500", "nasdaq", "us equity", "north america equity", "american"]),
    ("Azionari Europa",           ["europe equity", "european equity", "euro equity", "stoxx"]),
    ("Azionari Giappone",         ["japan equity", "japanese equity"]),
    ("Azionari Asia Pacifico",    ["asia pac", "asia equity", "pacific equity"]),
    ("Azionari Globali",          ["global equity", "world equity", "equity global", "global stock",
                                   "msci world", "all world", "ftse all", "acwi", "world fund"]),
    ("Azionari Tematici",         ["technology", "tech fund", "healthcare", "biotech", "pharma",
                                   "infrastructure", "energy transition", "clean energy", "robotics",
                                   "artificial intel", "semiconductor", "defense", "luxury",
                                   "water fund", "agriculture fund", "gold", "precious metal",
                                   "innovation", "digital", "cyber", "esg theme"]),
    ("Azionari",                  ["equity", "azionari", "azioni", "stock", "growth fund",
                                   "dividend", "value fund", "small cap", "large cap", "mid cap",
                                   "fund a2", "fund a acc", "fund e acc"]),
    ("Obbligazionari Emergenti",  ["emerging bond", "em bond", "emerging debt", "em debt"]),
    ("Obbligazionari High Yield", ["high yield", "junk bond", "hy bond", "coco", "at1"]),
    ("Obbligazionari Governativi",["government bond", "govt bond", "sovrani", "treasury",
                                   "sovereign", "gilts", "bund"]),
    ("Obbligazionari Globali",    ["global bond", "world bond", "aggregate bond", "bond global",
                                   "fixed income global", "obbligazionari globali"]),
    ("Obbligazionari",            ["bond", "obbligazionari", "fixed income", "reddito fisso",
                                   "debt fund", "income fund", "credit fund", "duration"]),
    ("Monetario",                 ["money market", "monetario", "liquidità", "cash fund",
                                   "overnight", "short term", "ultra short"]),
    ("Ritorno Assoluto",          ["absolute return", "ritorno assoluto", "total return",
                                   "unconstrained", "market neutral"]),
    ("Bilanciati",                ["balanced", "bilanciati", "multi asset", "multi-asset",
                                   "allocation", "60/40", "conservative alloc", "moderate alloc",
                                   "growth alloc", "income and growth", "patrimoine", "patrimony",
                                   "diversified", "portfolio fund", "income portfolio"]),
    ("Flessibili",                ["flexible", "flessibili", "dynamic allocation",
                                   "strategic allocation", "tactical", "kaldemorgen",
                                   "concept", "multi strategy", "unconstrained alloc"]),
    ("Alternativi",               ["alternative", "hedge", "long short", "long/short",
                                   "event driven", "merger arb", "macro fund"]),
]


def _infer_classificazione(nome: str, classificazione_raw: str) -> str:
    """
    Usa la classificazione raw se non vuota,
    altrimenti la inferisce dal nome del fondo.
    """
    if classificazione_raw and classificazione_raw.strip():
        return classificazione_raw.strip()
    n = nome.lower() if nome else ""
    for label, keywords in _CLASS_RULES:
        if any(k in n for k in keywords):
            return label
    return "Altro"


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte tutte le celle in stringa gestendo datetime, NaT e NaN.
    Necessario perché pd.read_excel con dtype=str non converte celle
    formattate come data in Excel.
    """
    def _cell(x):
        if x is None:
            return ""
        if isinstance(x, (datetime.datetime, datetime.date)):
            return x.strftime("%Y-%m-%d")
        if isinstance(x, float) and np.isnan(x):
            return ""
        try:
            if pd.isna(x):
                return ""
        except (TypeError, ValueError):
            pass
        return str(x)

    for col in df.columns:
        df[col] = df[col].apply(_cell)
    return df


def _to_float(val):
    """Converte stringa percentuale in float."""
    if val is None or val == "":
        return np.nan
    if isinstance(val, (int, float)):
        if isinstance(val, float) and np.isnan(val):
            return np.nan
        return float(val)
    try:
        if pd.isna(val):
            return np.nan
    except (TypeError, ValueError):
        pass
    s = str(val).replace("%", "").replace(",", ".").strip()
    if s == "" or s.lower() in ("nan", "none", "n/a", "-"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_stars(val):
    """Converte stelle in int (1-5) o None."""
    if val is None or val == "":
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, (int, float)):
        if isinstance(val, float) and np.isnan(val):
            return None
        v = int(val)
        return v if 1 <= v <= 5 else None
    s = str(val).strip()
    # conta simboli stella
    stars = s.count("★") + s.count("*")
    if stars:
        return min(5, max(1, stars))
    try:
        v = int(float(s))
        return v if 1 <= v <= 5 else None
    except ValueError:
        return None


def _infer_macro_area(testo: str) -> str:
    t = testo.lower()
    for area, keywords in MACRO_AREA_MAP.items():
        if any(k in t for k in keywords):
            return area
    return "Other"


def _fuzzy_col(df: pd.DataFrame, target: str) -> str | None:
    """
    Cerca la colonna più simile a `target` nel DataFrame.
    Strategia: esatto → case-insensitive → partial match.
    Ritorna il nome della colonna trovata o None.
    """
    # Esatto
    if target in df.columns:
        return target
    # Case-insensitive
    t_lower = target.lower().strip()
    for col in df.columns:
        if col.lower().strip() == t_lower:
            return col
    # Partial: il target è contenuto nella colonna o viceversa
    for col in df.columns:
        cl = col.lower().strip()
        if t_lower in cl or cl in t_lower:
            return col
    # Partial su parole chiave significative (rimuovi parole comuni)
    stop = {"di", "del", "della", "the", "a", "an", "e", "i", "il", "la", "le", "gli"}
    t_words = set(t_lower.split()) - stop
    for col in df.columns:
        c_words = set(col.lower().strip().split()) - stop
        if t_words & c_words:
            return col
    # Ultima chance: confronto dopo rimozione punteggiatura/apostrofi
    import re
    t_clean = re.sub(r"[^a-z0-9\s]", "", t_lower)
    for col in df.columns:
        c_clean = re.sub(r"[^a-z0-9\s]", "", col.lower().strip())
        if t_clean and c_clean and (t_clean in c_clean or c_clean in t_clean):
            return col
    return None


def _remap_columns(df: pd.DataFrame, col_map: dict) -> dict:
    """
    Ritorna dizionario {chiave_logica: nome_colonna_effettivo}.
    Per ogni colonna attesa prova prima esatto, poi fuzzy.
    """
    mapping = {}
    for key, target in col_map.items():
        found = _fuzzy_col(df, target)
        mapping[key] = found  # None se non trovata
    return mapping


def _normalize_perf(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    for key in ["perf_1y", "perf_3y", "perf_ytd", "perf_2024", "perf_2023", "perf_2022"]:
        col = col_map.get(key)
        if col and col in df.columns:
            df[col] = df[col].apply(_to_float)
    return df


def _autoscale_perf(df: pd.DataFrame, perf_cols: list[str]) -> pd.DataFrame:
    """
    Se le performance sono in formato decimale (es. 0.085 invece di 8.5%),
    moltiplica per 100.
    Criterio sicuro: il valore massimo assoluto è < 1.5 → certamente decimale.
    (Evita falsi positivi su dataset con molti fondi obbligazionari a basso rendimento)
    """
    for col in perf_cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        # Autoscale SOLO se il massimo assoluto è < 1.5 → tutti i valori sono in [−1.5, 1.5]
        # In formato percentuale anche i fondi obbligazionari mostrano valori > 1.5 su 3 anni
        if vals.abs().max() < 1.5 and len(vals) > 3:
            df[col] = df[col].apply(lambda x: x * 100 if pd.notna(x) and isinstance(x, (int, float)) else x)
    return df


# ---------------------------------------------------------------------------
# CARICAMENTO FONDI TERZI
# ---------------------------------------------------------------------------

def load_fondi_terzi(path=None) -> pd.DataFrame:
    if path is None:
        path = BASE_DIR / "tabella_fondi_arricchita.xlsx"
    import io
    is_filelike = hasattr(path, "read") or isinstance(path, (bytes, io.BytesIO))
    if not is_filelike and not os.path.exists(path):
        return _demo_fondi_terzi()
    df = pd.read_excel(path, sheet_name="tutti quelli trasferibili", dtype=object)
    df.columns = [str(c).strip() for c in df.columns]
    df = _sanitize_df(df)

    # Fuzzy mapping colonne reali
    fm = _remap_columns(df, TERZI_COLS)
    # Rinomina le colonne trovate con i nomi standard attesi
    rename_map = {v: TERZI_COLS[k] for k, v in fm.items() if v and v != TERZI_COLS[k]}
    if rename_map:
        df = df.rename(columns=rename_map)

    c = TERZI_COLS
    df = _normalize_perf(df, c)
    # Autoscale perf da decimale a percentuale
    perf_cols = [c[k] for k in ["perf_1y","perf_3y","perf_ytd","perf_2024","perf_2023","perf_2022"] if c[k] in df.columns]
    df = _autoscale_perf(df, perf_cols)

    for key in ["commissioni", "retrocessione", "volatilita"]:
        col = c.get(key)
        if col and col in df.columns:
            df[col] = df[col].apply(_to_float)
    if c["rating_fida"] in df.columns:
        df[c["rating_fida"]] = df[c["rating_fida"]].apply(_parse_stars)
    df["_source"] = "terzi"
    df["_macro_area"] = df.apply(
        lambda r: _infer_macro_area(
            str(r.get(c["classificazione"], "")) + " " + str(r.get(c["nome"], ""))
        ), axis=1
    )
    return df


# ---------------------------------------------------------------------------
# CARICAMENTO FONDI AZIMUT
# ---------------------------------------------------------------------------

def load_fondi_azimut(path=None) -> pd.DataFrame:
    if path is None:
        path = BASE_DIR / "fondi_azimut_isin_completo_RATED.xlsx"
    import io
    is_filelike = hasattr(path, "read") or isinstance(path, (bytes, io.BytesIO))
    if not is_filelike and not os.path.exists(path):
        return _demo_fondi_azimut()
    df = pd.read_excel(path, dtype=object)
    df.columns = [str(c).strip() for c in df.columns]
    df = _sanitize_df(df)

    # Fuzzy mapping colonne reali
    fm = _remap_columns(df, AZIMUT_COLS)
    rename_map = {v: AZIMUT_COLS[k] for k, v in fm.items() if v and v != AZIMUT_COLS[k]}
    if rename_map:
        df = df.rename(columns=rename_map)

    c = AZIMUT_COLS
    df = _normalize_perf(df, c)
    perf_cols = [c[k] for k in ["perf_1y","perf_3y","perf_ytd","perf_2024","perf_2023","perf_2022"] if c[k] in df.columns]
    df = _autoscale_perf(df, perf_cols)

    if c["ongoing_charges"] in df.columns:
        df[c["ongoing_charges"]] = df[c["ongoing_charges"]].apply(_to_float)
    for key in ["stelle_fida", "stelle_ms"]:
        col = c.get(key)
        if col and col in df.columns:
            df[col] = df[col].apply(_parse_stars)

    # Filtra di default: EUR non hedged ACC
    mask = pd.Series([True] * len(df))
    if c["valuta"] in df.columns:
        mask &= df[c["valuta"]].str.upper().fillna("") == "EUR"
    if c["hedged"] in df.columns:
        hedged_col = df[c["hedged"]].str.upper().fillna("")
        mask &= ~hedged_col.isin(["SI", "YES", "Y", "1", "TRUE", "HEDGED"])
    if c["acc_dis"] in df.columns:
        mask &= df[c["acc_dis"]].str.upper().fillna("").isin(["ACC", "A", "ACCUMULATION", "ACCUMULATING"])
    df = df[mask].copy()

    df["_source"] = "azimut"
    df["_macro_area"] = df.apply(
        lambda r: _infer_macro_area(
            str(r.get(c["classificazione"], "")) + " " + str(r.get(c["nome"], ""))
        ), axis=1
    )
    return df


# ---------------------------------------------------------------------------
# UNIONE E NORMALIZZAZIONE COMUNE
# ---------------------------------------------------------------------------

def build_unified_fund_df(df_terzi: pd.DataFrame, df_azimut: pd.DataFrame) -> pd.DataFrame:
    """Ritorna DataFrame unificato con colonne standard per scoring."""
    rows = []

    for _, r in df_terzi.iterrows():
        c = TERZI_COLS
        nome_val = str(r.get(c["nome"], "") or "")
        class_val = str(r.get(c["classificazione"], "") or "")
        rows.append({
            "isin": r.get(c["isin"], ""),
            "nome": nome_val,
            "casa": r.get(c["casa"], ""),
            "classificazione": _infer_classificazione(nome_val, class_val),
            "perf_1y": r.get(c["perf_1y"], np.nan),
            "perf_3y": r.get(c["perf_3y"], np.nan),
            "perf_ytd": r.get(c["perf_ytd"], np.nan),
            "perf_2024": r.get(c["perf_2024"], np.nan),
            "perf_2023": r.get(c["perf_2023"], np.nan),
            "perf_2022": r.get(c["perf_2022"], np.nan),
            "volatilita": r.get(c["volatilita"], np.nan),
            "rating_fida": r.get(c["rating_fida"], None),
            "rating_ms": None,
            "acc_distr": r.get(c["acc_distr"], ""),
            "retrocessione": r.get(c["retrocessione"], np.nan),
            "commissioni": r.get(c["commissioni"], np.nan),
            "_source": "terzi",
            "_macro_area": r.get("_macro_area", "Other"),
        })

    for _, r in df_azimut.iterrows():
        c = AZIMUT_COLS
        nome_val = str(r.get(c["nome"], "") or "")
        class_val = str(r.get(c["classificazione"], "") or "")
        rows.append({
            "isin": r.get(c["isin"], ""),
            "nome": nome_val,
            "casa": "Azimut",
            "classificazione": _infer_classificazione(nome_val, class_val),
            "perf_1y": r.get(c["perf_1y"], np.nan),
            "perf_3y": r.get(c["perf_3y"], np.nan),
            "perf_ytd": r.get(c["perf_ytd"], np.nan),
            "perf_2024": r.get(c["perf_2024"], np.nan),
            "perf_2023": r.get(c["perf_2023"], np.nan),
            "perf_2022": r.get(c["perf_2022"], np.nan),
            "volatilita": r.get(c.get("volatilita", "VOLATILITA'"), np.nan),
            "rating_fida": r.get(c["stelle_fida"], None),
            "rating_ms": r.get(c["stelle_ms"], None),
            "acc_distr": r.get(c["acc_dis"], ""),
            "retrocessione": np.nan,
            "commissioni": r.get(c["ongoing_charges"], np.nan),
            "_source": "azimut",
            "_macro_area": r.get("_macro_area", "Other"),
        })

    df = pd.DataFrame(rows)
    df["isin"] = df["isin"].fillna("").astype(str).str.strip()
    df = df[df["isin"] != ""].copy()
    df = df.drop_duplicates(subset=["isin"]).reset_index(drop=True)

    # Autoscale perf: se valori in range 0-1 (es. 0.27) → moltiplica x100
    perf_cols_unified = ["perf_1y", "perf_3y", "perf_ytd",
                          "perf_2024", "perf_2023", "perf_2022"]
    df = _autoscale_perf(df, perf_cols_unified)
    return df


# ---------------------------------------------------------------------------
# DEMO DATA (quando i file Excel non sono presenti)
# ---------------------------------------------------------------------------

def _demo_fondi_terzi() -> pd.DataFrame:
    demo = [
        {TERZI_COLS["isin"]: "LU0048578792", TERZI_COLS["nome"]: "Xtrackers MSCI World (Demo)",
         TERZI_COLS["casa"]: "DWS", TERZI_COLS["classificazione"]: "Azionari Globali",
         TERZI_COLS["perf_1y"]: 12.5, TERZI_COLS["perf_3y"]: 8.2,
         TERZI_COLS["perf_ytd"]: 4.1, TERZI_COLS["perf_2024"]: 15.0,
         TERZI_COLS["perf_2023"]: 18.0, TERZI_COLS["perf_2022"]: -14.0,
         TERZI_COLS["volatilita"]: 14.2, TERZI_COLS["rating_fida"]: 4,
         TERZI_COLS["retrocessione"]: 0.4, TERZI_COLS["commissioni"]: 1.5,
         TERZI_COLS["acc_distr"]: "ACC", "_source": "terzi", "_macro_area": "Global"},
        {TERZI_COLS["isin"]: "IE00B4WXJJ64", TERZI_COLS["nome"]: "iShares Euro Govt Bond (Demo)",
         TERZI_COLS["casa"]: "BlackRock", TERZI_COLS["classificazione"]: "Obbligazionari Globali",
         TERZI_COLS["perf_1y"]: 2.1, TERZI_COLS["perf_3y"]: -1.5,
         TERZI_COLS["perf_ytd"]: 1.2, TERZI_COLS["perf_2024"]: 3.5,
         TERZI_COLS["perf_2023"]: 5.0, TERZI_COLS["perf_2022"]: -18.0,
         TERZI_COLS["volatilita"]: 6.8, TERZI_COLS["rating_fida"]: 3,
         TERZI_COLS["retrocessione"]: 0.2, TERZI_COLS["commissioni"]: 0.8,
         TERZI_COLS["acc_distr"]: "DISTR", "_source": "terzi", "_macro_area": "Europe"},
    ]
    return pd.DataFrame(demo)


def _demo_fondi_azimut() -> pd.DataFrame:
    demo = [
        {AZIMUT_COLS["nome"]: "AZ Fund1 - AZ Equity Global (Demo)",
         AZIMUT_COLS["isin"]: "LU0123456789",
         AZIMUT_COLS["share_class"]: "A-ACC-EUR", AZIMUT_COLS["valuta"]: "EUR",
         AZIMUT_COLS["acc_dis"]: "ACC", AZIMUT_COLS["hedged"]: "NO",
         AZIMUT_COLS["stelle_fida"]: 4, AZIMUT_COLS["stelle_ms"]: 4,
         AZIMUT_COLS["classificazione"]: "Azionari Globali",
         AZIMUT_COLS["perf_1y"]: 11.0, AZIMUT_COLS["perf_3y"]: 7.5,
         AZIMUT_COLS["perf_ytd"]: 3.8, AZIMUT_COLS["perf_2024"]: 14.0,
         AZIMUT_COLS["perf_2023"]: 17.0, AZIMUT_COLS["perf_2022"]: -13.0,
         AZIMUT_COLS["ongoing_charges"]: 1.8,
         "_source": "azimut", "_macro_area": "Global"},
    ]
    return pd.DataFrame(demo)
