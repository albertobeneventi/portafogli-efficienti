"""
data_loader.py — carica e normalizza i file Excel fondi terzi e Azimut.
"""

import os
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
}

KEYWORD_GENERALISTE = [
    "Globali", "Globale", "Bilanciati", "Bilanciato", "Flessibili",
    "Flessibile", "Ritorno Assoluto", "Multi-Asset", "Allocation",
    "Azionari Globali", "Obbligazionari Globali",
]

MACRO_AREA_MAP = {
    "US": ["stati uniti", "usa", "america", "s&p", "nasdaq"],
    "Europe": ["europa", "european", "euro", "stoxx", "dax"],
    "Emerging": ["emergenti", "emerging", "em ", "bric", "asia ex"],
    "Japan": ["giappone", "japan", "japanese"],
    "Asia": ["asia", "pacifico", "pacific", "cina", "china", "india"],
    "Global": ["globali", "globale", "global", "world", "mondo", "acwi", "msci w"],
    "Italy": ["italia", "italian", "btp", "ftse mib"],
}


def _to_float(val):
    """Converte stringa percentuale in float."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace("%", "").replace(",", ".").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_stars(val):
    """Converte stelle in int (1-5) o None."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
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


def _normalize_perf(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    for key in ["perf_1y", "perf_3y", "perf_ytd", "perf_2024", "perf_2023", "perf_2022"]:
        col = col_map.get(key)
        if col and col in df.columns:
            df[col] = df[col].apply(_to_float)
    return df


# ---------------------------------------------------------------------------
# CARICAMENTO FONDI TERZI
# ---------------------------------------------------------------------------

def load_fondi_terzi(path=None) -> pd.DataFrame:
    if path is None:
        path = BASE_DIR / "tabella_fondi_arricchita.xlsx"
    # accetta sia Path/str che file-like (BytesIO da st.file_uploader)
    import io
    is_filelike = hasattr(path, "read") or isinstance(path, (bytes, io.BytesIO))
    if not is_filelike and not os.path.exists(path):
        return _demo_fondi_terzi()
    df = pd.read_excel(path, sheet_name="tutti quelli trasferibili", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = _normalize_perf(df, TERZI_COLS)
    c = TERZI_COLS
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
    df = pd.read_excel(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = _normalize_perf(df, AZIMUT_COLS)
    c = AZIMUT_COLS
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
        rows.append({
            "isin": r.get(c["isin"], ""),
            "nome": r.get(c["nome"], ""),
            "casa": r.get(c["casa"], ""),
            "classificazione": r.get(c["classificazione"], ""),
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
        rows.append({
            "isin": r.get(c["isin"], ""),
            "nome": r.get(c["nome"], ""),
            "casa": "Azimut",
            "classificazione": r.get(c["classificazione"], ""),
            "perf_1y": r.get(c["perf_1y"], np.nan),
            "perf_3y": r.get(c["perf_3y"], np.nan),
            "perf_ytd": r.get(c["perf_ytd"], np.nan),
            "perf_2024": r.get(c["perf_2024"], np.nan),
            "perf_2023": r.get(c["perf_2023"], np.nan),
            "perf_2022": r.get(c["perf_2022"], np.nan),
            "volatilita": np.nan,
            "rating_fida": r.get(c["stelle_fida"], None),
            "rating_ms": r.get(c["stelle_ms"], None),
            "acc_distr": r.get(c["acc_dis"], ""),
            "retrocessione": np.nan,
            "commissioni": r.get(c["ongoing_charges"], np.nan),
            "_source": "azimut",
            "_macro_area": r.get("_macro_area", "Other"),
        })

    df = pd.DataFrame(rows)
    df["isin"] = df["isin"].fillna("").str.strip()
    df = df[df["isin"] != ""].copy()
    df = df.drop_duplicates(subset=["isin"]).reset_index(drop=True)
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
