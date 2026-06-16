"""
constraints.py — vincoli di diversificazione per Portafoglio Qualità.
Costruisce anche le Liste A e B.
"""

import re
import numpy as np
import pandas as pd
from .scoring import compute_scores_df, is_generalista, KEYWORD_GENERALISTE

MACRO_AREAS = ["US", "Europe", "Emerging", "Japan", "Asia", "Global", "Italy"]

GENERALISTE_MACRO_COVER = [
    "Azionario globale", "Obbligazionario globale",
    "Bilanciato", "Ritorno assoluto", "Flessibile",
]


def strategy_root(name: str) -> str:
    """Prime 3 parole significative del nome (esclude ACC, MINC, EUR, USD…).
    Usata per individuare share class diverse dello stesso fondo."""
    stop = {"acc", "minc", "eur", "usd", "gbp", "chf", "a", "b", "c", "d",
            "i", "r", "e", "ret", "dist", "distr", "fund", "class", "sicav"}
    words = re.sub(r"[^a-zA-Z0-9\s]", " ", name.lower()).split()
    meaningful = [w for w in words if w not in stop and len(w) > 1]
    return " ".join(meaningful[:3])


# Alias retro-compatibile
_strategy_root = strategy_root


def select_top_n_with_constraints(
    df: pd.DataFrame,
    n: int = 5,
    max_per_casa: int = 1,
    max_per_classificazione: int = 1,
    max_per_macro_area: int = 2,
) -> pd.DataFrame:
    """
    Dato un DataFrame con score_qualita, seleziona i migliori N fondi
    rispettando i vincoli di diversificazione.
    Tiebreaker: a parità di score (±5%) vince il fondo con retrocessione più alta.
    """
    from .scoring import compute_scores_df
    df = df.copy()
    # Assicura che score_qualita esista (failsafe se chiamata senza compute_scores_df)
    if "score_qualita" not in df.columns:
        df = compute_scores_df(df)
    # Tiebreaker: ordina per score desc, poi retrocessione desc
    df["_ret_tie"] = df.get("retrocessione", pd.Series(np.zeros(len(df)))).fillna(0)
    df = df.sort_values(["score_qualita", "_ret_tie"], ascending=[False, False]).reset_index(drop=True)

    selected = []
    casa_count: dict = {}
    class_count: dict = {}
    macro_count: dict = {}
    strategy_seen: set = set()

    for _, row in df.iterrows():
        casa = str(row.get("casa", "")).strip()
        cl = str(row.get("classificazione", "")).strip()
        macro = str(row.get("_macro_area", "Other")).strip()
        strat = _strategy_root(str(row.get("nome", "")))

        if casa_count.get(casa, 0) >= max_per_casa:
            continue
        if class_count.get(cl, 0) >= max_per_classificazione:
            continue
        if macro_count.get(macro, 0) >= max_per_macro_area:
            continue
        if strat in strategy_seen:
            continue

        selected.append(row)
        casa_count[casa] = casa_count.get(casa, 0) + 1
        class_count[cl] = class_count.get(cl, 0) + 1
        macro_count[macro] = macro_count.get(macro, 0) + 1
        strategy_seen.add(strat)

        if len(selected) >= n:
            break

    result = pd.DataFrame(selected).drop(columns=["_ret_tie"], errors="ignore")
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# COSTRUZIONE LISTA A — 100 fondi generalisti/globali
# ---------------------------------------------------------------------------

def _is_missing(x) -> bool:
    """True se il valore è None, NaN o stringa vuota."""
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() == ""
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


_MONETARY_KEYWORDS = [
    "money market", "monetari", "monetario", "liquidità",
    "overnight", "cash fund", "ultra short", "ultrashort", "lvnav", "vnav",
]

def _cap_monetari(df: pd.DataFrame, max_n: int = 10) -> pd.DataFrame:
    """
    Limita i fondi monetari a max_n (i migliori per score).
    Evita che saturino la lista con rendimenti artificialmente alti.
    """
    def _is_mon(cl):
        cl = str(cl).lower()
        return any(k in cl for k in _MONETARY_KEYWORDS)

    is_mon = df["classificazione"].apply(_is_mon)
    mon_df  = df[is_mon].copy()
    rest_df = df[~is_mon].copy()

    if len(mon_df) > max_n:
        # Tieni solo i top max_n per score
        if "score_qualita" in mon_df.columns:
            mon_df = mon_df.sort_values("score_qualita", ascending=False).head(max_n)
        else:
            mon_df = mon_df.head(max_n)

    return pd.concat([rest_df, mon_df], ignore_index=True)


def build_lista_generalisti(df_unified: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """
    Fondi Generalisti — Top 100 per Score Qualità.
    Comprende: globali, bilanciati, flessibili, ritorno assoluto, multi-asset,
    obbligazionari diversificati, azionari globali/europei/emergenti.

    Vincoli:
      - max 5 fondi per casa (diversificazione emittente)
      - max 15 per classificazione (permette copertura ampia per categoria)
      - max monetari: 10 (cap esplicito per evitare distorsioni di score)
      - nessun duplicato di strategia (share class diverse dello stesso fondo)
    """
    df = df_unified.copy()
    df = compute_scores_df(df)

    # Colonne perf opzionali: usa quelle disponibili
    if "perf_3y" in df.columns and "perf_1y" in df.columns:
        perf_ref = df["perf_3y"].fillna(df["perf_1y"])
    elif "perf_3y" in df.columns:
        perf_ref = df["perf_3y"]
    elif "perf_1y" in df.columns:
        perf_ref = df["perf_1y"]
    else:
        perf_ref = pd.Series([np.nan] * len(df), index=df.index)

    _cl_col = df["classificazione"] if "classificazione" in df.columns else pd.Series([""] * len(df), index=df.index)
    mask = (
        _cl_col.apply(is_generalista) &
        perf_ref.notna()
    )
    df = df[mask].copy()

    # Cap monetari prima della selezione
    df = _cap_monetari(df, max_n=10)

    result = select_top_n_with_constraints(
        df, n=n,
        max_per_casa=5,
        max_per_classificazione=15,
        max_per_macro_area=30,   # praticamente illimitato: macro_area spesso non valorizzata
    )
    result["_lista"] = "generalisti"
    return result


# Alias per compatibilità
def build_lista_a(df_unified: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    return build_lista_generalisti(df_unified, n)


# ---------------------------------------------------------------------------
# COSTRUZIONE LISTA TEMATICI — fondi specializzati
# ---------------------------------------------------------------------------

def build_lista_tematici(
    df_unified: pd.DataFrame,
    n: int = 100,
    exclude_isins: list | None = None,
) -> pd.DataFrame:
    """
    Fondi Tematici — Top 100 per Score Qualità.
    Comprende: settoriali, tematici ESG, paesi specifici, high yield,
    convertibili, inflation-linked, materie prime, ecc.

    Vincoli:
      - max 5 fondi per casa
      - max 10 per classificazione
      - max monetari: 5 (ne bastano pochissimi nella lista tematici)
      - nessun duplicato di strategia
      - exclude_isins: ISIN già presenti nella lista generalisti (evita overlap)
    """
    df = df_unified.copy()
    df = compute_scores_df(df)

    _cl_col = df["classificazione"] if "classificazione" in df.columns else pd.Series([""] * len(df), index=df.index)
    _p1_col = df["perf_1y"] if "perf_1y" in df.columns else pd.Series([np.nan] * len(df), index=df.index)
    mask = (
        ~_cl_col.apply(is_generalista) &
        _p1_col.notna()
    )
    df = df[mask].copy()

    # Rimuovi ISIN già in lista generalisti
    if exclude_isins:
        df = df[~df["isin"].isin(exclude_isins)]

    # Cap monetari (lista tematici: massimo 5)
    df = _cap_monetari(df, max_n=5)

    result = select_top_n_with_constraints(
        df, n=n,
        max_per_casa=5,
        max_per_classificazione=10,
        max_per_macro_area=30,
    )
    result["_lista"] = "tematici"
    return result


# Alias per compatibilità
def build_lista_b(
    df_unified: pd.DataFrame,
    n: int = 100,
    exclude_isins: list | None = None,
) -> pd.DataFrame:
    return build_lista_tematici(df_unified, n, exclude_isins=exclude_isins)


# ---------------------------------------------------------------------------
# PROFILI DI RISCHIO E BUCKET
# ---------------------------------------------------------------------------

PROFILI = {
    "Conservativo": {
        "Obbligazionario": 60,
        "Monetario": 15,
        "Bilanciato": 15,
        "Azionario": 10,
    },
    "Equilibrato": {
        "Obbligazionario": 40,
        "Bilanciato": 25,
        "Azionario": 30,
        "Alternativo": 5,
    },
    "Accrescitivo": {
        "Azionario": 55,
        "Obbligazionario": 25,
        "Bilanciato": 15,
        "Alternativo": 5,
    },
    "Dinamico": {
        "Azionario": 75,
        "Obbligazionario": 15,
        "Alternativo": 10,
    },
}

BUCKET_KEYWORDS = {
    # 1. Monetario — parole esclusive, non ambigue
    "Monetario": ["money market", "liquidità", "overnight",
                  "ultra short", "ultrashort", "cash fund", "monetario"],
    # 2. Alternativo — parole esclusive
    "Alternativo": ["ritorno assoluto", "absolute return",
                    "market neutral", "event driven",
                    "long short", "long/short", "hedge fund"],
    # 3. Bilanciato — prima di az/obbl per gestire "multi-asset"
    "Bilanciato": ["bilanciati", "bilanciato", "balanced",
                   "multi-asset", "multi asset", "allocation",
                   "flessibili", "flessibile", "flexible",
                   "patrimoine", "income and growth", "diversified"],
    # 4. Obbligazionario — PRIMA di Azionario (evita match su "azionari" in "obbligazionari")
    "Obbligazionario": ["obbligazionari", "obbligazionario",
                        "bond", "fixed income", "reddito fisso",
                        "corporate bond", "government bond",
                        "high yield", "convertibili",
                        "inflation linked", "duration"],
    # 5. Azionario — per ultimo, parole non ambigue
    "Azionario": ["azionari", "azionario", "equity", "azioni", "stock",
                  "tematici", "tematico", "growth fund", "dividend fund",
                  "value fund", "small cap", "large cap", "mid cap",
                  "technology", "healthcare", "biotech",
                  "innovation", "world equity", "global equity",
                  "msci world", "s&p 500", "emerging equity"],
}


def classify_bucket(classificazione: str) -> str:
    """Mappa classificazione FIDA a bucket macro (ordine: Monetario→Alt→Bil→Obbl→Az)."""
    import re
    cl = str(classificazione).lower()

    def _match(keyword: str) -> bool:
        # Word-boundary: il keyword non deve essere preceduto/seguito da lettere
        pattern = r"(?<![a-z])" + re.escape(keyword) + r"(?![a-z])"
        return bool(re.search(pattern, cl))

    for bucket, keywords in BUCKET_KEYWORDS.items():
        if any(_match(k) for k in keywords):
            return bucket
    # Fallback: parole radice con word-boundary
    if _match("obbl") or _match("bond") or _match("reddito fisso"):
        return "Obbligazionario"
    if _match("azion") or _match("equit") or _match("stock"):
        return "Azionario"
    if _match("bilanci") or _match("flexi") or _match("multi"):
        return "Bilanciato"
    return "Altro"


def build_portfolio_quality(
    df_unified: pd.DataFrame,
    profilo: str = "Equilibrato",
    fondi_per_bucket: int = 4,
) -> dict:
    """
    Costruisce portafoglio qualità per profilo di rischio.
    Ritorna dict {bucket: DataFrame top-N fondi}.
    """
    df = compute_scores_df(df_unified)
    df["_bucket"] = df["classificazione"].apply(classify_bucket)
    allocazioni = PROFILI.get(profilo, PROFILI["Equilibrato"])
    result = {}
    for bucket in allocazioni:
        sub = df[df["_bucket"] == bucket].copy()
        if sub.empty:
            result[bucket] = pd.DataFrame()
            continue
        top = select_top_n_with_constraints(
            sub, n=fondi_per_bucket,
            max_per_casa=1,
            max_per_classificazione=1,
            max_per_macro_area=2,
        )
        top["_peso_bucket"] = allocazioni[bucket]
        # peso equo all'interno del bucket
        top["_peso_fondo"] = round(allocazioni[bucket] / len(top), 1) if len(top) else 0
        result[bucket] = top
    return result
