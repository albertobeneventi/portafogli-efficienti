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


def _strategy_root(name: str) -> str:
    """Prime 3 parole significative del nome (esclude ACC, MINC, EUR, USD…)."""
    stop = {"acc", "minc", "eur", "usd", "gbp", "chf", "a", "b", "c", "d",
            "i", "r", "e", "ret", "dist", "distr", "fund", "class", "sicav"}
    words = re.sub(r"[^a-zA-Z0-9\s]", " ", name.lower()).split()
    meaningful = [w for w in words if w not in stop and len(w) > 1]
    return " ".join(meaningful[:3])


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
    df = df.copy()
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

def build_lista_a(df_unified: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Top-100 fondi generalisti per Score Qualità con vincoli."""
    df = df_unified.copy()
    df = compute_scores_df(df)

    # Eleggibilità
    mask = (
        df["classificazione"].apply(is_generalista) &
        df["perf_3y"].notna() &
        (df["perf_3y"] != 0) &
        (df["perf_3y"] >= 0)
    )
    fida_ok = df["rating_fida"].apply(lambda x: x is None or (isinstance(x, (int, float)) and x >= 3))
    mask &= fida_ok
    df = df[mask].copy()

    result = select_top_n_with_constraints(
        df, n=n,
        max_per_casa=3,
        max_per_classificazione=2,
        max_per_macro_area=3,
    )
    result["_lista"] = "A"
    return result


# ---------------------------------------------------------------------------
# COSTRUZIONE LISTA B — 100 fondi tematici/specializzati
# ---------------------------------------------------------------------------

def build_lista_b(df_unified: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Top-100 fondi tematici per Score Qualità con vincoli."""
    df = df_unified.copy()
    df = compute_scores_df(df)

    # Eleggibilità: NON generalisti + perf_1y disponibile
    mask = (
        ~df["classificazione"].apply(is_generalista) &
        df["perf_1y"].notna()
    )
    df = df[mask].copy()

    result = select_top_n_with_constraints(
        df, n=n,
        max_per_casa=2,
        max_per_classificazione=1,
        max_per_macro_area=5,
    )
    result["_lista"] = "B"
    return result


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
    "Azionario": ["azionari", "equity", "azioni", "stock", "growth"],
    "Obbligazionario": ["obbligazionari", "bond", "fixed income", "obbligazioni", "monetari"],
    "Monetario": ["monetari", "money market", "liquidità", "overnight"],
    "Bilanciato": ["bilanciati", "balanced", "multi-asset", "allocation", "flessibili", "flexible"],
    "Alternativo": ["ritorno assoluto", "absolute return", "alternativo", "hedge", "long short"],
}


def classify_bucket(classificazione: str) -> str:
    """Mappa classificazione FIDA a bucket macro."""
    cl = str(classificazione).lower()
    for bucket, keywords in BUCKET_KEYWORDS.items():
        if any(k in cl for k in keywords):
            return bucket
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
