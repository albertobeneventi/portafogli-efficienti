"""
scoring.py — Score Qualità per fondi (sezione 4).
"""

import numpy as np
import pandas as pd

FIDA_MULT = {None: 1.0, 1: 0.8, 2: 0.9, 3: 1.0, 4: 1.15, 5: 1.30}

# Keyword che identificano fondi TEMATICI/NICCHIA (esclusi dai Generalisti).
# Tutto ciò che NON contiene queste keyword è considerato "generalista"
# (ampio mercato, multi-asset, globale, bilanciato, obbligazionario diversificato, ecc.)
KEYWORD_TEMATICI = [
    # Paesi singoli o aree molto specifiche
    "vietnam", "africa", "mena", "latin america", "latinoamerica",
    "pakistan", "nigeria", "kenya", "frontier",
    # Settoriali
    "tecnologia", "technology", "tech",
    "healthcare", "salute", "farmaceutic",
    "biotech", "biotechnology",
    "infrastructure", "infrastrutture",
    "robotics", "robotica", "automation",
    "acqua", "water",
    "agricoltur", "agriculture",
    "lusso", "luxury", "brand",
    "energia rinnovabile", "clean energy", "energia pulita",
    "difesa", "defense", "defence",
    "minerari", "mining", "gold", "oro", "metalli preziosi",
    "real estate", "immobiliare", "reit",
    "finanziari", "financial",
    # Obbligazionari molto specifici
    "coco", "at1", "convertibili", "convertible",
    "abs", "sukuk", "renminbi", "cnh",
    "inflation link", "inflation-link",
    # Tematici ESG ultra-specifici
    "impact invest", "microfinance",
]

# Keyword che identificano SEMPRE un fondo come generalista
# (usate come override positivo)
KEYWORD_GENERALISTE = [
    "Globali", "Globale", "Bilanciati", "Bilanciato", "Flessibili",
    "Flessibile", "Ritorno Assoluto", "Multi-Asset", "Allocation",
    "Azionari Globali", "Obbligazionari Globali",
    "Internazionali", "Internazionale",
    "Paesi Sviluppati", "Mercati Sviluppati", "Worldwide", "World",
    "Diversificati", "Diversificato",
    "Total Return", "Absolute Return",
    "Azionari Europa", "Azionari USA", "Azionari Emergenti",
    "Obbligazionari Euro", "Obbligazionari High Yield",
    "Obbligazionari Emergenti", "Obbligazionari Governativi",
    "Obbligazionari Corporate", "Obbligazionari Societari",
    "Bilanciati Moderati", "Bilanciati Aggressivi",
]
# Nota: NON includere "Azionari" o "Obbligazionari" da soli —
# matcherebbero "Azionari Settoriali" che è tematico.


def compute_score(
    perf_3y_ann: float,
    perf_1y: float,
    volatility: float,
    fida_stars,
    perf_2022: float = np.nan,
    perf_2023: float = np.nan,
    perf_2024: float = np.nan,
) -> float:
    """Calcola lo Score Qualità per un singolo fondo."""
    # Gestione valori mancanti
    p3 = perf_3y_ann if not (pd.isna(perf_3y_ann) or perf_3y_ann is None) else 0.0
    p1 = perf_1y if not (pd.isna(perf_1y) or perf_1y is None) else 0.0
    # Floor di volatilità a 0.5% — evita divisione per zero nei monetari (score ∞)
    raw_vol = volatility if not (pd.isna(volatility) or volatility is None) else 0.0
    vol = max(float(raw_vol), 0.5)

    sharpe_proxy = p3 / vol

    base_score = (p3 * 0.50) + (sharpe_proxy * 0.30) + (p1 * 0.20)

    mult = FIDA_MULT.get(fida_stars, 1.0)
    score = base_score * mult

    if p3 < 0:
        score *= 0.5

    bad_years = 0
    for perf_year in [perf_2022, perf_2023, perf_2024]:
        if not pd.isna(perf_year) and perf_year is not None:
            if perf_year < -10:
                bad_years += 1
    consistency_bonus = 1.0 - (bad_years * 0.15)
    score *= max(0.4, consistency_bonus)

    return round(score, 4)


def compute_scores_df(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge colonna 'score_qualita' al DataFrame unificato."""
    def _safe_float(v):
        """Converte a float, restituisce np.nan se non possibile."""
        if v is None:
            return np.nan
        try:
            f = float(v)
            return f if not (f != f) else np.nan   # NaN check
        except (TypeError, ValueError):
            return np.nan

    scores = []
    for _, row in df.iterrows():
        try:
            s = compute_score(
                perf_3y_ann=_safe_float(row.get("perf_3y")),
                perf_1y=_safe_float(row.get("perf_1y")),
                volatility=_safe_float(row.get("volatilita")),
                fida_stars=row.get("rating_fida", None),
                perf_2022=_safe_float(row.get("perf_2022")),
                perf_2023=_safe_float(row.get("perf_2023")),
                perf_2024=_safe_float(row.get("perf_2024")),
            )
        except Exception:
            s = 0.0
        scores.append(s)
    df = df.copy()
    df["score_qualita"] = scores
    return df


def is_generalista(classificazione: str) -> bool:
    """
    Un fondo è GENERALISTA se:
    - NON contiene keyword tematiche/nicchia (KEYWORD_TEMATICI), OPPURE
    - Contiene keyword esplicitamente generaliste (KEYWORD_GENERALISTE override)

    Logica invertita rispetto alla versione precedente: invece di cercare
    cosa è generalista, escludiamo cosa è tematico/nicchia.
    """
    if not isinstance(classificazione, str) or not classificazione.strip():
        return True   # classificazione assente → trattato come generalista
    cl = classificazione.lower()

    # Override positivo: keyword esplicitamente generaliste
    if any(k.lower() in cl for k in KEYWORD_GENERALISTE):
        return True

    # Escludi i tematici/nicchia
    if any(k.lower() in cl for k in KEYWORD_TEMATICI):
        return False

    # Per default: tutto ciò che non è esplicitamente tematico è generalista
    return True
