"""
scoring.py — Score Qualità per fondi (sezione 4).
"""

import numpy as np
import pandas as pd

FIDA_MULT = {None: 1.0, 1: 0.8, 2: 0.9, 3: 1.0, 4: 1.15, 5: 1.30}

KEYWORD_GENERALISTE = [
    "Globali", "Globale", "Bilanciati", "Bilanciato", "Flessibili",
    "Flessibile", "Ritorno Assoluto", "Multi-Asset", "Allocation",
    "Azionari Globali", "Obbligazionari Globali",
]


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
    vol = volatility if not (pd.isna(volatility) or volatility is None or volatility == 0) else 10.0

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
    scores = []
    for _, row in df.iterrows():
        s = compute_score(
            perf_3y_ann=row.get("perf_3y", np.nan),
            perf_1y=row.get("perf_1y", np.nan),
            volatility=row.get("volatilita", np.nan),
            fida_stars=row.get("rating_fida", None),
            perf_2022=row.get("perf_2022", np.nan),
            perf_2023=row.get("perf_2023", np.nan),
            perf_2024=row.get("perf_2024", np.nan),
        )
        scores.append(s)
    df = df.copy()
    df["score_qualita"] = scores
    return df


def is_generalista(classificazione: str) -> bool:
    if not isinstance(classificazione, str):
        return False
    cl = classificazione.lower()
    return any(k.lower() in cl for k in KEYWORD_GENERALISTE)
