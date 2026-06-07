"""
optimizer.py — PyPortfolioOpt: Frontiera Efficiente e Black-Litterman.

Approccio ibrido:
  - ETF / azioni: μ e Σ da serie storiche reali (yfinance)
  - Fondi UCITS : μ da perf_3y (Excel), σ da volatilita (Excel),
                  correlazioni da matrice categoriale parametrizzata
  - Blended     : unico vettore μ e matrice Σ che combina le due sorgenti

Black-Litterman:
  - Views automatiche derivate dallo Score Qualità (override manuale opzionale)
  - Omega proporzionale a (1 - confidenza)² × varianza
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from pypfopt import (
        EfficientFrontier,
        risk_models,
        expected_returns,
        BlackLittermanModel,
        plotting,
    )
    from pypfopt.exceptions import OptimizationError
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False


# ---------------------------------------------------------------------------
# BENCHMARK DI RENDIMENTO PER SOTTOCATEGORIA FIDA
# ---------------------------------------------------------------------------
# Rendimenti attesi annui forward-looking per sottocategoria.
# Basati su: equity risk premium CAPM area Euro, yield-to-maturity mercato,
# stime consenso asset manager (BlackRock, Vanguard, JPM LTCMA 2024).
#
# Logica: keyword matching sulla classificazione FIDA del fondo.
# La funzione _subcategory_prior() scorre le tuple (keyword, prior) in ordine
# e restituisce il primo match. Fallback = prior della macro-categoria.
#
# Riferimenti principali (stime 10Y forward, area Euro investore):
#   Azionari Emergenti:      9.0%  (rischio paese/valuta + ERP più alto)
#   Azionari USA:            8.0%  (ERP USA, dollaro, valutazione elevata)
#   Azionari Globali:        7.5%  (media ponderata sviluppati)
#   Azionari Europa:         7.0%  (valutazioni più basse, crescita più lenta)
#   Azionari Settoriali:     7.5%  (simile a globali, dispersion alta)
#   Obbligaz. High Yield:    5.5%  (spread HY IG + rischio default)
#   Obbligaz. Emergenti:     5.0%  (spread EM + rischio paese)
#   Obbligaz. Corporate IG:  3.5%  (spread IG + duration)
#   Obbligaz. Governativi:   2.5%  (risk-free EUR ~2.5%)
#   Bilanciati aggressivi:   6.0%
#   Bilanciati moderati:     5.0%
#   Bilanciati conservativi: 3.5%
#   Alternativi/Absolute:    4.0%
#   Monetari:                2.5%

# Lista ordinata: (keyword_lower_in_classificazione, prior_annuo)
_SUBCAT_PRIOR_MAP: list[tuple[str, float]] = [
    # ── AZIONARI per area geografica ──────────────────────────────────────
    ("emergent",          0.090),   # Emergenti, Emerging
    ("frontier",          0.095),   # Frontier markets
    ("asia",              0.085),   # Asia Pacific
    ("pacifico",          0.085),
    ("cina",              0.090),   # Cina, China
    ("china",             0.090),
    ("india",             0.095),
    ("latam",             0.090),
    ("latin",             0.090),
    ("africa",            0.095),
    ("giappone",          0.075),
    ("japan",             0.075),
    ("usa",               0.080),
    ("america",           0.080),
    ("stati uniti",       0.080),
    ("nord america",      0.080),
    ("europa",            0.070),
    ("europe",            0.070),
    ("eurozona",          0.070),
    ("italia",            0.070),
    ("globali",           0.075),
    ("global",            0.075),
    ("worldwide",         0.075),
    ("internazional",     0.075),
    # ── OBBLIGAZIONARI per tipo ───────────────────────────────────────────
    ("high yield",        0.055),
    ("alto rendimento",   0.055),
    ("convertibil",       0.055),
    ("subordinati",       0.050),
    ("at1",               0.060),
    ("coco",              0.060),
    ("obbligaz.*emerg",   0.050),   # regex non supportato — usiamo keyword semplice
    ("obbligaz. emerg",   0.050),
    ("emerging bond",     0.050),
    ("corporate",         0.035),
    ("societar",          0.035),
    ("inflation",         0.030),
    ("indicizzat",        0.030),
    ("governativ",        0.025),
    ("governo",           0.025),
    ("statali",           0.025),
    ("treasury",          0.025),
    # ── BILANCIATI per profilo ────────────────────────────────────────────
    ("aggressiv",         0.060),
    ("dinamic",           0.058),
    ("moderato",          0.050),
    ("equilibrat",        0.050),
    ("conservativ",       0.035),
    ("prudente",          0.035),
    ("flessibil",         0.050),
    ("multi-asset",       0.050),
    ("multi asset",       0.050),
    ("allocation",        0.050),
    # ── ALTERNATIVI ───────────────────────────────────────────────────────
    ("absolute return",   0.040),
    ("ritorno assoluto",  0.040),
    ("long short",        0.045),
    ("market neutral",    0.035),
    ("real asset",        0.050),
    ("commodit",          0.050),
    ("materie prime",     0.050),
    ("oro",               0.045),
    ("gold",              0.045),
    ("immobiliar",        0.055),
    ("reit",              0.055),
    # ── MONETARI ─────────────────────────────────────────────────────────
    ("monetar",           0.025),
    ("liquidit",          0.025),
    ("money market",      0.025),
]

# Macro-prior di fallback (quando classificazione assente o non matchata)
_MU_PRIOR_MACRO: dict[str, float] = {
    "Azionario":       0.075,
    "Obbligazionario": 0.030,
    "Bilanciato":      0.050,
    "Alternativo":     0.040,
    "Monetario":       0.025,
    "ETF":             0.070,
    "Azione":          0.080,
}


def _subcategory_prior(classificazione: str, macro_bucket: str) -> float:
    """
    Restituisce il rendimento atteso forward-looking per sottocategoria FIDA.
    Prima scorre _SUBCAT_PRIOR_MAP (keyword sulla classificazione),
    poi usa _MU_PRIOR_MACRO come fallback.
    """
    if classificazione:
        cl_low = str(classificazione).lower()
        for kw, prior in _SUBCAT_PRIOR_MAP:
            if kw in cl_low:
                return prior
    return _MU_PRIOR_MACRO.get(macro_bucket, 0.060)


# ---------------------------------------------------------------------------
# MATRICE DI CORRELAZIONE CATEGORIALE (fondi UCITS senza serie storiche)
# ---------------------------------------------------------------------------
# Valori derivati dalla letteratura empirica su fondi UCITS europei.
# Chiavi: bucket restituito da classify_bucket()

_CAT_CORR: dict[tuple, float] = {
    ("Azionario",    "Azionario"):    0.78,
    ("Azionario",    "Bilanciato"):   0.55,
    ("Azionario",    "Obbligazionario"): -0.10,
    ("Azionario",    "Alternativo"):  0.22,
    ("Azionario",    "Monetario"):    0.03,
    ("Bilanciato",   "Bilanciato"):   0.68,
    ("Bilanciato",   "Obbligazionario"): 0.42,
    ("Bilanciato",   "Alternativo"):  0.28,
    ("Bilanciato",   "Monetario"):    0.05,
    ("Obbligazionario", "Obbligazionario"): 0.62,
    ("Obbligazionario", "Alternativo"): 0.18,
    ("Obbligazionario", "Monetario"): 0.12,
    ("Alternativo",  "Alternativo"):  0.32,
    ("Alternativo",  "Monetario"):    0.06,
    ("Monetario",    "Monetario"):    0.85,
    # ETF/azioni vs fondi (approssimazioni conservative)
    ("ETF",          "Azionario"):    0.72,
    ("ETF",          "Bilanciato"):   0.50,
    ("ETF",          "Obbligazionario"): -0.08,
    ("ETF",          "Alternativo"):  0.20,
    ("ETF",          "Monetario"):    0.04,
    ("ETF",          "ETF"):          0.70,   # placeholder; per ETF reali usiamo Σ empirica
    ("Azione",       "Azionario"):    0.75,
    ("Azione",       "Bilanciato"):   0.52,
    ("Azione",       "Obbligazionario"): -0.05,
    ("Azione",       "Alternativo"):  0.18,
    ("Azione",       "Monetario"):    0.03,
    ("Azione",       "ETF"):          0.72,
    ("Azione",       "Azione"):       0.65,
}


def _cat_corr(cat_a: str, cat_b: str) -> float:
    """Restituisce la correlazione categoriale tra due asset class."""
    c = _CAT_CORR.get((cat_a, cat_b)) or _CAT_CORR.get((cat_b, cat_a))
    return c if c is not None else 0.25   # default conservativo


def _bucket_of(info: dict) -> str:
    """Ricava il bucket macro di un asset dal suo info dict."""
    from utils.constraints import classify_bucket
    cl = str(info.get("classificazione", info.get("categoria", ""))).strip()
    if not cl:
        return "ETF" if info.get("ticker") else "Azionario"
    b = classify_bucket(cl)
    if b == "Altro":
        # Distinguish ETF vs stock
        cl_l = cl.lower()
        if "etf" in cl_l or "indice" in cl_l:
            return "ETF"
        if "azion" in cl_l or "azione" in cl_l or "equity" in cl_l:
            return "Azione"
        return "Azionario"
    return b


# ---------------------------------------------------------------------------
# ALLINEAMENTO SERIE STORICHE
# ---------------------------------------------------------------------------

def _align_prices(price_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Allinea serie storiche, forward-fill, dropna."""
    df = pd.DataFrame(price_dict)
    df = df.sort_index().ffill().dropna()
    return df


# ---------------------------------------------------------------------------
# COSTRUZIONE μ e Σ IBRIDA
# ---------------------------------------------------------------------------

def build_hybrid_mu_sigma(
    assets_info: dict[str, dict],   # isin → info con perf_3y, volatilita, classificazione
    price_dict:  dict[str, pd.Series],  # isin → serie storica (ETF/azioni)
    frequency: int = 12,
    risk_free_rate: float = 0.025,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Costruisce vettore μ e matrice Σ ibrida:
      - Asset con serie storica (price_dict): μ e σ da dati reali, correlazione reale
      - Asset solo da Excel (fondi UCITS): μ = perf_3y/100, σ = volatilita/100,
        correlazioni categoriali
    Ritorna (mu: pd.Series, cov: pd.DataFrame) in decimali annualizzati.
    """
    all_isins = list(assets_info.keys())
    n = len(all_isins)

    # ── 1. Asset con serie storica reale ──────────────────────────────────
    real_isins = [i for i in all_isins if i in price_dict
                  and isinstance(price_dict[i], pd.Series)
                  and len(price_dict[i]) >= 6]

    real_mu: dict[str, float] = {}
    real_vol: dict[str, float] = {}
    real_corr = pd.DataFrame(index=real_isins, columns=real_isins, dtype=float)

    if real_isins:
        prices_df = _align_prices({i: price_dict[i] for i in real_isins})
        if len(prices_df) >= 6:
            try:
                _mu_ser = expected_returns.mean_historical_return(prices_df, frequency=frequency)
                for i in _mu_ser.index:
                    real_mu[i] = float(_mu_ser[i])
                _ret_df = prices_df.pct_change().dropna()
                _corr   = _ret_df.corr()
                _vol_ser = _ret_df.std() * np.sqrt(frequency)
                for i in real_isins:
                    real_vol[i] = float(_vol_ser.get(i, 0.15))
                for i in real_isins:
                    for j in real_isins:
                        real_corr.loc[i, j] = float(_corr.loc[i, j]) if i in _corr.index and j in _corr.columns else (1.0 if i == j else 0.3)
            except Exception:
                pass

    # ── 2. σ per asset senza serie storica (fondi UCITS da Excel) ────────
    #
    # SCELTA METODOLOGICA — rendimenti attesi (μ):
    # Non usiamo perf_3y o perf_1y come μ. Il motivo è che:
    # - usare rendimenti storici in un ottimizzatore produce selezione a posteriori
    #   (l'ottimizzatore sceglie sempre i fondi col miglior passato)
    # - i rendimenti passati non predicono i rendimenti futuri dei fondi attivi
    #
    # μ = prior di categoria (forward-looking, basato su asset class):
    #   Azionario:       7.5% anno  (equity risk premium ~5% + rfr ~2.5%)
    #   Obbligazionario: 3.0% anno  (rendimento medio mercato europeo IG)
    #   Bilanciato:      5.0% anno
    #   Alternativo:     4.0% anno
    #   Monetario:       2.5% anno
    #
    # I benchmark per sottocategoria e la funzione _subcategory_prior()
    # sono definiti a livello di modulo (sopra).
    # μᵢ = _subcategory_prior(classificazione, macro_bucket) + alpha_i(score)

    fund_isins = [i for i in all_isins if i not in real_mu]
    stat_vol: dict[str, float] = {}

    # σ floor per categoria (dati di mercato europeo 2015-2024)
    _VOL_FLOOR = {
        "Monetario":       0.003,
        "Obbligazionario": 0.030,
        "Bilanciato":      0.065,
        "Alternativo":     0.060,
        "Azionario":       0.110,
        "ETF":             0.100,
        "Azione":          0.150,
    }
    _VOL_DEFAULT = {
        "Monetario": 0.004, "Obbligazionario": 0.045, "Bilanciato": 0.080,
        "Alternativo": 0.070, "Azionario": 0.150, "ETF": 0.130, "Azione": 0.220,
    }

    for i in fund_isins:
        info = assets_info.get(i, {})
        vol  = info.get("volatilita")
        bkt  = _bucket_of(info)
        _floor = _VOL_FLOOR.get(bkt, 0.090)

        if vol is not None and not (isinstance(vol, float) and np.isnan(vol)) and float(vol) > 0:
            v_raw = float(vol)
            if v_raw < 1.0:
                v_raw *= 100.0          # formato decimale → percentuale
            vol_val = max(v_raw / 100.0, _floor)
        else:
            vol_val = _VOL_DEFAULT.get(bkt, 0.120)
        stat_vol[i] = vol_val

    # ── 3. Costruisce μ vettore: benchmark categoria + alpha del gestore ─────
    #
    # μᵢ = μ_benchmark_categoria + αᵢ
    #
    # αᵢ è stimato dallo Score Qualità normalizzato PER CATEGORIA:
    #   - Il gestore che supera stabilmente il benchmark ha alpha > 0
    #   - Il gestore che sottoperforma ha alpha < 0 (costo quantificato)
    #   - Alpha = 0 per il gestore mediano della categoria
    #
    # Normalizzazione: rank percentile all'interno della categoria → z-score
    # Ampiezza alpha: ±ALPHA_MAX per anno (letteratura empirica UCITS europei)
    #   Top manager (>75° percentile):  +1% → +2%/anno
    #   Mediano (50° percentile):        0%/anno  (= solo benchmark categoria)
    #   Bottom manager (<25° percentile): -1% → -2%/anno
    #
    ALPHA_MAX = 0.020   # ±2%/anno: range empirico per gestione attiva UCITS

    # Raggruppa score per SOTTOCATEGORIA (classificazione FIDA) per normalizzare
    # l'alpha correttamente: un emergenti è confrontato solo con altri emergenti
    subcat_scores: dict[str, list] = {}
    for i in all_isins:
        info_i = assets_info.get(i, {})
        cl_i   = str(info_i.get("classificazione", "") or "").strip()
        sc     = info_i.get("score_qualita")
        try:
            sc_f = float(sc)
            if not np.isnan(sc_f):
                subcat_scores.setdefault(cl_i, []).append((i, sc_f))
        except (TypeError, ValueError):
            pass

    # Statistiche per sottocategoria (mediana + std)
    subcat_stats: dict[str, tuple] = {}
    for cl_i, items in subcat_scores.items():
        vals = np.array([v for _, v in items])
        med  = float(np.median(vals))
        std  = float(np.std(vals)) if len(vals) > 1 else 1.0
        subcat_stats[cl_i] = (med, max(std, 0.1))

    mu_vals = {}
    for i in all_isins:
        info_i  = assets_info.get(i, {})
        bkt_i   = _bucket_of(info_i)
        cl_i    = str(info_i.get("classificazione", "") or "").strip()

        # Prior specifico per sottocategoria FIDA
        prior_i = _subcategory_prior(cl_i, bkt_i)

        # Alpha da score normalizzato nella stessa sottocategoria
        sc = info_i.get("score_qualita")
        alpha_i = 0.0
        try:
            sc_f = float(sc)
            if not np.isnan(sc_f) and cl_i in subcat_stats:
                med, std = subcat_stats[cl_i]
                z = (sc_f - med) / std
                alpha_i = float(np.clip(z * ALPHA_MAX * 0.5, -ALPHA_MAX, ALPHA_MAX))
        except (TypeError, ValueError):
            pass

        # Segnale negativo reale (ETF/azione con serie storica in perdita)
        if i in real_mu and real_mu[i] < 0:
            mu_vals[i] = 0.5 * real_mu[i] + 0.5 * (prior_i + alpha_i)
        else:
            mu_vals[i] = prior_i + alpha_i

    mu = pd.Series(mu_vals)

    # ── 4. Costruisce Σ matrice ────────────────────────────────────────────
    cov = pd.DataFrame(index=all_isins, columns=all_isins, dtype=float)

    for i in all_isins:
        vi = real_vol.get(i, stat_vol.get(i, 0.12))
        for j in all_isins:
            vj = real_vol.get(j, stat_vol.get(j, 0.12))
            if i == j:
                cov.loc[i, j] = vi ** 2
            elif i in real_isins and j in real_isins:
                # Entrambi reali: usa correlazione empirica
                c_ij = float(real_corr.loc[i, j]) if not pd.isna(real_corr.loc[i, j]) else 0.3
                cov.loc[i, j] = vi * vj * c_ij
            else:
                # Almeno uno è un fondo: correlazione categoriale
                bi = _bucket_of(assets_info.get(i, {}))
                bj = _bucket_of(assets_info.get(j, {}))
                c_ij = _cat_corr(bi, bj)
                cov.loc[i, j] = vi * vj * c_ij

    # Assicura simmetria e PSD (piccola regolarizzazione)
    cov_np = cov.to_numpy(dtype=float)
    cov_np = (cov_np + cov_np.T) / 2
    cov_np += np.eye(n) * 1e-6   # nugget per stabilità numerica
    # Clamp eigenvalues negativi (può accadere con correlazioni categoriali)
    eigvals, eigvecs = np.linalg.eigh(cov_np)
    eigvals = np.maximum(eigvals, 1e-8)
    cov_np = eigvecs @ np.diag(eigvals) @ eigvecs.T
    cov = pd.DataFrame(cov_np, index=all_isins, columns=all_isins)

    return mu, cov


# ---------------------------------------------------------------------------
# FRONTIERA EFFICIENTE (ibrida)
# ---------------------------------------------------------------------------

def compute_efficient_frontier(
    price_dict: dict[str, pd.Series],
    weight_bounds: tuple = (0.03, 0.30),
    risk_free_rate: float = 0.025,
    n_points: int = 50,
    n_monte_carlo: int = 4000,
    forced_include: Optional[list] = None,
    forced_exclude: Optional[list] = None,
    sector_constraints: Optional[dict] = None,
    assets_info: Optional[dict] = None,   # isin → {perf_3y, volatilita, classificazione, ...}
    selected_isins: Optional[list] = None,  # lista primaria asset (include fondi senza NAV)
) -> dict:
    """
    Calcola frontiera efficiente con approccio ibrido.
    Ritorna dict con chiavi:
      frontier_df, max_sharpe, min_variance, monte_carlo, error (se fallisce)
    """
    if not HAS_PYPFOPT:
        return {"error": "PyPortfolioOpt non installato."}

    # Applica exclusion
    if forced_exclude:
        price_dict = {k: v for k, v in price_dict.items() if k not in forced_exclude}

    _info = assets_info or {}

    # Costruisce all_isins:
    # - se passato selected_isins: usa quelli come lista primaria (fondi UCITS inclusi anche senza price series)
    # - altrimenti: solo asset presenti in price_dict
    if selected_isins:
        all_isins = [i for i in selected_isins if not forced_exclude or i not in forced_exclude]
    else:
        all_isins = [i for i in price_dict if not forced_exclude or i not in forced_exclude]

    # Conta asset "viabili": con serie storica O con statistiche Excel (perf_3y/perf_1y)
    _n_series = sum(1 for i in all_isins if i in price_dict)
    _n_stats  = sum(1 for i in all_isins
                    if i not in price_dict
                    and (_info.get(i, {}).get("perf_3y") is not None
                         or _info.get(i, {}).get("perf_1y") is not None))
    _n_viable = _n_series + _n_stats

    if _n_viable < 3:
        return {"error": f"Asset insufficienti: {_n_series} serie storiche + {_n_stats} da Excel = {_n_viable} totali (min 3)."}
    if len(all_isins) < 3:
        return {"error": f"Asset selezionati insufficienti: {len(all_isins)} (min 3)."}

    # μ e Σ ibride — passa solo i dati dei fondi selezionati
    _sel_info = {i: _info.get(i, {}) for i in all_isins}
    mu, cov = build_hybrid_mu_sigma(_sel_info, price_dict,
                                    risk_free_rate=risk_free_rate)
    # Allinea all_isins a quelli in mu
    all_isins = [i for i in all_isins if i in mu.index]
    mu  = mu[all_isins]
    cov = cov.loc[all_isins, all_isins]

    if len(all_isins) < 3:
        return {"error": "Troppi asset esclusi dopo pulizia dati."}

    # Weight bounds per asset (forced_include ha peso minimo maggiore)
    w_bounds_list = []
    for isin in all_isins:
        if forced_include and isin in forced_include:
            w_bounds_list.append((max(weight_bounds[0], 0.05), weight_bounds[1]))
        else:
            w_bounds_list.append(weight_bounds)

    def _apply_sector(ef_obj):
        if not sector_constraints:
            return
        mapper = {k: v for k, v in sector_constraints.get("mapper", {}).items()
                  if k in all_isins}
        if mapper:
            try:
                ef_obj.add_sector_constraints(
                    mapper,
                    sector_constraints.get("lower", {}),
                    sector_constraints.get("upper", {}),
                )
            except Exception:
                pass

    # ── Max Sharpe ────────────────────────────────────────────────────────
    try:
        ef = EfficientFrontier(mu, cov, weight_bounds=w_bounds_list)
        _apply_sector(ef)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        ms_weights = ef.clean_weights()
        ms_perf    = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        max_sharpe = {
            "weights": ms_weights,
            "ret":     float(ms_perf[0]),
            "vol":     float(ms_perf[1]),
            "sharpe":  float(ms_perf[2]),
        }
    except Exception as e:
        _e_str = str(e).lower()
        if "risk-free" in _e_str or "expected return" in _e_str:
            # Tutti i fondi hanno μ ≤ rfr (es. solo obbligazionari a basso rendimento)
            # Fallback: usa rfr=0 (massimizza Sharpe grezzo) oppure min_variance
            try:
                ef_fb = EfficientFrontier(mu, cov, weight_bounds=w_bounds_list)
                _apply_sector(ef_fb)
                ef_fb.max_sharpe(risk_free_rate=0.0)
                ms_weights = ef_fb.clean_weights()
                ms_perf    = ef_fb.portfolio_performance(risk_free_rate=risk_free_rate)
                max_sharpe = {
                    "weights": ms_weights,
                    "ret":     float(ms_perf[0]),
                    "vol":     float(ms_perf[1]),
                    "sharpe":  float(ms_perf[2]),
                    "_note":   "rfr=0 (rendimenti tutti sotto tasso risk-free)",
                }
            except Exception as e2:
                # Ultimo fallback: min variance
                try:
                    ef_fb2 = EfficientFrontier(mu, cov, weight_bounds=w_bounds_list)
                    _apply_sector(ef_fb2)
                    ef_fb2.min_volatility()
                    ms_weights = ef_fb2.clean_weights()
                    ms_perf    = ef_fb2.portfolio_performance(risk_free_rate=risk_free_rate)
                    max_sharpe = {
                        "weights": ms_weights,
                        "ret":     float(ms_perf[0]),
                        "vol":     float(ms_perf[1]),
                        "sharpe":  float(ms_perf[2]),
                        "_note":   "min-variance (fallback)",
                    }
                except Exception as e3:
                    max_sharpe = {"error": str(e3)}
        else:
            max_sharpe = {"error": str(e)}

    # ── Min Variance ──────────────────────────────────────────────────────
    try:
        ef2 = EfficientFrontier(mu, cov, weight_bounds=w_bounds_list)
        _apply_sector(ef2)
        ef2.min_volatility()
        mv_weights = ef2.clean_weights()
        mv_perf    = ef2.portfolio_performance(risk_free_rate=risk_free_rate)
        min_variance = {
            "weights": mv_weights,
            "ret":     float(mv_perf[0]),
            "vol":     float(mv_perf[1]),
            "sharpe":  float(mv_perf[2]),
        }
    except Exception as e:
        min_variance = {"error": str(e)}

    # ── Frontiera efficiente (n_points punti) ────────────────────────────
    frontier_points = []
    try:
        min_ret = float(mu.min())
        max_ret = float(mu.max())
        min_vol_val = min_variance.get("vol", 0.05) if "error" not in min_variance else 0.05
        targets = np.linspace(max(min_ret, min_vol_val * 0.5), max_ret * 0.95, n_points)
        for target in targets:
            try:
                ef_pt = EfficientFrontier(mu, cov, weight_bounds=w_bounds_list)
                ef_pt.efficient_return(target_return=float(target))
                p = ef_pt.portfolio_performance(risk_free_rate=risk_free_rate)
                frontier_points.append({"ret": p[0], "vol": p[1], "sharpe": p[2]})
            except Exception:
                pass
    except Exception:
        pass
    frontier_df = pd.DataFrame(frontier_points)

    # ── Monte Carlo ───────────────────────────────────────────────────────
    mc_rows = []
    np.random.seed(42)
    n_a = len(all_isins)
    mu_np  = mu.values
    cov_np = cov.values
    lo, hi = weight_bounds
    for _ in range(n_monte_carlo):
        w = np.random.dirichlet(np.ones(n_a))
        w = np.clip(w, lo, hi); w /= w.sum()
        r_p = float(mu_np @ w)
        v_p = float(np.sqrt(w @ cov_np @ w))
        sh  = (r_p - risk_free_rate) / v_p if v_p > 0 else 0.0
        mc_rows.append({"ret": r_p, "vol": v_p, "sharpe": sh})
    mc_df = pd.DataFrame(mc_rows)

    return {
        "frontier_df":  frontier_df,
        "max_sharpe":   max_sharpe,
        "min_variance": min_variance,
        "monte_carlo":  mc_df,
        "assets":       all_isins,
        "mu":           mu.to_dict(),
        "cov":          cov.to_dict(),
    }


# ---------------------------------------------------------------------------
# BLACK-LITTERMAN
# ---------------------------------------------------------------------------

def compute_black_litterman(
    price_dict: dict[str, pd.Series],
    views: dict[str, float],        # {isin: rendimento_atteso_%}
    confidences: dict[str, float],  # {isin: 0-1}
    weight_bounds: tuple = (0.03, 0.30),
    risk_free_rate: float = 0.025,
    assets_info: Optional[dict] = None,
    forced_include: Optional[list] = None,
    sector_constraints: Optional[dict] = None,
    selected_isins: Optional[list] = None,  # lista primaria asset (include fondi senza NAV)
) -> dict:
    """
    Portafoglio Black-Litterman con views manuali o auto.
    """
    if not HAS_PYPFOPT:
        return {"error": "PyPortfolioOpt non installato."}

    _info = assets_info or {}
    all_isins = list(selected_isins) if selected_isins else list(price_dict.keys())

    _sel_info = {i: _info.get(i, {}) for i in all_isins}
    mu_prior, cov = build_hybrid_mu_sigma(_sel_info, price_dict,
                                          risk_free_rate=risk_free_rate)
    all_isins = [i for i in all_isins if i in mu_prior.index]
    mu_prior  = mu_prior[all_isins]
    cov       = cov.loc[all_isins, all_isins]

    if len(all_isins) < 3:
        return {"error": "Asset insufficienti per Black-Litterman."}

    # Views: converti % → decimali, filtra solo asset nel pool
    viewdict = {}
    for k, v in views.items():
        if k in all_isins:
            viewdict[k] = float(v) / 100.0

    if not viewdict:
        return {"error": "Nessuna view valida per gli asset selezionati."}

    # Omega: diagonale con incertezza proporzionale a (1-confidenza)² × varianza
    omega_diag = []
    for k in viewdict:
        conf = float(confidences.get(k, 0.5))
        var_k = float(cov.loc[k, k]) if k in cov.index else 0.04
        omega_diag.append(((1 - conf) ** 2) * var_k)
    omega = np.diag(omega_diag)

    # Market weights: equal-weight (proxy per mancanza di cap pesi reali)
    mkt_weights = pd.Series(
        np.ones(len(all_isins)) / len(all_isins), index=all_isins
    )

    try:
        bl = BlackLittermanModel(
            cov, pi="market", market_caps=mkt_weights,
            absolute_views=viewdict, omega=omega,
        )
        rets_bl = bl.bl_returns()
        S_bl    = bl.bl_cov()

        # Weight bounds con forced_include
        w_bounds_list = []
        for isin in all_isins:
            if forced_include and isin in forced_include:
                w_bounds_list.append((max(weight_bounds[0], 0.05), weight_bounds[1]))
            else:
                w_bounds_list.append(weight_bounds)

        ef = EfficientFrontier(rets_bl, S_bl, weight_bounds=w_bounds_list)
        if sector_constraints:
            mapper = {k: v for k, v in sector_constraints.get("mapper", {}).items()
                      if k in all_isins}
            if mapper:
                try:
                    ef.add_sector_constraints(
                        mapper,
                        sector_constraints.get("lower", {}),
                        sector_constraints.get("upper", {}),
                    )
                except Exception:
                    pass
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        bl_weights = ef.clean_weights()
        bl_perf    = ef.portfolio_performance(risk_free_rate=risk_free_rate)

        return {
            "weights":    bl_weights,
            "ret":        float(bl_perf[0]),
            "vol":        float(bl_perf[1]),
            "sharpe":     float(bl_perf[2]),
            "bl_returns": rets_bl.to_dict(),
            "views_used": viewdict,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# BLACK-LITTERMAN AUTOMATICO (views da Score Qualità)
# ---------------------------------------------------------------------------

def compute_bl_auto_views(
    assets_info: dict[str, dict],
    base_returns: Optional[dict] = None,   # {isin: μ} da build_hybrid_mu_sigma
) -> tuple[dict, dict]:
    """
    Genera views e confidenze automatiche da Score Qualità.
    View = μ_base × (1 + alpha)  dove alpha dipende dalla distanza dello score dalla mediana.
    Confidenza = min(0.9, max(0.2, score_rel))

    Ritorna (views_dict, confidences_dict) in formato % (da passare a compute_black_litterman).
    """
    if not assets_info:
        return {}, {}

    scores = {}
    for isin, info in assets_info.items():
        s = info.get("score_qualita")
        if s is not None and not (isinstance(s, float) and np.isnan(s)):
            scores[isin] = float(s)

    if not scores:
        return {}, {}

    score_vals = list(scores.values())
    median_s = float(np.median(score_vals))
    std_s    = max(float(np.std(score_vals)), 0.1)

    views: dict[str, float] = {}
    confs: dict[str, float] = {}

    for isin, score in scores.items():
        # Z-score normalizzato dello score
        z = (score - median_s) / std_s
        # Alpha: da -15% a +20% rispetto al rendimento base
        alpha = np.clip(z * 0.07, -0.15, 0.20)

        base_ret = (base_returns or {}).get(isin, 0.06)   # default 6%
        view_ret = base_ret * (1.0 + alpha)

        # Converti in %
        views[isin] = round(view_ret * 100.0, 2)
        # Confidenza: maggiore se lo score è lontano dalla mediana
        confs[isin] = round(np.clip(0.4 + abs(z) * 0.15, 0.25, 0.85), 2)

    return views, confs


# ---------------------------------------------------------------------------
# MAX DRAWDOWN STORICO
# ---------------------------------------------------------------------------

def estimate_max_drawdown(
    weights: dict,
    price_dict: dict[str, pd.Series],
) -> float:
    """Stima max drawdown storico del portafoglio (%)."""
    try:
        prices = _align_prices({k: v for k, v in price_dict.items() if k in weights})
        if prices.empty:
            return 0.0
        w = np.array([weights.get(c, 0) for c in prices.columns])
        w_sum = w.sum()
        if w_sum == 0:
            return 0.0
        w = w / w_sum
        portfolio = (prices / prices.iloc[0] * 100) @ w
        rolling_max = portfolio.cummax()
        dd = (portfolio - rolling_max) / rolling_max
        return float(dd.min()) * 100
    except Exception:
        return 0.0
