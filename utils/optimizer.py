"""
optimizer.py — wrapper PyPortfolioOpt per frontiera efficiente e Black-Litterman.
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    from pypfopt import (
        EfficientFrontier,
        risk_models,
        expected_returns,
        BlackLittermanModel,
        black_litterman,
        plotting,
    )
    from pypfopt.exceptions import OptimizationError
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False


def _align_prices(price_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Allinea serie storiche, forward-fill, dropna."""
    df = pd.DataFrame(price_dict)
    df = df.sort_index().ffill().dropna()
    return df


def compute_efficient_frontier(
    price_dict: dict[str, pd.Series],
    weight_bounds: tuple = (0.03, 0.30),
    risk_free_rate: float = 0.025,
    n_points: int = 50,
    n_monte_carlo: int = 5000,
    forced_include: Optional[list] = None,
    forced_exclude: Optional[list] = None,
) -> dict:
    """
    Calcola frontiera efficiente.
    Ritorna dict con chiavi:
      - "frontier_df": DataFrame con punti frontiera (ret, vol, sharpe)
      - "max_sharpe": dict con weights, ret, vol, sharpe
      - "min_variance": dict con weights, ret, vol, sharpe
      - "monte_carlo": DataFrame (ret, vol, sharpe, weights)
      - "error": stringa se fallisce
    """
    if not HAS_PYPFOPT:
        return {"error": "PyPortfolioOpt non installato."}

    prices = _align_prices(price_dict)
    if len(prices) < 12:
        return {"error": f"Dati insufficienti: {len(prices)} osservazioni (min 12)."}
    if len(prices.columns) < 3:
        return {"error": "Selezionare almeno 3 asset."}

    # Applica exclusion
    if forced_exclude:
        cols = [c for c in prices.columns if c not in forced_exclude]
        prices = prices[cols]

    mu = expected_returns.mean_historical_return(prices, frequency=12)
    S = risk_models.CovarianceShrinkage(prices, frequency=12).ledoit_wolf()

    # Includi forzati: peso minimo aumentato — PyPortfolioOpt vuole lista di tuple
    w_bounds_list = []
    for col in prices.columns:
        if forced_include and col in forced_include:
            w_bounds_list.append((0.05, weight_bounds[1]))
        else:
            w_bounds_list.append(weight_bounds)

    # Max Sharpe
    try:
        ef = EfficientFrontier(mu, S, weight_bounds=w_bounds_list)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        ms_weights = ef.clean_weights()
        ms_perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        max_sharpe = {
            "weights": ms_weights,
            "ret": float(ms_perf[0]),
            "vol": float(ms_perf[1]),
            "sharpe": float(ms_perf[2]),
        }
    except (OptimizationError, Exception) as e:
        max_sharpe = {"error": str(e)}

    # Min Variance
    try:
        ef2 = EfficientFrontier(mu, S, weight_bounds=w_bounds_list)
        ef2.min_volatility()
        mv_weights = ef2.clean_weights()
        mv_perf = ef2.portfolio_performance(risk_free_rate=risk_free_rate)
        min_variance = {
            "weights": mv_weights,
            "ret": float(mv_perf[0]),
            "vol": float(mv_perf[1]),
            "sharpe": float(mv_perf[2]),
        }
    except (OptimizationError, Exception) as e:
        min_variance = {"error": str(e)}

    # Frontiera efficiente (n_points punti)
    frontier_points = []
    try:
        min_vol = min_variance.get("vol", 0.05) if "error" not in min_variance else 0.05
        max_ret = float(mu.max())
        min_ret = float(mu.min())
        target_returns = np.linspace(max(min_ret, min_vol * 0.5), max_ret * 0.95, n_points)
        for target in target_returns:
            try:
                ef_pt = EfficientFrontier(mu, S, weight_bounds=w_bounds_list)
                ef_pt.efficient_return(target_return=target)
                p = ef_pt.portfolio_performance(risk_free_rate=risk_free_rate)
                frontier_points.append({"ret": p[0], "vol": p[1], "sharpe": p[2]})
            except Exception:
                pass
    except Exception:
        pass
    frontier_df = pd.DataFrame(frontier_points)

    # Monte Carlo
    mc_rows = []
    np.random.seed(42)
    n_assets = len(prices.columns)
    for _ in range(n_monte_carlo):
        w = np.random.dirichlet(np.ones(n_assets))
        # applica bounds: clip e renormalize
        w = np.clip(w, weight_bounds[0], weight_bounds[1])
        w /= w.sum()
        r_p = float(mu.values @ w)
        v_p = float(np.sqrt(w @ S.values @ w))
        sh = (r_p - risk_free_rate) / v_p if v_p > 0 else 0
        mc_rows.append({"ret": r_p, "vol": v_p, "sharpe": sh,
                        "weights": dict(zip(prices.columns, w.round(4)))})
    mc_df = pd.DataFrame(mc_rows)

    return {
        "frontier_df": frontier_df,
        "max_sharpe": max_sharpe,
        "min_variance": min_variance,
        "monte_carlo": mc_df,
        "assets": list(prices.columns),
        "mu": mu.to_dict(),
        "cov": pd.DataFrame(S, index=prices.columns, columns=prices.columns).to_dict(),
    }


def compute_black_litterman(
    price_dict: dict[str, pd.Series],
    views: dict[str, float],  # {isin: expected_return}
    confidences: dict[str, float],  # {isin: confidence 0-1}
    weight_bounds: tuple = (0.03, 0.30),
    risk_free_rate: float = 0.025,
) -> dict:
    """Calcola portafoglio Black-Litterman."""
    if not HAS_PYPFOPT:
        return {"error": "PyPortfolioOpt non installato."}

    prices = _align_prices(price_dict)
    if len(prices) < 12:
        return {"error": "Dati insufficienti per Black-Litterman."}

    S = risk_models.CovarianceShrinkage(prices, frequency=12).ledoit_wolf()
    mkt_weights = pd.Series(
        np.ones(len(prices.columns)) / len(prices.columns),
        index=prices.columns
    )

    try:
        viewdict = {k: v / 100 for k, v in views.items() if k in prices.columns}
        if not viewdict:
            return {"error": "Nessuna view valida per gli asset selezionati."}

        omega = np.diag([
            (1 - confidences.get(k, 0.5)) ** 2 * 0.04
            for k in viewdict.keys()
        ])

        bl = BlackLittermanModel(S, pi="market", market_caps=mkt_weights,
                                 absolute_views=viewdict, omega=omega)
        rets_bl = bl.bl_returns()
        S_bl = bl.bl_cov()

        ef = EfficientFrontier(rets_bl, S_bl, weight_bounds=weight_bounds)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        bl_weights = ef.clean_weights()
        bl_perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)

        return {
            "weights": bl_weights,
            "ret": bl_perf[0],
            "vol": bl_perf[1],
            "sharpe": bl_perf[2],
            "bl_returns": rets_bl.to_dict(),
        }
    except Exception as e:
        return {"error": str(e)}


def estimate_max_drawdown(weights: dict, price_dict: dict[str, pd.Series]) -> float:
    """Stima max drawdown storico del portafoglio."""
    prices = _align_prices(price_dict)
    w = np.array([weights.get(c, 0) for c in prices.columns])
    portfolio = (prices * w).sum(axis=1)
    portfolio = portfolio / portfolio.iloc[0] * 100
    rolling_max = portfolio.cummax()
    dd = (portfolio - rolling_max) / rolling_max
    return float(dd.min()) * 100  # in %
