"""
nav_fetcher.py — recupero serie storiche NAV per fondi e prezzi per ETF.
Cascata: Morningstar → FondiDoc → serie sintetica da rendimenti annuali.
"""

import json
import numpy as np
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
NAV_CACHE_FILE = DATA_DIR / "nav_cache.json"
CACHE_TTL_HOURS = 24

MS_SCREENER_URL = (
    "https://lt.morningstar.com/api/rest.svc/klr5zyak8x/security/screener"
    "?page=1&pageSize=1&sortOrder=LegalName+asc&outputType=json"
    "&version=1&languageId=it-IT&currencyId=EUR&universeIds=FOESP%24%24ALL"
    "&securityDataPoints=SecId%7CReturnM12%7CReturnM36%7CReturnM0%7CStandardDeviationM36"
    "&filters=ISIN+IN+{isin}"
)

FONDIDOC_URL = "https://www.fondidoc.it/Fondo/{isin}/Rendimenti"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/html,*/*;q=0.8",
}


def _load_cache() -> dict:
    if NAV_CACHE_FILE.exists():
        try:
            with open(NAV_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    DATA_DIR.mkdir(exist_ok=True)
    with open(NAV_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _cache_valid(entry: dict) -> bool:
    ts = entry.get("timestamp")
    if not ts:
        return False
    try:
        age = datetime.now() - datetime.fromisoformat(ts)
        return age < timedelta(hours=CACHE_TTL_HOURS)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# SORGENTE 1: Morningstar
# ---------------------------------------------------------------------------

def _fetch_morningstar(isin: str) -> pd.Series | None:
    """Prova a recuperare serie mensile da Morningstar via cloudscraper."""
    url = MS_SCREENER_URL.format(isin=isin)
    try:
        if HAS_CLOUDSCRAPER:
            scraper = cloudscraper.create_scraper()
            resp = scraper.get(url, timeout=15)
        else:
            resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        rows = data.get("rows", [])
        if not rows:
            return None
        row = rows[0]
        # Morningstar screener: rendimenti aggregati, non serie storica
        # Usiamo i punti disponibili per costruire serie sintetica
        r1y = row.get("ReturnM12")
        r3y = row.get("ReturnM36")
        r_ytd = row.get("ReturnM0")
        if r1y is None and r3y is None:
            return None
        return _synthetic_series_from_returns(
            perf_1y=r1y, perf_3y=r3y, perf_ytd=r_ytd
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SORGENTE 2: FondiDoc
# ---------------------------------------------------------------------------

def _fetch_fondidoc(isin: str) -> pd.Series | None:
    url = FONDIDOC_URL.format(isin=isin)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")
        # Cerca tabella rendimenti annuali
        table = soup.find("table")
        if not table:
            return None
        rows = table.find_all("tr")
        yearly: dict = {}
        for row in rows:
            cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if len(cols) >= 2:
                year_match = re.search(r"(20\d{2})", cols[0])
                if year_match:
                    try:
                        val = float(cols[1].replace("%", "").replace(",", "."))
                        yearly[int(year_match.group(1))] = val
                    except ValueError:
                        pass
        if not yearly:
            return None
        return _synthetic_series_from_yearly(yearly)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SORGENTE 3: yfinance (ETF/azioni/obbligazioni quotate)
# ---------------------------------------------------------------------------

def _fetch_yfinance(ticker: str, period: str = "3y") -> pd.Series | None:
    try:
        data = yf.download(ticker, period=period, interval="1mo",
                           auto_adjust=True, progress=False)
        if data.empty:
            return None
        prices = data["Close"]
        if hasattr(prices, "squeeze"):
            prices = prices.squeeze()
        prices.index = pd.to_datetime(prices.index)
        prices.name = ticker
        return prices
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SERIE SINTETICA DA RENDIMENTI ANNUALI
# ---------------------------------------------------------------------------

def _synthetic_series_from_returns(
    perf_1y=None, perf_3y=None, perf_ytd=None,
    perf_2022=None, perf_2023=None, perf_2024=None,
) -> pd.Series:
    """
    Genera serie mensile sintetica interpolando rendimenti annuali disponibili.
    Base 100 a 3 anni fa, crescita mensile costante per anno.
    """
    today = datetime.now()
    dates = pd.date_range(end=today, periods=36, freq="MS")
    values = [100.0]

    # Tenta di usare rendimenti annuali storici se disponibili
    yearly = {}
    if perf_2022 is not None:
        yearly[2022] = perf_2022
    if perf_2023 is not None:
        yearly[2023] = perf_2023
    if perf_2024 is not None:
        yearly[2024] = perf_2024

    if yearly:
        return _synthetic_series_from_yearly(yearly)

    # Fallback: usa perf_3y annualizzata
    if perf_3y is not None:
        monthly_ret = (1 + perf_3y / 100) ** (1 / 12) - 1
        for _ in range(35):
            values.append(values[-1] * (1 + monthly_ret))
    elif perf_1y is not None:
        monthly_ret = (1 + perf_1y / 100) ** (1 / 12) - 1
        for _ in range(35):
            values.append(values[-1] * (1 + monthly_ret))
    else:
        values = [100.0] * 36

    series = pd.Series(values, index=dates)
    return series


def _synthetic_series_from_yearly(yearly: dict) -> pd.Series:
    """Costruisce serie mensile da dict {anno: rendimento%}."""
    today = datetime.now()
    current_year = today.year
    start_year = min(yearly.keys()) if yearly else current_year - 3

    dates = []
    values = []
    nav = 100.0

    for year in range(start_year, current_year + 1):
        annual_ret = yearly.get(year, 0.0) / 100
        monthly_ret = (1 + annual_ret) ** (1 / 12) - 1
        for month in range(1, 13):
            d = datetime(year, month, 1)
            if d > today:
                break
            dates.append(d)
            nav = nav * (1 + monthly_ret)
            values.append(nav)

    if not dates:
        return pd.Series(dtype=float)
    return pd.Series(values, index=pd.DatetimeIndex(dates))


# ---------------------------------------------------------------------------
# FUNZIONE PRINCIPALE
# ---------------------------------------------------------------------------

def get_nav_series(
    isin: str,
    ticker: str | None = None,
    perf_1y: float | None = None,
    perf_3y: float | None = None,
    perf_ytd: float | None = None,
    perf_2022: float | None = None,
    perf_2023: float | None = None,
    perf_2024: float | None = None,
    period: str = "3y",
) -> pd.Series | None:
    """
    Recupera serie storica prezzi/NAV per un ISIN.
    Cascata: cache → Morningstar → FondiDoc → yfinance → sintetica.
    """
    cache = _load_cache()
    if isin in cache and _cache_valid(cache[isin]):
        entry = cache[isin]
        series_data = entry.get("series")
        if series_data:
            idx = pd.to_datetime(list(series_data.keys()))
            vals = list(series_data.values())
            return pd.Series(vals, index=idx, name=isin)

    series = None

    # 1. Morningstar
    series = _fetch_morningstar(isin)

    # 2. FondiDoc
    if series is None:
        series = _fetch_fondidoc(isin)
        time.sleep(0.2)

    # 3. yfinance (per ETF con ticker noto)
    if series is None and ticker:
        series = _fetch_yfinance(ticker, period=period)

    # 4. Serie sintetica
    if series is None:
        series = _synthetic_series_from_returns(
            perf_1y=perf_1y, perf_3y=perf_3y, perf_ytd=perf_ytd,
            perf_2022=perf_2022, perf_2023=perf_2023, perf_2024=perf_2024,
        )

    if series is not None and not series.empty:
        cache[isin] = {
            "timestamp": datetime.now().isoformat(),
            "series": {str(k): float(v) for k, v in series.items() if not np.isnan(v)},
        }
        _save_cache(cache)

    return series


def get_multiple_nav(assets: list[dict], period: str = "3y") -> dict[str, pd.Series]:
    """
    Recupera serie per lista di asset.
    Ogni elemento: {"isin": ..., "ticker": ..., "perf_1y": ..., ...}
    """
    result = {}
    for asset in assets:
        isin = asset.get("isin", "")
        if not isin:
            continue
        series = get_nav_series(
            isin=isin,
            ticker=asset.get("ticker"),
            perf_1y=asset.get("perf_1y"),
            perf_3y=asset.get("perf_3y"),
            perf_ytd=asset.get("perf_ytd"),
            perf_2022=asset.get("perf_2022"),
            perf_2023=asset.get("perf_2023"),
            perf_2024=asset.get("perf_2024"),
            period=period,
        )
        if series is not None and len(series) >= 12:
            result[isin] = series
    return result
