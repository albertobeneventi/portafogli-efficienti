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
    """Scarica serie mensile da yfinance. Ritorna None se fallisce."""
    try:
        data = yf.download(ticker, period=period, interval="1mo",
                           auto_adjust=True, progress=False, threads=False)
        if data.empty:
            return None
        # Gestisci sia MultiIndex (multi-ticker) che Index semplice
        if hasattr(data.columns, "levels"):
            close = data["Close"][ticker] if ticker in data["Close"].columns else None
        else:
            close = data["Close"] if "Close" in data.columns else None
        if close is None or close.empty:
            return None
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        close.index = pd.to_datetime(close.index)
        close.name = ticker
        close = close.dropna()
        return close if len(close) >= 6 else None
    except Exception:
        return None


def _fetch_yfinance_with_fallbacks(ticker: str, period: str = "3y") -> pd.Series | None:
    """
    Prova il ticker principale, poi varianti di mercato.
    Utile per ETF che potrebbero non essere quotati su .MI ma su .L o .DE.
    """
    # Ticker esatto
    s = _fetch_yfinance(ticker, period)
    if s is not None:
        return s

    # Varianti: cambia suffisso mercato
    base = ticker.split(".")[0] if "." in ticker else ticker
    variants = []
    if ticker.endswith(".MI"):
        variants = [f"{base}.L", f"{base}.DE", f"{base}.AS", base]
    elif ticker.endswith(".L"):
        variants = [f"{base}.MI", f"{base}.DE", f"{base}.AS", base]
    elif ticker.endswith(".DE"):
        variants = [f"{base}.L", f"{base}.MI", base]
    else:
        variants = [f"{base}.MI", f"{base}.L", f"{base}.DE"]

    for v in variants:
        if v == ticker:
            continue
        s = _fetch_yfinance(v, period)
        if s is not None:
            return s
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

_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{10}$")


def _looks_like_isin(token: str) -> bool:
    return bool(_ISIN_RE.match(token.upper()))


def _looks_like_ticker(token: str) -> bool:
    """Ticker = non-ISIN e contiene lettere (con possibile suffisso .MI .DE .L ecc.)."""
    return not _looks_like_isin(token) and bool(re.match(r"^[A-Z0-9]{1,6}(\.[A-Z]{1,3})?$", token.upper()))


def _resolve_ticker(isin: str, provided_ticker: str | None) -> list[str]:
    """
    Ritorna lista di ticker da provare su yfinance per un dato ISIN/token.
    Priorità:
      1. ticker fornito esplicitamente
      2. ISIN_TO_TICKER map (ETF hardcoded)
      3. Se il token stesso sembra un ticker → usalo direttamente
      4. Varianti comuni (ISIN.MI, ISIN.L, ISIN.DE)
    """
    from .etf_tickers import ISIN_TO_TICKER

    candidates = []
    token = isin.strip().upper()

    if provided_ticker:
        candidates.append(provided_ticker)

    # ETF map
    if token in ISIN_TO_TICKER:
        t = ISIN_TO_TICKER[token]
        if t not in candidates:
            candidates.append(t)

    # Token sembra già un ticker (es. "AAPL", "ENI.MI")
    if _looks_like_ticker(token):
        if token not in candidates:
            candidates.append(token)
        # Prova anche versione con .MI per azioni italiane
        if "." not in token:
            candidates.extend([f"{token}.MI", f"{token}.L", f"{token}.DE"])

    # ETF europei: ISIN funziona spesso come ticker su Yahoo con suffisso .MI
    if _looks_like_isin(token) and token not in ISIN_TO_TICKER:
        candidates.extend([
            f"{token}.MI", f"{token}.L", f"{token}.DE", f"{token}.F"
        ])

    # Rimuovi duplicati mantenendo ordine
    seen = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


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
    Recupera serie storica prezzi/NAV per un ISIN, ticker o codice qualsiasi.

    Cascata per ISIN di fondo:
      cache → Morningstar → FondiDoc → yfinance (ticker map) → sintetica

    Cascata per ticker (azioni, ETF, obbligazioni quotate):
      cache → yfinance (ticker diretto) → sintetica

    Cascata per ISIN ETF noto:
      cache → yfinance (ISIN_TO_TICKER map) → JustETF scraping → sintetica
    """
    cache = _load_cache()
    cache_key = isin.strip().upper()
    if cache_key in cache and _cache_valid(cache[cache_key]):
        entry = cache[cache_key]
        series_data = entry.get("series")
        if series_data:
            idx = pd.to_datetime(list(series_data.keys()))
            vals = list(series_data.values())
            return pd.Series(vals, index=idx, name=cache_key)

    series = None
    token = cache_key
    is_pure_isin = _looks_like_isin(token)
    is_ticker_input = _looks_like_ticker(token)

    # ── Percorso A: ticker diretto (azioni, indici, ETF con ticker) ──────────
    if is_ticker_input or (ticker and not is_pure_isin):
        yf_ticker = ticker or token
        series = _fetch_yfinance(yf_ticker, period=period)
        # Se non trova con suffisso, prova varianti
        if series is None and "." not in yf_ticker:
            for suffix in [".MI", ".L", ".DE", ".F", ".PA"]:
                series = _fetch_yfinance(yf_ticker + suffix, period=period)
                if series is not None:
                    break

    # ── Percorso B: ISIN di fondo → Morningstar/FondiDoc ────────────────────
    if series is None and is_pure_isin:
        series = _fetch_morningstar(token)
        if series is None:
            series = _fetch_fondidoc(token)
            time.sleep(0.2)

    # ── Percorso C: yfinance con ticker risolti (ETF map + varianti mercato) ──
    if series is None:
        for try_ticker in _resolve_ticker(token, ticker):
            # Usa la versione con fallback .MI/.L/.DE automatico
            series = _fetch_yfinance_with_fallbacks(try_ticker, period=period)
            if series is not None:
                break
            time.sleep(0.1)

    # ── Percorso D: serie sintetica da rendimenti annuali ────────────────────
    if series is None:
        series = _synthetic_series_from_returns(
            perf_1y=perf_1y, perf_3y=perf_3y, perf_ytd=perf_ytd,
            perf_2022=perf_2022, perf_2023=perf_2023, perf_2024=perf_2024,
        )

    if series is not None and not series.empty:
        cache[cache_key] = {
            "timestamp": datetime.now().isoformat(),
            "series": {str(k): float(v) for k, v in series.items()
                       if not (isinstance(v, float) and np.isnan(v))},
        }
        _save_cache(cache)

    return series


def get_multiple_nav(assets: list[dict], period: str = "3y") -> dict[str, pd.Series]:
    """
    Recupera serie per lista di asset.
    Ogni elemento: {"isin": ..., "ticker": ..., "perf_1y": ..., ...}

    Riconosce automaticamente:
    - ISIN fondi (12 char alfanumerici) → Morningstar/FondiDoc/sintetica
    - ISIN ETF noti → yfinance via ticker map
    - Ticker azioni/indici ("AAPL", "ENI.MI", "BTP5.MI") → yfinance diretto
    """
    result = {}
    for asset in assets:
        isin = str(asset.get("isin", "")).strip()
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
        if series is not None and len(series) >= 6:
            result[isin] = series
    return result


def classify_asset_type(token: str) -> str:
    """
    Classifica un token come tipo di asset per mostrare info utente.
    Ritorna: 'ETF', 'Fondo', 'Azione/Obbligazione', 'Sconosciuto'
    """
    from .etf_tickers import ISIN_TO_TICKER
    t = token.strip().upper()
    if t in ISIN_TO_TICKER:
        return "ETF (ticker noto)"
    if _looks_like_ticker(t):
        return "Azione / ETF ticker"
    if _looks_like_isin(t):
        return "Fondo / ISIN generico"
    return "Sconosciuto"
