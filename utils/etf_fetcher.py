"""
etf_fetcher.py — scraping JustETF e gestione etf_universe.xlsx + cache.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ETF_UNIVERSE_FILE = DATA_DIR / "etf_universe.xlsx"
ETF_CACHE_FILE = DATA_DIR / "etf_cache.json"
ETF_ERRORS_LOG = DATA_DIR / "etf_errors.log"
CACHE_TTL_HOURS = 24

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

JUSTETF_URL = "https://www.justetf.com/it/etf-profile.html?isin={isin}"

HARDCODED_ISINS = [
    # Azioni - Mondo
    "IE00B4L5Y983", "IE00BK5BQT80", "IE00B6R52259", "IE00B3RBWM25",
    "IE00BJ0KDQ92", "IE00BFY0GT14", "IE000BI8OT95", "IE00B4X9L533",
    "IE00BD4TXV59", "IE00B8GKDB10", "IE00BGV5VN51", "IE00BMC38736",
    "NL0011683594", "IE000YYE6WK5", "IE00B60SX394", "IE00BYX2JD69",
    "IE00BP3QZB59",
    # Azioni - USA
    "IE00B5BMR087", "IE00B3XXRP09", "IE00B3YCGJ38", "IE00BFMXXD54",
    "IE00B53SZB19", "IE0031442068", "LU1135865084", "IE0032077012",
    # Azioni - Europa
    "LU0908500753", "IE00B4K48X80", "IE00B1YZSC51", "DE0002635307",
    "IE00B53L3W79", "IE00B53QG562",
    # Azioni - Mercati Emergenti
    "IE00BKM4GZ66", "IE00BTJRMP35", "IE00B0M63177", "IE00BHZPJ239",
    "LU0950674175", "IE00BMG6Z448", "LU2009202107",
    # Obbligazioni - Governativi Euro
    "IE00B4WXJJ64", "IE00BH04GL39", "LU1681046261", "LU0290356871",
    "LU0290355717", "IE00B3VTMJ91", "IE00B1FZS681",
    # Obbligazioni - Societari Euro
    "IE00B3F81R35", "IE00BF11F565", "LU1437018168", "LU0478205379",
    "IE00B4L60045", "IE00BCRY6557", "LU2037748774",
    # Obbligazioni - High Yield
    "IE00B66F4759", "IE00BJK55C48", "IE00B4PY7Y77", "LU1109943388",
    "IE00B74DQ490", "IE00BCRY6003", "IE00BF8HV600", "LU1681040496",
    # Obbligazioni - Mercati Emergenti
    "IE00B5M4WH52", "IE00B2NPKV68", "IE00B9M6RS56", "IE00BF553838",
    "IE00B6TLBW47", "IE00BGYWCB81", "IE00BZ163L38",
    # iBonds
    "IE000264WWY0", "IE0008UEVOE0", "IE000ZOI8OK5", "IE000SIZJ2B2",
    "IE000WA6L436", "IE000LX17BP9",
    # BTP/Italia
    "IE00B3T9LM79", "IE00B99470V8", "IE00B1FZS798",
    # Materie prime
    "IE00BD6FTQ80", "GB00B15KXQ89", "LU1829218749", "GB00B15KYG56",
    "GB00B15KYH63", "IE00B53H0131", "IE00BFXR6159",
    "GB00B15KXV33", "JE00B78CGV99",
]

CATEGORY_MAP = {
    "IE00B4L5Y983": "Azioni Mondo", "IE00BK5BQT80": "Azioni Mondo",
    "IE00B6R52259": "Azioni Mondo", "IE00B3RBWM25": "Azioni Mondo",
    "IE00BJ0KDQ92": "Azioni Mondo", "IE00BFY0GT14": "Azioni Mondo",
    "IE000BI8OT95": "Azioni Mondo", "IE00B4X9L533": "Azioni Mondo",
    "IE00BD4TXV59": "Azioni Mondo", "IE00B8GKDB10": "Azioni Mondo",
    "IE00BGV5VN51": "Azioni Mondo", "IE00BMC38736": "Azioni Mondo",
    "NL0011683594": "Azioni Mondo", "IE000YYE6WK5": "Azioni Mondo",
    "IE00B60SX394": "Azioni Mondo", "IE00BYX2JD69": "Azioni Mondo",
    "IE00BP3QZB59": "Azioni Mondo",
    "IE00B5BMR087": "Azioni USA", "IE00B3XXRP09": "Azioni USA",
    "IE00B3YCGJ38": "Azioni USA", "IE00BFMXXD54": "Azioni USA",
    "IE00B53SZB19": "Azioni USA", "IE0031442068": "Azioni USA",
    "LU1135865084": "Azioni USA", "IE0032077012": "Azioni USA",
    "LU0908500753": "Azioni Europa", "IE00B4K48X80": "Azioni Europa",
    "IE00B1YZSC51": "Azioni Europa", "DE0002635307": "Azioni Europa",
    "IE00B53L3W79": "Azioni Europa", "IE00B53QG562": "Azioni Europa",
    "IE00BKM4GZ66": "Azioni Emergenti", "IE00BTJRMP35": "Azioni Emergenti",
    "IE00B0M63177": "Azioni Emergenti", "IE00BHZPJ239": "Azioni Emergenti",
    "LU0950674175": "Azioni Emergenti", "IE00BMG6Z448": "Azioni Emergenti",
    "LU2009202107": "Azioni Emergenti",
    "IE00B4WXJJ64": "Obbligazioni Governativi EUR",
    "IE00BH04GL39": "Obbligazioni Governativi EUR",
    "LU1681046261": "Obbligazioni Governativi EUR",
    "LU0290356871": "Obbligazioni Governativi EUR",
    "LU0290355717": "Obbligazioni Governativi EUR",
    "IE00B3VTMJ91": "Obbligazioni Governativi EUR",
    "IE00B1FZS681": "Obbligazioni Governativi EUR",
    "IE00B3F81R35": "Obbligazioni Societari EUR",
    "IE00BF11F565": "Obbligazioni Societari EUR",
    "LU1437018168": "Obbligazioni Societari EUR",
    "LU0478205379": "Obbligazioni Societari EUR",
    "IE00B4L60045": "Obbligazioni Societari EUR",
    "IE00BCRY6557": "Obbligazioni Societari EUR",
    "LU2037748774": "Obbligazioni Societari EUR",
    "IE00B66F4759": "Obbligazioni High Yield",
    "IE00BJK55C48": "Obbligazioni High Yield",
    "IE00B4PY7Y77": "Obbligazioni High Yield",
    "LU1109943388": "Obbligazioni High Yield",
    "IE00B74DQ490": "Obbligazioni High Yield",
    "IE00BCRY6003": "Obbligazioni High Yield",
    "IE00BF8HV600": "Obbligazioni High Yield",
    "LU1681040496": "Obbligazioni High Yield",
    "IE00B5M4WH52": "Obbligazioni Emergenti",
    "IE00B2NPKV68": "Obbligazioni Emergenti",
    "IE00B9M6RS56": "Obbligazioni Emergenti",
    "IE00BF553838": "Obbligazioni Emergenti",
    "IE00B6TLBW47": "Obbligazioni Emergenti",
    "IE00BGYWCB81": "Obbligazioni Emergenti",
    "IE00BZ163L38": "Obbligazioni Emergenti",
    "IE000264WWY0": "iBonds", "IE0008UEVOE0": "iBonds",
    "IE000ZOI8OK5": "iBonds", "IE000SIZJ2B2": "iBonds",
    "IE000WA6L436": "iBonds", "IE000LX17BP9": "iBonds",
    "IE00B3T9LM79": "BTP/Italia", "IE00B99470V8": "BTP/Italia",
    "IE00B1FZS798": "BTP/Italia",
    "IE00BD6FTQ80": "Materie Prime", "GB00B15KXQ89": "Materie Prime",
    "LU1829218749": "Materie Prime", "GB00B15KYG56": "Materie Prime",
    "GB00B15KYH63": "Materie Prime", "IE00B53H0131": "Materie Prime",
    "IE00BFXR6159": "Materie Prime", "GB00B15KXV33": "Materie Prime",
    "JE00B78CGV99": "Materie Prime",
}


def _log_error(isin: str, msg: str):
    DATA_DIR.mkdir(exist_ok=True)
    with open(ETF_ERRORS_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {isin} | {msg}\n")


def _load_cache() -> dict:
    if ETF_CACHE_FILE.exists():
        try:
            with open(ETF_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    DATA_DIR.mkdir(exist_ok=True)
    with open(ETF_CACHE_FILE, "w", encoding="utf-8") as f:
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


def _fetch_justetf(isin: str, session: requests.Session) -> dict | None:
    url = JUSTETF_URL.format(isin=isin)
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 404:
            _log_error(isin, "404 Not Found on JustETF")
            return None
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Nome ETF
        nome = ""
        h1 = soup.find("h1")
        if h1:
            nome = h1.get_text(strip=True)

        # TER
        ter = None
        for label in soup.find_all(string=lambda t: t and "TER" in t):
            parent = label.find_parent()
            if parent:
                nxt = parent.find_next_sibling()
                if nxt:
                    try:
                        ter = float(nxt.get_text(strip=True).replace("%", "").replace(",", "."))
                    except ValueError:
                        pass
                    break

        # AUM
        aum = None
        for label in soup.find_all(string=lambda t: t and ("AUM" in t or "patrimonio" in t.lower())):
            parent = label.find_parent()
            if parent:
                nxt = parent.find_next_sibling()
                if nxt:
                    try:
                        raw = nxt.get_text(strip=True).replace("€", "").replace(".", "").replace(",", ".").strip()
                        # converti in milioni
                        if "mrd" in raw.lower() or "mia" in raw.lower():
                            aum = float(raw.lower().replace("mrd", "").replace("mia", "").strip()) * 1000
                        elif "mln" in raw.lower() or "mil" in raw.lower():
                            aum = float(raw.lower().replace("mln", "").replace("mil", "").strip())
                        else:
                            aum = float(raw) / 1_000_000
                    except ValueError:
                        pass
                    break

        # Rendimenti: cerca tabella con 1J / 3J / 5J
        perf_1y = perf_3y = perf_5y = None
        for td in soup.find_all("td"):
            txt = td.get_text(strip=True)
            # cerca valori percentuali vicini a etichette anno
            try:
                val = float(txt.replace("%", "").replace(",", ".").strip())
                prev = td.find_previous_sibling("td")
                if prev:
                    label_txt = prev.get_text(strip=True).lower()
                    if "1j" in label_txt or "1 j" in label_txt or "1yr" in label_txt:
                        perf_1y = val
                    elif "3j" in label_txt or "3 j" in label_txt or "3yr" in label_txt:
                        perf_3y = val
                    elif "5j" in label_txt or "5 j" in label_txt or "5yr" in label_txt:
                        perf_5y = val
            except ValueError:
                pass

        return {
            "isin": isin,
            "nome": nome or isin,
            "categoria": CATEGORY_MAP.get(isin, "ETF"),
            "ter": ter,
            "aum_mln": aum,
            "perf_1y": perf_1y,
            "perf_3y": perf_3y,
            "perf_5y": perf_5y,
            "timestamp": datetime.now().isoformat(),
        }
    except requests.RequestException as e:
        _log_error(isin, f"Request error: {e}")
        return None


def _fallback_record(isin: str) -> dict:
    """Record con dati statici hardcoded quando JustETF non risponde."""
    from .etf_static import ETF_STATIC
    static = {e["isin"]: e for e in ETF_STATIC}
    base = static.get(isin, {})
    return {
        "isin": isin,
        "nome": base.get("nome", isin),
        "categoria": base.get("categoria", CATEGORY_MAP.get(isin, "ETF")),
        "ter": base.get("ter"),
        "aum_mln": base.get("aum_mln"),
        "perf_1y": None,
        "perf_3y": None,
        "perf_5y": None,
        "timestamp": datetime.now().isoformat(),
    }


def fetch_etf_universe(
    extra_isins: list[str] | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Recupera dati per tutti gli ISIN hardcoded + extra_isins.
    Usa cache 24h; salva su etf_universe.xlsx.
    """
    DATA_DIR.mkdir(exist_ok=True)
    all_isins = list(dict.fromkeys(HARDCODED_ISINS + (extra_isins or [])))
    cache = _load_cache()
    records = []
    session = requests.Session()
    to_fetch = [isin for isin in all_isins if isin not in cache or not _cache_valid(cache[isin])]

    # Carica da cache quelli già validi
    for isin in all_isins:
        if isin in cache and _cache_valid(cache[isin]):
            records.append(cache[isin])

    # Fetch per quelli scaduti o mancanti
    for i, isin in enumerate(to_fetch):
        if progress_callback:
            progress_callback(i, len(to_fetch), isin)
        data = _fetch_justetf(isin, session)
        if data is None:
            data = _fallback_record(isin)
        cache[isin] = data
        records.append(data)
        time.sleep(0.3)  # cortesia verso JustETF

    _save_cache(cache)
    df = pd.DataFrame(records)
    df = df.sort_values("isin").reset_index(drop=True)

    # Aggiunge colonna _lista
    df["_lista"] = "C"
    df.to_excel(ETF_UNIVERSE_FILE, index=False)
    return df


def load_etf_universe(extra_isins: list[str] | None = None) -> pd.DataFrame:
    """
    Carica etf_universe.xlsx se esiste e valido (< 24h),
    altrimenti prova fetch JustETF, fallback su dataset statico.
    """
    if ETF_UNIVERSE_FILE.exists():
        mod_time = datetime.fromtimestamp(ETF_UNIVERSE_FILE.stat().st_mtime)
        if datetime.now() - mod_time < timedelta(hours=CACHE_TTL_HOURS) and not extra_isins:
            try:
                df = pd.read_excel(ETF_UNIVERSE_FILE, dtype=str)
                if not df.empty:
                    return df
            except Exception:
                pass
    try:
        df = fetch_etf_universe(extra_isins=extra_isins)
        # Se tutti i nomi sono ISIN (scraping fallito), usa dataset statico
        isin_as_nome = (df["nome"] == df["isin"]).sum()
        if isin_as_nome > len(df) * 0.5:
            raise ValueError("JustETF scraping non riuscito — uso dataset statico")
        return df
    except Exception:
        from .etf_static import get_static_etf_df
        df = get_static_etf_df()
        DATA_DIR.mkdir(exist_ok=True)
        df.to_excel(ETF_UNIVERSE_FILE, index=False)
        return df
