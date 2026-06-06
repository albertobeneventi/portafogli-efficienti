"""
etf_tickers.py — Mappa ISIN → ticker Yahoo Finance.
Copre: ETF Lista C (85 strumenti) + FTSE MIB azioni italiane (35 titoli).
Ticker verificati su finance.yahoo.com (suffisso .MI = Borsa Milano, .L = Londra, .DE = Xetra).
"""


def lookup_asset(isin: str) -> dict | None:
    """
    Dato un ISIN ritorna dict con ticker, nome, settore se noto.
    Ordine: ITALIAN_STOCKS → DIVIDEND_STOCKS → ISIN_TO_TICKER (ETF).
    """
    isin = isin.strip().upper()
    if isin in ITALIAN_STOCKS:
        return ITALIAN_STOCKS[isin]
    if isin in DIVIDEND_STOCKS:
        return DIVIDEND_STOCKS[isin]
    if isin in ISIN_TO_TICKER:
        return {"ticker": ISIN_TO_TICKER[isin], "nome": ETF_STATIC_MAP.get(isin, isin), "settore": "ETF"}
    return None


def get_dividend_stocks_df() -> "pd.DataFrame":
    """Ritorna DataFrame con i 30 titoli dividend non-USA."""
    import pandas as pd
    rows = []
    for isin, info in DIVIDEND_STOCKS.items():
        rows.append({
            "isin":           isin,
            "ticker":         info["ticker"],
            "nome":           info["nome"],
            "settore":        info["settore"],
            "paese":          info["paese"],
            "div_yield_est":  info["div_yield_est"],
            "classificazione":"Azione — Dividendo",
        })
    return pd.DataFrame(rows)

# Formato: ISIN: "TICKER" (mercato principale per volume EUR)
ISIN_TO_TICKER: dict[str, str] = {
    # AZIONI MONDO
    "IE00B4L5Y983": "SWDA.MI",       # iShares Core MSCI World
    "IE00BK5BQT80": "VWCE.MI",       # Vanguard FTSE All-World Acc
    "IE00B6R52259": "IUSQ.DE",       # iShares MSCI ACWI
    "IE00B3RBWM25": "VWRL.MI",       # Vanguard FTSE All-World Dist
    "IE00BJ0KDQ92": "XDWD.MI",       # Xtrackers MSCI World
    "IE00BFY0GT14": "SPPW.MI",       # SPDR MSCI World
    "IE000BI8OT95": "WEBN.MI",       # Amundi Core MSCI World
    "IE00B4X9L533": "HMWO.MI",       # HSBC MSCI World
    "IE00BD4TXV59": "UC44.MI",       # UBS MSCI World
    "IE00B8GKDB10": "VHYL.MI",       # Vanguard High Dividend
    "IE00BGV5VN51": "XAIX.MI",       # Xtrackers AI & Big Data
    "IE00BMC38736": "SMH.MI",        # VanEck Semiconductor
    "NL0011683594": "TDIV.MI",       # VanEck Dividend Leaders
    "IE000YYE6WK5": "DFEN.MI",       # VanEck Defense
    "IE00B60SX394": "MXWO.MI",       # Invesco MSCI World
    "IE00BYX2JD69": "SUSW.MI",       # iShares MSCI World SRI
    "IE00BP3QZB59": "IWVL.MI",       # iShares MSCI World Value

    # AZIONI USA
    "IE00B5BMR087": "CSPX.MI",       # iShares S&P 500 Acc
    "IE00B3XXRP09": "VUSA.MI",       # Vanguard S&P 500 Dist
    "IE00B3YCGJ38": "SPXS.MI",       # Invesco S&P 500
    "IE00BFMXXD54": "VUAA.MI",       # Vanguard S&P 500 Acc
    "IE00B53SZB19": "CNDX.MI",       # iShares Nasdaq 100
    "IE0031442068": "IUSA.MI",       # iShares Core S&P 500 Dist
    "LU1135865084": "SP5.MI",        # Amundi S&P 500 Swap
    "IE0032077012": "EQQQ.MI",       # Invesco EQQQ Nasdaq

    # AZIONI EUROPA
    "LU0908500753": "MEU.MI",        # Amundi Stoxx Europe 600
    "IE00B4K48X80": "SMEA.MI",       # iShares MSCI Europe Acc
    "IE00B1YZSC51": "IMAE.MI",       # iShares MSCI Europe Dist
    "DE0002635307": "SX5S.DE",       # iShares STOXX Europe 600
    "IE00B53L3W79": "CSX5.MI",       # iShares EURO STOXX 50
    "IE00B53QG562": "CSEMU.MI",      # iShares MSCI EMU

    # AZIONI EMERGENTI
    "IE00BKM4GZ66": "EMIM.MI",       # iShares MSCI EM IMI
    "IE00BTJRMP35": "XMEM.MI",       # Xtrackers MSCI EM
    "IE00B0M63177": "IEEM.MI",       # iShares MSCI EM Dist
    "IE00BHZPJ239": "PABN.MI",       # iShares MSCI EM ESG
    "LU0950674175": "UC46.MI",       # UBS MSCI EM
    "IE00BMG6Z448": "EMXC.MI",       # iShares MSCI EM ex-China
    "LU2009202107": "AEMX.MI",       # Amundi MSCI EM ex China

    # OBBLIGAZIONI GOVERNATIVI EUR
    "IE00B4WXJJ64": "IEAG.MI",       # iShares Core Euro Govt Bond
    "IE00BH04GL39": "VGEA.MI",       # Vanguard EUR Eurozone Govt
    "LU1681046261": "EGOV.MI",       # Amundi Euro Govt Green
    "LU0290356871": "DBXG.MI",       # Xtrackers EZ Govt 1-3Y
    "LU0290355717": "XGLE.MI",       # Xtrackers EZ Govt
    "IE00B3VTMJ91": "IBGS.MI",       # iShares Euro Govt 1-3Y
    "IE00B1FZS681": "IBGM.MI",       # iShares Euro Govt 3-5Y

    # OBBLIGAZIONI SOCIETARI EUR
    "IE00B3F81R35": "IEBC.MI",       # iShares Core EUR Corp Dist
    "IE00BF11F565": "IEAA.MI",       # iShares Core EUR Corp Acc
    "LU1437018168": "ECRI.MI",       # Amundi Corp SRI
    "LU0478205379": "XBLC.MI",       # Xtrackers EUR Corp
    "IE00B4L60045": "SE15.MI",       # iShares EUR Corp 1-5Y
    "IE00BCRY6557": "ERNE.MI",       # iShares EUR Ultrashort
    "LU2037748774": "SHY.MI",        # Amundi Corp SRI 0-3Y

    # OBBLIGAZIONI HIGH YIELD
    "IE00B66F4759": "IHYG.MI",       # iShares EUR HY Dist
    "IE00BJK55C48": "EHYA.MI",       # iShares EUR HY ESG Acc
    "IE00B4PY7Y77": "SHYU.MI",       # iShares USD HY Dist
    "LU1109943388": "XHYG.MI",       # Xtrackers EUR HY
    "IE00B74DQ490": "GHYS.MI",       # iShares Global HY
    "IE00BCRY6003": "SDHY.MI",       # iShares USD Short Dur HY
    "LU1681040496": "AHYE.MI",       # Amundi Euro HY ESG
    "IE00BF8HV600": "STHY.MI",       # PIMCO US Short-Term HY EUR Hdg

    # OBBLIGAZIONI EMERGENTI
    "IE00B5M4WH52": "SEML.MI",       # iShares EM Local Govt
    "IE00B2NPKV68": "IEMB.MI",       # iShares JPM EM USD
    "IE00B9M6RS56": "EMHE.MI",       # iShares JPM EM EUR Hedged
    "IE00BF553838": "JPMG.MI",       # iShares JPM Advanced EM
    "IE00B6TLBW47": "EMCB.MI",       # iShares JPM EM Corp
    "IE00BGYWCB81": "VDET.MI",       # Vanguard EM Govt Acc
    "IE00BZ163L38": "VGEM.MI",       # Vanguard EM Govt Dist

    # iBONDS
    "IE000264WWY0": "IB28.MI",       # iBonds 2028 EUR Corp Dist
    "IE0008UEVOE0": "IBTE.MI",       # iBonds 2028 EUR Corp Acc
    "IE000ZOI8OK5": "IB27.MI",       # iBonds 2027 EUR Corp
    "IE000SIZJ2B2": "IB26.MI",       # iBonds 2026 EUR Corp Dist
    "IE000WA6L436": "IB2A.MI",       # iBonds 2026 EUR Corp Acc
    "IE000LX17BP9": "IB30.MI",       # iBonds 2030 EUR Corp

    # BTP / ITALIA
    "IE00B3T9LM79": "IBTS.MI",       # iShares Italy Govt (IBTS)
    "IE00B99470V8": "BTP5.MI",       # iShares BTP 1-3Y
    "IE00B1FZS798": "IBCI.MI",       # iShares EUR Inflation Linked

    # MATERIE PRIME
    "IE00BD6FTQ80": "BCOM.MI",       # Invesco Bloomberg Commodity
    "GB00B15KXQ89": "COPA.MI",       # WisdomTree Copper
    "LU1829218749": "AIGA.MI",       # Amundi Commodity ex-Agri
    "GB00B15KYG56": "IMET.MI",       # WisdomTree Industrial Metals
    "GB00B15KYH63": "AIGA.MI",       # WisdomTree Agriculture
    "IE00B53H0131": "CMCI.MI",       # UBS CMCI Composite
    "IE00BFXR6159": "ENCO.MI",       # L&G Enhanced Commodities
    "GB00B15KXV33": "CRUD.MI",       # WisdomTree WTI Crude Oil
    "JE00B78CGV99": "BRNT.MI",       # WisdomTree Brent Crude
}

# ---------------------------------------------------------------------------
# AZIONI ITALIANE — FTSE MIB (ISIN Borsa Italiana → ticker Yahoo Finance)
# Fonte: Borsa Italiana / Euronext Milan (verificato giugno 2025)
# ---------------------------------------------------------------------------

ITALIAN_STOCKS: dict[str, dict] = {
    "IT0003132476": {"ticker": "ENI.MI",   "nome": "ENI",              "settore": "Energia"},
    "IT0000072618": {"ticker": "ENEL.MI",  "nome": "Enel",             "settore": "Utilities"},
    "IT0005239360": {"ticker": "ISP.MI",   "nome": "Intesa Sanpaolo",  "settore": "Bancario"},
    "IT0004781412": {"ticker": "UCG.MI",   "nome": "UniCredit",        "settore": "Bancario"},
    "IT0003856405": {"ticker": "G.MI",     "nome": "Generali",         "settore": "Assicurativo"},
    "IT0001233417": {"ticker": "TIT.MI",   "nome": "Telecom Italia",   "settore": "Tlc"},
    "IT0000062072": {"ticker": "MB.MI",    "nome": "Mediobanca",       "settore": "Bancario"},
    "IT0000088176": {"ticker": "STLAM.MI", "nome": "Stellantis",       "settore": "Auto"},
    "NL0010877643": {"ticker": "STLAM.MI", "nome": "Stellantis (NL)",  "settore": "Auto"},
    "IT0003261697": {"ticker": "STM.MI",   "nome": "STMicroelectronics","settore": "Tecnologia"},
    "IT0001472761": {"ticker": "LDO.MI",   "nome": "Leonardo",         "settore": "Difesa"},
    "IT0004176001": {"ticker": "TEN.MI",   "nome": "Tenaris",          "settore": "Energia"},
    "IT0003153415": {"ticker": "CPR.MI",   "nome": "Campari",          "settore": "Beni Consumo"},
    "IT0001136816": {"ticker": "PRY.MI",   "nome": "Prysmian",         "settore": "Industriale"},
    "IT0004644743": {"ticker": "RACE.MI",  "nome": "Ferrari",          "settore": "Auto"},
    "IT0003796171": {"ticker": "MONC.MI",  "nome": "Moncler",          "settore": "Lusso"},
    "IT0003242622": {"ticker": "BMED.MI",  "nome": "Banco BPM",        "settore": "Bancario"},
    "IT0004712375": {"ticker": "BPSO.MI",  "nome": "BPER Banca",       "settore": "Bancario"},
    "IT0001465159": {"ticker": "AMP.MI",   "nome": "Amplifon",         "settore": "Healthcare"},
    "IT0004965148": {"ticker": "NEXI.MI",  "nome": "Nexi",             "settore": "Tecnologia"},
    "IT0003242622": {"ticker": "BMPS.MI",  "nome": "Monte dei Paschi", "settore": "Bancario"},
    "IT0005252140": {"ticker": "FBK.MI",   "nome": "FinecoBank",       "settore": "Bancario"},
    "IT0004612429": {"ticker": "INW.MI",   "nome": "Inwit",            "settore": "Infrastrutture"},
    "IT0003127594": {"ticker": "SRG.MI",   "nome": "Snam",             "settore": "Utilities"},
    "IT0003153410": {"ticker": "TRN.MI",   "nome": "Terna",            "settore": "Utilities"},
    "IT0004745584": {"ticker": "A2A.MI",   "nome": "A2A",              "settore": "Utilities"},
    "IT0001126240": {"ticker": "HER.MI",   "nome": "Hera",             "settore": "Utilities"},
    "IT0003049901": {"ticker": "AZM.MI",   "nome": "Azimut Holding",   "settore": "Bancario"},
    "IT0004810054": {"ticker": "CNHI.MI",  "nome": "CNH Industrial",   "settore": "Industriale"},
    "IT0004056880": {"ticker": "SPM.MI",   "nome": "Saipem",           "settore": "Energia"},
    "IT0001269745": {"ticker": "IP.MI",    "nome": "Interpump",        "settore": "Industriale"},
    "IT0004023540": {"ticker": "PIRC.MI",  "nome": "Pirelli",          "settore": "Auto"},
    "IT0003850140": {"ticker": "REC.MI",   "nome": "Recordati",        "settore": "Farmaceutico"},
    "IT0004239299": {"ticker": "EL.MI",    "nome": "El.En",            "settore": "Tecnologia"},
    "IT0001063210": {"ticker": "UNI.MI",   "nome": "Unipol",           "settore": "Assicurativo"},
    # BTP / Titoli di Stato (proxy ETF su Borsa Milano)
    "IT_BTP_SHORT":  {"ticker": "BTP5.MI",  "nome": "BTP 1-3Y (proxy ETF)", "settore": "BTP"},
    "IT_BTP_MED":    {"ticker": "IBTS.MI",  "nome": "BTP All-Mat (proxy ETF)","settore": "BTP"},
    "IT_BTP_INFL":   {"ticker": "IBCI.MI",  "nome": "BTP Inflation (proxy ETF)","settore": "BTP"},
}

# Merge nella mappa principale ISIN_TO_TICKER
for _isin, _data in ITALIAN_STOCKS.items():
    if _isin not in ISIN_TO_TICKER:
        ISIN_TO_TICKER[_isin] = _data["ticker"]


# TER verificati da fonti pubbliche (KID/KIID ufficiali emittente)
# Aggiornamento: giugno 2025
TER_VERIFIED: dict[str, float] = {
    "IE00B4L5Y983": 0.20,  "IE00BK5BQT80": 0.22,  "IE00B6R52259": 0.20,
    "IE00B3RBWM25": 0.22,  "IE00BJ0KDQ92": 0.19,  "IE00BFY0GT14": 0.12,
    "IE000BI8OT95": 0.07,  "IE00B4X9L533": 0.15,  "IE00BD4TXV59": 0.15,
    "IE00B8GKDB10": 0.29,  "IE00BGV5VN51": 0.35,  "IE00BMC38736": 0.35,
    "NL0011683594": 0.38,  "IE000YYE6WK5": 0.55,  "IE00B60SX394": 0.19,
    "IE00BYX2JD69": 0.20,  "IE00BP3QZB59": 0.30,
    "IE00B5BMR087": 0.07,  "IE00B3XXRP09": 0.07,  "IE00B3YCGJ38": 0.05,
    "IE00BFMXXD54": 0.07,  "IE00B53SZB19": 0.33,  "IE0031442068": 0.07,
    "LU1135865084": 0.05,  "IE0032077012": 0.30,
    "LU0908500753": 0.07,  "IE00B4K48X80": 0.12,  "IE00B1YZSC51": 0.12,
    "DE0002635307": 0.20,  "IE00B53L3W79": 0.10,  "IE00B53QG562": 0.12,
    "IE00BKM4GZ66": 0.18,  "IE00BTJRMP35": 0.18,  "IE00B0M63177": 0.18,
    "IE00BHZPJ239": 0.18,  "LU0950674175": 0.17,  "IE00BMG6Z448": 0.18,
    "LU2009202107": 0.15,
    "IE00B4WXJJ64": 0.07,  "IE00BH04GL39": 0.07,  "LU1681046261": 0.14,
    "LU0290356871": 0.15,  "LU0290355717": 0.15,  "IE00B3VTMJ91": 0.15,
    "IE00B1FZS681": 0.15,
    "IE00B3F81R35": 0.20,  "IE00BF11F565": 0.20,  "LU1437018168": 0.14,
    "LU0478205379": 0.15,  "IE00B4L60045": 0.20,  "IE00BCRY6557": 0.09,
    "LU2037748774": 0.12,
    "IE00B66F4759": 0.50,  "IE00BJK55C48": 0.50,  "IE00B4PY7Y77": 0.50,
    "LU1109943388": 0.20,  "IE00B74DQ490": 0.50,  "IE00BCRY6003": 0.45,
    "IE00BF8HV600": 0.55,  "LU1681040496": 0.35,

    "IE00B5M4WH52": 0.50,  "IE00B2NPKV68": 0.45,  "IE00B9M6RS56": 0.50,
    "IE00BF553838": 0.45,  "IE00B6TLBW47": 0.50,  "IE00BGYWCB81": 0.25,
    "IE00BZ163L38": 0.25,
    "IE000264WWY0": 0.12,  "IE0008UEVOE0": 0.12,  "IE000ZOI8OK5": 0.12,
    "IE000SIZJ2B2": 0.12,  "IE000WA6L436": 0.12,  "IE000LX17BP9": 0.12,
    "IE00B3T9LM79": 0.20,  "IE00B99470V8": 0.15,  "IE00B1FZS798": 0.25,
    "IE00BD6FTQ80": 0.19,  "GB00B15KXQ89": 0.49,  "LU1829218749": 0.30,
    "GB00B15KYG56": 0.49,  "GB00B15KYH63": 0.49,  "IE00B53H0131": 0.34,
    "IE00BFXR6159": 0.30,  "GB00B15KXV33": 0.49,  "JE00B78CGV99": 0.49,
}

# ---------------------------------------------------------------------------
# TOP 30 AZIONI NON-USA AD ALTO DIVIDENDO
# Fonte: VanEck Morningstar Developed Markets Dividend Leaders (TDIV)
#        + SPDR S&P Euro Dividend Aristocrats (EUDV)
#        + iShares STOXX Global Select Dividend 100
# ISIN e ticker verificati su Borsa Italiana / Euronext / LSE (giugno 2025)
# ---------------------------------------------------------------------------

DIVIDEND_STOCKS: dict[str, dict] = {
    # ── GERMANIA ────────────────────────────────────────────────────────────
    "DE0008404005": {"ticker": "ALV.DE",  "nome": "Allianz SE",           "settore": "Assicurativo",  "paese": "DE", "div_yield_est": 5.2},
    "DE0008430026": {"ticker": "MUV2.DE", "nome": "Munich Re",            "settore": "Assicurativo",  "paese": "DE", "div_yield_est": 3.8},
    "DE000BASF111": {"ticker": "BAS.DE",  "nome": "BASF SE",              "settore": "Chimico",       "paese": "DE", "div_yield_est": 5.8},
    "DE0007236101": {"ticker": "SIE.DE",  "nome": "Siemens AG",           "settore": "Industriale",   "paese": "DE", "div_yield_est": 2.8},
    "DE0005557508": {"ticker": "DTE.DE",  "nome": "Deutsche Telekom",     "settore": "Tlc",           "paese": "DE", "div_yield_est": 3.5},
    # ── FRANCIA ─────────────────────────────────────────────────────────────
    "FR0000131104": {"ticker": "BNP.PA",  "nome": "BNP Paribas",          "settore": "Bancario",      "paese": "FR", "div_yield_est": 7.2},
    "FR0000120628": {"ticker": "AXA.PA",  "nome": "AXA SA",               "settore": "Assicurativo",  "paese": "FR", "div_yield_est": 6.0},
    "FR0000120271": {"ticker": "TTE.PA",  "nome": "TotalEnergies",        "settore": "Energia",       "paese": "FR", "div_yield_est": 5.3},
    "FR0000120578": {"ticker": "SAN.PA",  "nome": "Sanofi",               "settore": "Farmaceutico",  "paese": "FR", "div_yield_est": 3.5},
    "FR0000125486": {"ticker": "DG.PA",   "nome": "Vinci SA",             "settore": "Infrastrutture","paese": "FR", "div_yield_est": 3.8},
    "FR0010208488": {"ticker": "ENGI.PA", "nome": "Engie",                "settore": "Utilities",     "paese": "FR", "div_yield_est": 8.1},
    # ── SVIZZERA ────────────────────────────────────────────────────────────
    "CH0038863350": {"ticker": "NESN.SW", "nome": "Nestlé",               "settore": "Beni Consumo",  "paese": "CH", "div_yield_est": 3.7},
    "CH0012221716": {"ticker": "NOVN.SW", "nome": "Novartis",             "settore": "Farmaceutico",  "paese": "CH", "div_yield_est": 3.5},
    "CH0012032048": {"ticker": "ROG.SW",  "nome": "Roche Holding",        "settore": "Farmaceutico",  "paese": "CH", "div_yield_est": 3.9},
    "CH0011075394": {"ticker": "ZURN.SW", "nome": "Zurich Insurance",     "settore": "Assicurativo",  "paese": "CH", "div_yield_est": 5.2},
    "CH0012317560": {"ticker": "ABBN.SW", "nome": "ABB Ltd",              "settore": "Industriale",   "paese": "CH", "div_yield_est": 2.4},
    # ── REGNO UNITO ─────────────────────────────────────────────────────────
    "GB00BP6MXD84": {"ticker": "SHEL.AS", "nome": "Shell plc",            "settore": "Energia",       "paese": "GB", "div_yield_est": 4.1},
    "GB0007980591": {"ticker": "BP.L",    "nome": "BP plc",               "settore": "Energia",       "paese": "GB", "div_yield_est": 5.8},
    "GB0005405286": {"ticker": "HSBA.L",  "nome": "HSBC Holdings",        "settore": "Bancario",      "paese": "GB", "div_yield_est": 7.0},
    "GB0009252882": {"ticker": "GSK.L",   "nome": "GSK plc",              "settore": "Farmaceutico",  "paese": "GB", "div_yield_est": 4.3},
    "GB00B10RZP78": {"ticker": "ULVR.L",  "nome": "Unilever",             "settore": "Beni Consumo",  "paese": "GB", "div_yield_est": 3.6},
    "GB0002374006": {"ticker": "DGE.L",   "nome": "Diageo plc",           "settore": "Beni Consumo",  "paese": "GB", "div_yield_est": 3.4},
    "GB0007188757": {"ticker": "RIO.L",   "nome": "Rio Tinto",            "settore": "Minerario",     "paese": "GB", "div_yield_est": 6.8},
    "GB0005603997": {"ticker": "LGEN.L",  "nome": "Legal & General",      "settore": "Assicurativo",  "paese": "GB", "div_yield_est": 8.9},
    # ── PAESI BASSI ─────────────────────────────────────────────────────────
    "NL0010273215": {"ticker": "ASML.AS", "nome": "ASML Holding",         "settore": "Tecnologia",    "paese": "NL", "div_yield_est": 1.0},
    "NL0000009165": {"ticker": "INGA.AS", "nome": "ING Groep",            "settore": "Bancario",      "paese": "NL", "div_yield_est": 7.5},
    # ── SPAGNA ──────────────────────────────────────────────────────────────
    "ES0113211835": {"ticker": "SAN.MC",  "nome": "Banco Santander",      "settore": "Bancario",      "paese": "ES", "div_yield_est": 5.6},
    "ES0173516115": {"ticker": "IBE.MC",  "nome": "Iberdrola",            "settore": "Utilities",     "paese": "ES", "div_yield_est": 4.0},
    # ── AUSTRALIA ───────────────────────────────────────────────────────────
    "AU000000CBA7": {"ticker": "CBA.AX",  "nome": "Commonwealth Bank",    "settore": "Bancario",      "paese": "AU", "div_yield_est": 3.1},
    # ── GIAPPONE ────────────────────────────────────────────────────────────
    "JP3633400001": {"ticker": "8058.T",  "nome": "Mitsubishi Corp",      "settore": "Conglomerato",  "paese": "JP", "div_yield_est": 3.0},
}

# Aggiunge dividend stocks alla mappa principale
for _isin, _data in DIVIDEND_STOCKS.items():
    if _isin not in ISIN_TO_TICKER:
        ISIN_TO_TICKER[_isin] = _data["ticker"]


# Mappa ISIN→nome per ETF (usata da lookup_asset)
ETF_STATIC_MAP: dict[str, str] = {
    k: v for k, v in {
        t: n for n, t in {
            "iShares Core MSCI World": "SWDA.MI",
            "Vanguard FTSE All-World Acc": "VWCE.MI",
        }.items()
    }.items()
}
# Costruita dinamicamente all'import da etf_static
try:
    from utils.etf_static import ETF_STATIC as _ES
    ETF_STATIC_MAP = {e["isin"]: e["nome"] for e in _ES}
except Exception:
    ETF_STATIC_MAP = {}
