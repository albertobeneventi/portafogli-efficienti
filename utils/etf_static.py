"""
etf_static.py — Dataset ETF hardcoded come fallback quando JustETF non è raggiungibile.
Dati di riferimento (TER approssimati, aggiornare periodicamente).
"""

ETF_STATIC = [
    # AZIONI MONDO
    {"isin": "IE00B4L5Y983", "nome": "iShares Core MSCI World UCITS ETF USD (Acc)", "categoria": "Azioni Mondo", "ter": 0.20, "aum_mln": 75000},
    {"isin": "IE00BK5BQT80", "nome": "Vanguard FTSE All-World UCITS ETF (USD) Acc", "categoria": "Azioni Mondo", "ter": 0.22, "aum_mln": 25000},
    {"isin": "IE00B6R52259", "nome": "iShares MSCI ACWI UCITS ETF USD (Acc)", "categoria": "Azioni Mondo", "ter": 0.20, "aum_mln": 18000},
    {"isin": "IE00B3RBWM25", "nome": "Vanguard FTSE All-World UCITS ETF (USD) Dist", "categoria": "Azioni Mondo", "ter": 0.22, "aum_mln": 12000},
    {"isin": "IE00BJ0KDQ92", "nome": "Xtrackers MSCI World UCITS ETF 1C", "categoria": "Azioni Mondo", "ter": 0.19, "aum_mln": 9000},
    {"isin": "IE00BFY0GT14", "nome": "SPDR MSCI World UCITS ETF USD Unhedged", "categoria": "Azioni Mondo", "ter": 0.12, "aum_mln": 7000},
    {"isin": "IE000BI8OT95", "nome": "Amundi Core MSCI World UCITS ETF Acc", "categoria": "Azioni Mondo", "ter": 0.07, "aum_mln": 5000},
    {"isin": "IE00B4X9L533", "nome": "HSBC MSCI World UCITS ETF USD", "categoria": "Azioni Mondo", "ter": 0.15, "aum_mln": 4500},
    {"isin": "IE00BD4TXV59", "nome": "UBS Core MSCI World UCITS ETF USD acc", "categoria": "Azioni Mondo", "ter": 0.15, "aum_mln": 4000},
    {"isin": "IE00B8GKDB10", "nome": "Vanguard FTSE All-World High Dividend Yield UCITS ETF", "categoria": "Azioni Mondo", "ter": 0.29, "aum_mln": 4000},
    {"isin": "IE00BGV5VN51", "nome": "Xtrackers Artificial Intelligence & Big Data UCITS ETF 1C", "categoria": "Azioni Mondo", "ter": 0.35, "aum_mln": 3500},
    {"isin": "IE00BMC38736", "nome": "VanEck Semiconductor UCITS ETF", "categoria": "Azioni Mondo", "ter": 0.35, "aum_mln": 3000},
    {"isin": "NL0011683594", "nome": "VanEck Morningstar Developed Markets Dividend Leaders UCITS ETF", "categoria": "Azioni Mondo", "ter": 0.38, "aum_mln": 2500},
    {"isin": "IE000YYE6WK5", "nome": "VanEck Defense UCITS ETF A", "categoria": "Azioni Mondo", "ter": 0.55, "aum_mln": 2000},
    {"isin": "IE00B60SX394", "nome": "Invesco MSCI World UCITS ETF Acc", "categoria": "Azioni Mondo", "ter": 0.19, "aum_mln": 1800},
    {"isin": "IE00BYX2JD69", "nome": "iShares MSCI World SRI UCITS ETF EUR (Acc)", "categoria": "Azioni Mondo", "ter": 0.20, "aum_mln": 1500},
    {"isin": "IE00BP3QZB59", "nome": "iShares Edge MSCI World Value Factor UCITS ETF", "categoria": "Azioni Mondo", "ter": 0.30, "aum_mln": 1200},
    # AZIONI USA
    {"isin": "IE00B5BMR087", "nome": "iShares Core S&P 500 UCITS ETF USD (Acc)", "categoria": "Azioni USA", "ter": 0.07, "aum_mln": 90000},
    {"isin": "IE00B3XXRP09", "nome": "Vanguard S&P 500 UCITS ETF (USD) Dist", "categoria": "Azioni USA", "ter": 0.07, "aum_mln": 50000},
    {"isin": "IE00B3YCGJ38", "nome": "Invesco S&P 500 UCITS ETF Acc", "categoria": "Azioni USA", "ter": 0.05, "aum_mln": 15000},
    {"isin": "IE00BFMXXD54", "nome": "Vanguard S&P 500 UCITS ETF (USD) Acc", "categoria": "Azioni USA", "ter": 0.07, "aum_mln": 12000},
    {"isin": "IE00B53SZB19", "nome": "iShares Nasdaq 100 UCITS ETF (Acc)", "categoria": "Azioni USA", "ter": 0.33, "aum_mln": 14000},
    {"isin": "IE0031442068", "nome": "iShares Core S&P 500 UCITS ETF USD (Dist)", "categoria": "Azioni USA", "ter": 0.07, "aum_mln": 8000},
    {"isin": "LU1135865084", "nome": "Amundi Core S&P 500 Swap UCITS ETF Acc", "categoria": "Azioni USA", "ter": 0.05, "aum_mln": 6000},
    {"isin": "IE0032077012", "nome": "Invesco EQQQ Nasdaq-100 UCITS ETF", "categoria": "Azioni USA", "ter": 0.30, "aum_mln": 7000},
    # AZIONI EUROPA
    {"isin": "LU0908500753", "nome": "Amundi Core Stoxx Europe 600 UCITS ETF Acc", "categoria": "Azioni Europa", "ter": 0.07, "aum_mln": 8000},
    {"isin": "IE00B4K48X80", "nome": "iShares Core MSCI Europe UCITS ETF EUR (Acc)", "categoria": "Azioni Europa", "ter": 0.12, "aum_mln": 7000},
    {"isin": "IE00B1YZSC51", "nome": "iShares Core MSCI Europe UCITS ETF EUR (Dist)", "categoria": "Azioni Europa", "ter": 0.12, "aum_mln": 5000},
    {"isin": "DE0002635307", "nome": "iShares STOXX Europe 600 UCITS ETF (DE)", "categoria": "Azioni Europa", "ter": 0.20, "aum_mln": 8000},
    {"isin": "IE00B53L3W79", "nome": "iShares Core EURO STOXX 50 UCITS ETF EUR (Acc)", "categoria": "Azioni Europa", "ter": 0.10, "aum_mln": 4000},
    {"isin": "IE00B53QG562", "nome": "iShares Core MSCI EMU UCITS ETF EUR (Acc)", "categoria": "Azioni Europa", "ter": 0.12, "aum_mln": 3500},
    # AZIONI EMERGENTI
    {"isin": "IE00BKM4GZ66", "nome": "iShares Core MSCI Emerging Markets IMI UCITS ETF (Acc)", "categoria": "Azioni Emergenti", "ter": 0.18, "aum_mln": 18000},
    {"isin": "IE00BTJRMP35", "nome": "Xtrackers MSCI Emerging Markets UCITS ETF 1C", "categoria": "Azioni Emergenti", "ter": 0.18, "aum_mln": 8000},
    {"isin": "IE00B0M63177", "nome": "iShares MSCI EM UCITS ETF (Dist)", "categoria": "Azioni Emergenti", "ter": 0.18, "aum_mln": 6000},
    {"isin": "IE00BHZPJ239", "nome": "iShares MSCI EM ESG Enhanced CTB UCITS ETF USD (Acc)", "categoria": "Azioni Emergenti", "ter": 0.18, "aum_mln": 5000},
    {"isin": "LU0950674175", "nome": "UBS Core MSCI EM UCITS ETF USD acc", "categoria": "Azioni Emergenti", "ter": 0.17, "aum_mln": 3000},
    {"isin": "IE00BMG6Z448", "nome": "iShares MSCI EM ex-China UCITS ETF USD (Acc)", "categoria": "Azioni Emergenti", "ter": 0.18, "aum_mln": 2500},
    {"isin": "LU2009202107", "nome": "Amundi MSCI Emerging Ex China UCITS ETF Acc", "categoria": "Azioni Emergenti", "ter": 0.15, "aum_mln": 2000},
    # OBBLIGAZIONI GOVERNATIVI EUR
    {"isin": "IE00B4WXJJ64", "nome": "iShares Core Euro Government Bond UCITS ETF (Dist)", "categoria": "Obbligazioni Governativi EUR", "ter": 0.07, "aum_mln": 12000},
    {"isin": "IE00BH04GL39", "nome": "Vanguard EUR Eurozone Government Bond UCITS ETF Acc", "categoria": "Obbligazioni Governativi EUR", "ter": 0.07, "aum_mln": 4000},
    {"isin": "LU1681046261", "nome": "Amundi Euro Government tilted Green Bond UCITS ETF Acc", "categoria": "Obbligazioni Governativi EUR", "ter": 0.14, "aum_mln": 3000},
    {"isin": "LU0290356871", "nome": "Xtrackers Eurozone Government Bond 1-3 UCITS ETF 1C", "categoria": "Obbligazioni Governativi EUR", "ter": 0.15, "aum_mln": 2500},
    {"isin": "LU0290355717", "nome": "Xtrackers II Eurozone Government Bond UCITS ETF 1C", "categoria": "Obbligazioni Governativi EUR", "ter": 0.15, "aum_mln": 2000},
    {"isin": "IE00B3VTMJ91", "nome": "iShares Euro Government Bond 1-3yr UCITS ETF (Acc)", "categoria": "Obbligazioni Governativi EUR", "ter": 0.15, "aum_mln": 1800},
    {"isin": "IE00B1FZS681", "nome": "iShares Euro Government Bond 3-5yr UCITS ETF", "categoria": "Obbligazioni Governativi EUR", "ter": 0.15, "aum_mln": 1500},
    # OBBLIGAZIONI SOCIETARI EUR
    {"isin": "IE00B3F81R35", "nome": "iShares Core EUR Corporate Bond UCITS ETF (Dist)", "categoria": "Obbligazioni Societari EUR", "ter": 0.20, "aum_mln": 8000},
    {"isin": "IE00BF11F565", "nome": "iShares Core EUR Corporate Bond UCITS ETF (Acc)", "categoria": "Obbligazioni Societari EUR", "ter": 0.20, "aum_mln": 6000},
    {"isin": "LU1437018168", "nome": "Amundi Index Euro Corporate SRI UCITS ETF DR (C)", "categoria": "Obbligazioni Societari EUR", "ter": 0.14, "aum_mln": 3000},
    {"isin": "LU0478205379", "nome": "Xtrackers II EUR Corporate Bond UCITS ETF 1C", "categoria": "Obbligazioni Societari EUR", "ter": 0.15, "aum_mln": 2500},
    {"isin": "IE00B4L60045", "nome": "iShares EUR Corporate Bond 1-5yr UCITS ETF EUR (Dist)", "categoria": "Obbligazioni Societari EUR", "ter": 0.20, "aum_mln": 2000},
    {"isin": "IE00BCRY6557", "nome": "iShares EUR Ultrashort Bond UCITS ETF EUR (Dist)", "categoria": "Obbligazioni Societari EUR", "ter": 0.09, "aum_mln": 5000},
    {"isin": "LU2037748774", "nome": "Amundi Index Euro Corporate SRI 0-3Y UCITS ETF DR (C)", "categoria": "Obbligazioni Societari EUR", "ter": 0.12, "aum_mln": 1500},
    # OBBLIGAZIONI HIGH YIELD
    {"isin": "IE00B66F4759", "nome": "iShares EUR High Yield Corporate Bond UCITS ETF EUR (Dist)", "categoria": "Obbligazioni High Yield", "ter": 0.50, "aum_mln": 7000},
    {"isin": "IE00BJK55C48", "nome": "iShares EUR High Yield Corporate Bond ESG SRI UCITS ETF (Acc)", "categoria": "Obbligazioni High Yield", "ter": 0.50, "aum_mln": 2000},
    {"isin": "IE00B4PY7Y77", "nome": "iShares USD High Yield Corporate Bond UCITS ETF USD (Dist)", "categoria": "Obbligazioni High Yield", "ter": 0.50, "aum_mln": 4000},
    {"isin": "LU1109943388", "nome": "Xtrackers EUR High Yield Corporate Bond UCITS ETF 1C", "categoria": "Obbligazioni High Yield", "ter": 0.20, "aum_mln": 2500},
    {"isin": "IE00B74DQ490", "nome": "iShares Global High Yield Corporate Bond UCITS ETF", "categoria": "Obbligazioni High Yield", "ter": 0.50, "aum_mln": 2000},
    {"isin": "IE00BCRY6003", "nome": "iShares USD Short Duration High Yield Corporate Bond UCITS ETF", "categoria": "Obbligazioni High Yield", "ter": 0.45, "aum_mln": 1500},
    {"isin": "IE00BF8HV600", "nome": "PIMCO US Short-Term High Yield Corporate Bond EUR Hdg Dist", "categoria": "Obbligazioni High Yield", "ter": 0.55, "aum_mln": 1000},
    {"isin": "LU1681040496", "nome": "Amundi Euro High Yield Bond ESG UCITS ETF EUR (C)", "categoria": "Obbligazioni High Yield", "ter": 0.35, "aum_mln": 800},
    # OBBLIGAZIONI EMERGENTI
    {"isin": "IE00B5M4WH52", "nome": "iShares J.P. Morgan EM Local Government Bond UCITS ETF", "categoria": "Obbligazioni Emergenti", "ter": 0.50, "aum_mln": 3000},
    {"isin": "IE00B2NPKV68", "nome": "iShares J.P. Morgan USD Emerging Markets Bond UCITS ETF (Dist)", "categoria": "Obbligazioni Emergenti", "ter": 0.45, "aum_mln": 6000},
    {"isin": "IE00B9M6RS56", "nome": "iShares J.P. Morgan USD EM Bond EUR Hedged UCITS ETF (Dist)", "categoria": "Obbligazioni Emergenti", "ter": 0.50, "aum_mln": 2000},
    {"isin": "IE00BF553838", "nome": "iShares J.P. Morgan Advanced USD EM Bond UCITS ETF (Acc)", "categoria": "Obbligazioni Emergenti", "ter": 0.45, "aum_mln": 1500},
    {"isin": "IE00B6TLBW47", "nome": "iShares J.P. Morgan USD EM Corporate Bond UCITS ETF (Dist)", "categoria": "Obbligazioni Emergenti", "ter": 0.50, "aum_mln": 1200},
    {"isin": "IE00BGYWCB81", "nome": "Vanguard USD Emerging Markets Government Bond UCITS ETF Acc", "categoria": "Obbligazioni Emergenti", "ter": 0.25, "aum_mln": 1000},
    {"isin": "IE00BZ163L38", "nome": "Vanguard USD Emerging Markets Government Bond UCITS ETF Dist", "categoria": "Obbligazioni Emergenti", "ter": 0.25, "aum_mln": 800},
    # iBONDS
    {"isin": "IE000264WWY0", "nome": "iShares iBonds Dec 2028 Term EUR Corporate UCITS ETF (Dist)", "categoria": "iBonds", "ter": 0.12, "aum_mln": 800},
    {"isin": "IE0008UEVOE0", "nome": "iShares iBonds Dec 2028 Term EUR Corporate UCITS ETF (Acc)", "categoria": "iBonds", "ter": 0.12, "aum_mln": 600},
    {"isin": "IE000ZOI8OK5", "nome": "iShares iBonds Dec 2027 Term EUR Corporate UCITS ETF (Acc)", "categoria": "iBonds", "ter": 0.12, "aum_mln": 700},
    {"isin": "IE000SIZJ2B2", "nome": "iShares iBonds Dec 2026 Term EUR Corporate UCITS ETF (Dist)", "categoria": "iBonds", "ter": 0.12, "aum_mln": 900},
    {"isin": "IE000WA6L436", "nome": "iShares iBonds Dec 2026 Term EUR Corporate UCITS ETF (Acc)", "categoria": "iBonds", "ter": 0.12, "aum_mln": 700},
    {"isin": "IE000LX17BP9", "nome": "iShares iBonds Dec 2030 Term EUR Corporate UCITS ETF", "categoria": "iBonds", "ter": 0.12, "aum_mln": 500},
    # BTP / ITALIA
    {"isin": "IE00B3T9LM79", "nome": "iShares Italy Govt Bond UCITS ETF (IBTS)", "categoria": "BTP/Italia", "ter": 0.20, "aum_mln": 1200},
    {"isin": "IE00B99470V8", "nome": "iShares BTP 1-3yr UCITS ETF (BTP5)", "categoria": "BTP/Italia", "ter": 0.15, "aum_mln": 800},
    {"isin": "IE00B1FZS798", "nome": "iShares EUR Inflation Linked Bond UCITS ETF (BTPI)", "categoria": "BTP/Italia", "ter": 0.25, "aum_mln": 1500},
    # MATERIE PRIME
    {"isin": "IE00BD6FTQ80", "nome": "Invesco Bloomberg Commodity UCITS ETF Acc", "categoria": "Materie Prime", "ter": 0.19, "aum_mln": 2000},
    {"isin": "GB00B15KXQ89", "nome": "WisdomTree Copper", "categoria": "Materie Prime", "ter": 0.49, "aum_mln": 300},
    {"isin": "LU1829218749", "nome": "Amundi Bloomberg Equal-weight Commodity ex-Agriculture UCITS ETF", "categoria": "Materie Prime", "ter": 0.30, "aum_mln": 600},
    {"isin": "GB00B15KYG56", "nome": "WisdomTree Industrial Metals", "categoria": "Materie Prime", "ter": 0.49, "aum_mln": 250},
    {"isin": "GB00B15KYH63", "nome": "WisdomTree Agriculture", "categoria": "Materie Prime", "ter": 0.49, "aum_mln": 200},
    {"isin": "IE00B53H0131", "nome": "UBS CMCI Composite SF UCITS ETF USD acc", "categoria": "Materie Prime", "ter": 0.34, "aum_mln": 500},
    {"isin": "IE00BFXR6159", "nome": "L&G Multi-Strategy Enhanced Commodities UCITS ETF USD Acc", "categoria": "Materie Prime", "ter": 0.30, "aum_mln": 400},
    {"isin": "GB00B15KXV33", "nome": "WisdomTree WTI Crude Oil", "categoria": "Materie Prime", "ter": 0.49, "aum_mln": 350},
    {"isin": "JE00B78CGV99", "nome": "WisdomTree Brent Crude Oil", "categoria": "Materie Prime", "ter": 0.49, "aum_mln": 300},
]


def get_static_etf_df() -> "pd.DataFrame":
    import pandas as pd
    df = pd.DataFrame(ETF_STATIC)
    df["perf_1y"] = None
    df["perf_3y"] = None
    df["perf_5y"] = None
    df["_lista"] = "C"
    return df
