"""
portafogli_efficienti.py — App Streamlit per costruzione portafogli di investimento.
Entry point principale.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(BASE_DIR))

from utils.data_loader import (
    load_fondi_terzi, load_fondi_azimut, build_unified_fund_df,
    TERZI_COLS, AZIMUT_COLS,
)
from utils.scoring import compute_scores_df
from utils.constraints import (
    build_lista_a, build_lista_b, PROFILI, build_portfolio_quality, classify_bucket,
)
from utils.etf_fetcher import (
    load_etf_universe, fetch_etf_universe, HARDCODED_ISINS, CATEGORY_MAP,
)
from utils.nav_fetcher import get_multiple_nav, get_nav_series
from utils.optimizer import compute_efficient_frontier, compute_black_litterman, estimate_max_drawdown
from utils.exporter import export_portfolio_excel, export_portfolio_pdf

# ---------------------------------------------------------------------------
# COSTANTI STILE
# ---------------------------------------------------------------------------
NAVY = "#1A2C54"
LIGHT_GRAY = "#F5F7FA"
PAGE_TITLE = "Portafogli Efficienti"

PRESEL_CACHE_FILE = DATA_DIR / "preselection_cache.json"
PRESEL_CACHE_TTL = 24


# ---------------------------------------------------------------------------
# CONFIGURAZIONE PAGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizzato
st.markdown(f"""
<style>
    [data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        color: white !important;
    }}
    .metric-card {{
        background: {LIGHT_GRAY};
        border-left: 4px solid {NAVY};
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
    }}
    .stDataFrame thead th {{
        background-color: {NAVY} !important;
        color: white !important;
    }}
    h1, h2, h3 {{
        color: {NAVY};
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CACHE & SESSIONE
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def _load_all_data(terzi_bytes: bytes | None = None, azimut_bytes: bytes | None = None):
    """Carica tutti i dati all'avvio (cache 24h). Accetta bytes da file_uploader."""
    import io
    terzi_src = io.BytesIO(terzi_bytes) if terzi_bytes else None
    azimut_src = io.BytesIO(azimut_bytes) if azimut_bytes else None
    df_terzi = load_fondi_terzi(path=terzi_src)
    df_azimut = load_fondi_azimut(path=azimut_src)
    df_unified = build_unified_fund_df(df_terzi, df_azimut)
    return df_terzi, df_azimut, df_unified


@st.cache_data(ttl=86400, show_spinner=False)
def _load_preselection(df_unified_json: str):
    import io
    df_unified = pd.read_json(io.StringIO(df_unified_json), orient="records")
    lista_a = build_lista_a(df_unified)
    lista_b = build_lista_b(df_unified)
    return lista_a, lista_b


@st.cache_data(ttl=86400, show_spinner=False)
def _load_etf_universe_cached():
    return load_etf_universe()


def _stars_emoji(n) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    try:
        n = int(n)
        return "★" * n + "☆" * (5 - n)
    except Exception:
        return "—"


def _pct_fmt(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:+.2f}%"


# ---------------------------------------------------------------------------
# INIT SESSIONE
# ---------------------------------------------------------------------------
def init_session():
    defaults = {
        "risk_free_rate": 2.5,
        "opt_period": "3Y",
        "min_weight": 3,
        "max_weight": 30,
        "profilo": "Equilibrato",
        "fondi_per_bucket": 4,
        "selected_assets": [],
        "bl_views": {},
        "extra_etf_isins": [],
        "locked_funds": set(),
        "manual_replacements": {},
        "demo_mode": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ---------------------------------------------------------------------------
# SIDEBAR — FILE UPLOAD + NAVIGAZIONE
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## 📊 {PAGE_TITLE}")
    st.markdown("---")

    # Upload file Excel
    with st.expander("📂 Carica file dati", expanded=False):
        st.caption("Obbligatori per dati reali; senza file → modalità Demo")
        up_terzi = st.file_uploader(
            "Fondi Terzi (tabella_fondi_arricchita.xlsx)",
            type=["xlsx"],
            key="up_terzi",
        )
        up_azimut = st.file_uploader(
            "Fondi Azimut (fondi_azimut_isin_completo_RATED.xlsx)",
            type=["xlsx"],
            key="up_azimut",
        )
        if up_terzi or up_azimut:
            if st.button("🔄 Ricarica dati", use_container_width=True):
                _load_all_data.clear()
                _load_preselection.clear()
                st.rerun()

    st.markdown("---")


# ---------------------------------------------------------------------------
# CARICAMENTO DATI
# ---------------------------------------------------------------------------
_terzi_bytes = st.session_state.get("up_terzi") and st.session_state["up_terzi"].read() or None
_azimut_bytes = st.session_state.get("up_azimut") and st.session_state["up_azimut"].read() or None

# Leggi i bytes dagli uploader se presenti
_terzi_bytes = None
_azimut_bytes = None
if "up_terzi" in st.session_state and st.session_state["up_terzi"] is not None:
    _terzi_bytes = st.session_state["up_terzi"].getvalue()
if "up_azimut" in st.session_state and st.session_state["up_azimut"] is not None:
    _azimut_bytes = st.session_state["up_azimut"].getvalue()

with st.spinner("Caricamento dati in corso..."):
    try:
        df_terzi, df_azimut, df_unified = _load_all_data(
            terzi_bytes=_terzi_bytes,
            azimut_bytes=_azimut_bytes,
        )
        demo_mode = (len(df_terzi) <= 2 and len(df_azimut) <= 1)
        st.session_state["demo_mode"] = demo_mode
    except Exception as e:
        st.warning(f"Errore caricamento dati — modalità Demo attiva. ({e})")
        df_terzi, df_azimut, df_unified = _load_all_data()
        demo_mode = True
        st.session_state["demo_mode"] = True

try:
    lista_a, lista_b = _load_preselection(df_unified.to_json(orient="records"))
except Exception as e:
    st.error(f"Errore nella costruzione delle liste: {e}")
    lista_a, lista_b = pd.DataFrame(), pd.DataFrame()


# ---------------------------------------------------------------------------
# SIDEBAR — RESTO NAVIGAZIONE
# ---------------------------------------------------------------------------
with st.sidebar:
    if st.session_state.get("demo_mode"):
        st.warning("⚠️ Modalità Demo — carica i file Excel per dati reali")
    else:
        st.success(f"✅ {len(df_terzi)+len(df_azimut)} fondi caricati")
    st.markdown("---")
    nav = st.radio(
        "Navigazione",
        options=[
            "🏠 Home",
            "📈 Frontiera Efficiente",
            "⭐ Portafoglio Qualità",
            "🌐 ETF Universe",
            "⚙️ Impostazioni",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### Impostazioni rapide")
    st.session_state["risk_free_rate"] = st.slider(
        "Risk-free rate (%)", 0.0, 5.0,
        float(st.session_state["risk_free_rate"]), 0.1,
    )
    st.session_state["opt_period"] = st.selectbox(
        "Periodo ottimizzazione",
        ["1Y", "3Y", "5Y"],
        index=["1Y", "3Y", "5Y"].index(st.session_state["opt_period"]),
    )


# ===========================================================================
# HOME
# ===========================================================================
if nav == "🏠 Home":
    st.title("📊 Portafogli Efficienti")
    st.markdown(
        "Costruttore professionale di portafogli di investimento. "
        "Combina **fondi terzi**, **fondi Azimut** ed **ETF** con ottimizzazione "
        "quantitativa (Frontiera Efficiente, Black-Litterman) o per Score Qualità."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fondi Terzi", len(df_terzi))
    col2.metric("Fondi Azimut", len(df_azimut))
    col3.metric("Lista A (Generalisti)", len(lista_a))
    col4.metric("Lista B (Tematici)", len(lista_b))

    # DEBUG — visibile solo se liste vuote
    if len(lista_a) == 0 or len(lista_b) == 0:
        with st.expander("🔍 Debug dati caricati", expanded=True):
            st.markdown("**Colonne rilevate nel df_unified:**")
            st.code(list(df_unified.columns))
            if not df_unified.empty:
                st.markdown("**Campione dati (prime 3 righe):**")
                st.dataframe(df_unified.head(3))
                st.markdown("**Distribuzione classificazione (top 10):**")
                st.dataframe(df_unified["classificazione"].value_counts().head(10))
                st.markdown("**Statistiche perf_3y:**")
                st.write(df_unified["perf_3y"].describe())
                st.markdown("**Distribuzione rating_fida:**")
                st.write(df_unified["rating_fida"].value_counts(dropna=False))
                from utils.scoring import is_generalista
                from utils.data_loader import _remap_columns, TERZI_COLS
                n_gen = df_unified["classificazione"].apply(is_generalista).sum()
                st.write(f"Fondi con classificazione generalista: **{n_gen}**")
                n_perf = df_unified["perf_3y"].notna().sum()
                st.write(f"Fondi con perf_3y disponibile: **{n_perf}**")
                st.markdown("**Sample classificazione (10 valori):**")
                st.write(df_unified["classificazione"].dropna().head(10).tolist())
                st.markdown("**Sample perf_3y (10 valori):**")
                st.write(df_unified["perf_3y"].dropna().head(10).tolist())

    st.markdown("---")
    st.subheader("Liste Preselezionate")
    tab_a, tab_b = st.tabs(["📋 Lista A — Generalisti", "🎯 Lista B — Tematici"])

    def _render_fund_table(df: pd.DataFrame, key: str):
        if df.empty:
            st.info("Nessun fondo disponibile.")
            return
        cols_show = ["isin", "nome", "casa", "classificazione",
                     "perf_1y", "perf_3y", "volatilita", "rating_fida", "score_qualita"]
        cols_show = [c for c in cols_show if c in df.columns]
        display = df[cols_show].copy()

        # Ricerca
        q = st.text_input("Cerca per nome/ISIN/classificazione", key=f"search_{key}")
        if q:
            mask = display.apply(
                lambda col: col.astype(str).str.contains(q, case=False, na=False)
            ).any(axis=1)
            display = display[mask]

        col_config = {
            "isin": st.column_config.TextColumn("ISIN", width="small"),
            "nome": st.column_config.TextColumn("Fondo", width="large"),
            "casa": st.column_config.TextColumn("Casa", width="medium"),
            "classificazione": st.column_config.TextColumn("Classificazione", width="medium"),
            "perf_1y": st.column_config.NumberColumn("Perf 1Y %", format="%.2f"),
            "perf_3y": st.column_config.NumberColumn("Perf 3Y %", format="%.2f"),
            "volatilita": st.column_config.NumberColumn("Volatilità %", format="%.2f"),
            "rating_fida": st.column_config.NumberColumn("★ FIDA", format="%d"),
            "score_qualita": st.column_config.NumberColumn("Score", format="%.3f"),
        }
        st.dataframe(display, column_config=col_config, use_container_width=True, height=400)

    with tab_a:
        _render_fund_table(lista_a, "a")
    with tab_b:
        _render_fund_table(lista_b, "b")


# ===========================================================================
# FRONTIERA EFFICIENTE
# ===========================================================================
elif nav == "📈 Frontiera Efficiente":
    st.title("📈 Frontiera Efficiente")
    st.markdown(
        "Costruisci un portafoglio ottimizzato con PyPortfolioOpt. "
        "**Step 1**: scegli gli strumenti → **Step 2**: imposta i vincoli → **Step 3**: calcola."
    )

    # ── STEP 1: RICERCA E SELEZIONE STRUMENTI ──────────────────────────────
    st.subheader("1️⃣ Cerca e seleziona strumenti")

    fe_tab_a, fe_tab_b, fe_tab_c, fe_tab_isin = st.tabs(
        ["📋 Lista A — Generalisti", "🎯 Lista B — Tematici",
         "🌐 Lista C — ETF", "✏️ Inserisci ISIN/Ticker"]
    )

    # Dizionario globale isin→info per tutti gli asset disponibili
    _all_fund_pool: dict = {}

    def _pool_from_df(df: pd.DataFrame, prefix: str):
        for _, r in df.iterrows():
            isin = str(r.get("isin", r.get("ISIN", ""))).strip()
            if isin:
                _all_fund_pool[isin] = r.to_dict()

    if not lista_a.empty:
        _pool_from_df(lista_a, "A")
    if not lista_b.empty:
        _pool_from_df(lista_b, "B")
    try:
        df_etf_fe = _load_etf_universe_cached()
        _pool_from_df(df_etf_fe, "C")
    except Exception:
        df_etf_fe = pd.DataFrame()

    # Stato selezione
    if "fe_selected_isins" not in st.session_state:
        st.session_state["fe_selected_isins"] = []

    def _render_selectable_table(df: pd.DataFrame, tab_key: str, cols_show: list):
        """Tabella con checkbox per aggiungere alla selezione."""
        if df.empty:
            st.info("Nessun dato disponibile.")
            return
        search = st.text_input("🔍 Cerca per nome / ISIN / classificazione",
                               key=f"srch_{tab_key}")
        disp = df.copy()
        if search:
            mask = disp.apply(
                lambda col: col.astype(str).str.contains(search, case=False, na=False)
            ).any(axis=1)
            disp = disp[mask]

        show_cols = [c for c in cols_show if c in disp.columns]
        if not show_cols:
            show_cols = list(disp.columns[:6])

        # Aggiungi colonna "Selezionato"
        disp = disp[show_cols].copy().head(200)
        if "isin" in disp.columns:
            disp.insert(0, "➕", disp["isin"].isin(st.session_state["fe_selected_isins"]))
        st.dataframe(disp, use_container_width=True, hide_index=True,
                     column_config={"➕": st.column_config.CheckboxColumn("Sel.", width="small")},
                     height=300)

        # Multiselect per aggiungere alla selezione
        isin_list = [str(r.get("isin", "")) for _, r in
                     df[show_cols if "isin" in show_cols else []].iterrows()
                     if str(r.get("isin", "")).strip()] if "isin" in df.columns else []

        # Filtro ricerca
        filtered_isins = []
        if "isin" in df.columns:
            fdf = df.copy()
            if search:
                mask2 = fdf.apply(
                    lambda col: col.astype(str).str.contains(search, case=False, na=False)
                ).any(axis=1)
                fdf = fdf[mask2]
            filtered_isins = fdf["isin"].dropna().astype(str).tolist()[:200]

        to_add = st.multiselect(
            "Aggiungi alla selezione",
            options=filtered_isins,
            format_func=lambda x: f"{x} — {str(_all_fund_pool.get(x, {}).get('nome', ''))[:55]}",
            key=f"ms_{tab_key}",
        )
        if st.button("➕ Aggiungi selezionati", key=f"add_{tab_key}"):
            for isin in to_add:
                if isin not in st.session_state["fe_selected_isins"]:
                    st.session_state["fe_selected_isins"].append(isin)
            st.rerun()

    FUND_COLS = ["isin", "nome", "classificazione", "perf_1y", "perf_3y", "volatilita", "rating_fida"]
    ETF_COLS  = ["isin", "nome", "categoria", "ter", "aum_mln"]

    with fe_tab_a:
        _render_selectable_table(lista_a, "A", FUND_COLS)
    with fe_tab_b:
        _render_selectable_table(lista_b, "B", FUND_COLS)
    with fe_tab_c:
        _render_selectable_table(df_etf_fe if not df_etf_fe.empty else pd.DataFrame(), "C", ETF_COLS)
    with fe_tab_isin:
        st.markdown("""
**Inserisci ISIN o ticker — uno per riga.**

| Tipo | Esempi | Fonte dati |
|------|--------|------------|
| ETF europei (ISIN) | `IE00B4L5Y983`, `LU0908500753` | yfinance via ticker map |
| Azioni italiane | `ENI.MI`, `ISP.MI`, `UCG.MI` | yfinance diretto |
| Azioni USA | `AAPL`, `MSFT`, `NVDA` | yfinance diretto |
| Azioni europee | `ADS.DE`, `ASML.AS`, `MC.PA` | yfinance diretto |
| ETF su Xetra | `EXW1.DE`, `EUNL.DE` | yfinance diretto |
| Fondi (ISIN) | `LU0048578792` | Morningstar → FondiDoc → sintetica |
| BTP proxy | `IBTS.MI`, `BTP5.MI` | yfinance diretto |
""")
        custom_raw = st.text_area("ISIN / Ticker", height=120, key="fe_custom_raw",
                                  placeholder="ENI.MI\nAAPL\nNVDA\nISP.MI")
        if st.button("➕ Aggiungi ISIN/Ticker", key="add_custom"):
            from utils.nav_fetcher import classify_asset_type
            from utils.etf_tickers import ITALIAN_STOCKS, ISIN_TO_TICKER
            from utils.etf_static import ETF_STATIC
            _etf_nome_map = {e["isin"]: e["nome"] for e in ETF_STATIC}

            added_info = []
            for line in custom_raw.strip().split("\n"):
                token = line.strip().upper()
                if not token:
                    continue
                if token not in st.session_state["fe_selected_isins"]:
                    st.session_state["fe_selected_isins"].append(token)
                if token not in _all_fund_pool:
                    # Cerca info: prima azioni italiane, poi ETF map
                    if token in ITALIAN_STOCKS:
                        info = ITALIAN_STOCKS[token]
                        _all_fund_pool[token] = {
                            "isin": token,
                            "nome": info["nome"],
                            "classificazione": f"Azione — {info['settore']}",
                            "ticker": info["ticker"],
                        }
                    elif token in _etf_nome_map:
                        _all_fund_pool[token] = {
                            "isin": token,
                            "nome": _etf_nome_map[token],
                            "classificazione": "ETF (ticker noto)",
                            "ticker": ISIN_TO_TICKER.get(token, ""),
                        }
                    else:
                        asset_type = classify_asset_type(token)
                        _all_fund_pool[token] = {
                            "isin": token, "nome": token,
                            "classificazione": asset_type,
                        }
                added_info.append(
                    f"**{token}** → {_all_fund_pool[token].get('nome', token)} "
                    f"({_all_fund_pool[token].get('classificazione','')})"
                )
            if added_info:
                st.success("Aggiunti:")
                for msg in added_info:
                    st.markdown(f"- {msg}")
            st.rerun()

    # ── SELEZIONE CORRENTE ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📌 Strumenti selezionati per l'ottimizzazione")

    sel_isins = st.session_state["fe_selected_isins"]
    if sel_isins:
        sel_rows = []
        for isin in sel_isins:
            info = _all_fund_pool.get(isin, {"isin": isin, "nome": isin})
            sel_rows.append({
                "ISIN": isin,
                "Nome": str(info.get("nome", info.get("FONDO AZIMUT", isin)))[:60],
                "Classificazione": str(info.get("classificazione", info.get("categoria", ""))),
                "Perf 1Y %": info.get("perf_1y"),
                "Perf 3Y %": info.get("perf_3y"),
                "Volatilità %": info.get("volatilita"),
            })
        sel_df = pd.DataFrame(sel_rows)
        st.dataframe(sel_df, use_container_width=True, hide_index=True,
                     column_config={
                         "Perf 1Y %": st.column_config.NumberColumn(format="%.2f"),
                         "Perf 3Y %": st.column_config.NumberColumn(format="%.2f"),
                         "Volatilità %": st.column_config.NumberColumn(format="%.2f"),
                     })
        # Rimuovi asset
        to_remove = st.multiselect("Rimuovi dalla selezione",
                                   options=sel_isins,
                                   format_func=lambda x: f"{x} — {str(_all_fund_pool.get(x,{}).get('nome',x))[:50]}",
                                   key="fe_remove")
        if st.button("🗑️ Rimuovi", key="btn_remove") and to_remove:
            st.session_state["fe_selected_isins"] = [i for i in sel_isins if i not in to_remove]
            st.rerun()
        if st.button("🗑️ Svuota tutto", key="btn_clear_all"):
            st.session_state["fe_selected_isins"] = []
            st.session_state.pop("fe_result", None)
            st.rerun()
    else:
        st.info("Nessuno strumento selezionato. Usa i tab qui sopra per aggiungerne.")

    # ── STEP 2: VINCOLI ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("2️⃣ Vincoli di peso")
    v_col1, v_col2, v_col3 = st.columns(3)
    min_w = v_col1.slider("Peso minimo per asset (%)", 0, 20,
                           st.session_state["min_weight"]) / 100
    max_w = v_col2.slider("Peso massimo per asset (%)", 10, 100,
                           st.session_state["max_weight"]) / 100
    forced_include_sel = v_col3.multiselect(
        "Forza inclusione",
        options=sel_isins,
        format_func=lambda x: f"{x}",
        key="fe_forced_include",
    )

    # ── STEP 2b: BLACK-LITTERMAN ───────────────────────────────────────────
    with st.expander("⚙️ Black-Litterman (opzionale)"):
        use_bl = st.checkbox("Abilita Black-Litterman")
        bl_views: dict = {}
        bl_conf: dict = {}
        if use_bl and sel_isins:
            st.caption("Inserisci le tue aspettative di rendimento per uno o più asset:")
            for isin in sel_isins[:15]:
                nome_bl = str(_all_fund_pool.get(isin, {}).get("nome", isin))[:45]
                c1, c2, c3 = st.columns([3, 2, 2])
                en = c1.checkbox(f"{isin} — {nome_bl}", key=f"bl_en_{isin}")
                if en:
                    bl_views[isin] = c2.number_input("Rend. atteso (%)", -30.0, 60.0, 5.0,
                                                      key=f"bl_r_{isin}")
                    bl_conf[isin] = c3.slider("Confidenza", 0.1, 1.0, 0.5,
                                               key=f"bl_c_{isin}")

    # ── STEP 3: CALCOLO ────────────────────────────────────────────────────
    st.markdown("---")
    run_c1, run_c2 = st.columns([1, 4])
    run = run_c1.button("🚀 Calcola Frontiera", type="primary", use_container_width=True)
    if run_c2.button("🔄 Reset risultati", use_container_width=False):
        st.session_state.pop("fe_result", None)
        st.session_state.pop("fe_price_dict", None)
        st.rerun()

    if run:
        if len(sel_isins) < 3:
            st.error("Seleziona almeno 3 strumenti.")
        else:
            with st.spinner(f"Recupero dati storici per {len(sel_isins)} strumenti..."):
                period_map = {"1Y": "1y", "3Y": "3y", "5Y": "5y"}
                period = period_map.get(st.session_state["opt_period"], "3y")
                asset_list = []
                for isin in sel_isins:
                    info = _all_fund_pool.get(isin, {})
                    asset_list.append({
                        "isin": isin,
                        "ticker": info.get("ticker"),
                        "perf_1y": info.get("perf_1y"),
                        "perf_3y": info.get("perf_3y"),
                        "perf_ytd": info.get("perf_ytd"),
                        "perf_2022": info.get("perf_2022"),
                        "perf_2023": info.get("perf_2023"),
                        "perf_2024": info.get("perf_2024"),
                    })
                price_dict = get_multiple_nav(asset_list, period=period)

            if len(price_dict) < 3:
                st.error(f"Dati storici insufficienti: recuperati {len(price_dict)}/{len(sel_isins)} serie.")
                st.info("Suggerimento: gli ETF (Lista C) hanno dati su yfinance. I fondi richiedono Morningstar/FondiDoc.")
            else:
                with st.spinner("Ottimizzazione portafoglio..."):
                    rfr = st.session_state["risk_free_rate"] / 100
                    result = compute_efficient_frontier(
                        price_dict, weight_bounds=(min_w, max_w),
                        risk_free_rate=rfr,
                        forced_include=forced_include_sel or None,
                    )
                if "error" in result:
                    st.error(f"Errore ottimizzazione: {result['error']}")
                else:
                    st.success(f"Ottimizzazione completata su {len(price_dict)} strumenti.")
                    st.session_state["fe_result"] = result
                    st.session_state["fe_price_dict"] = price_dict
                    if use_bl and bl_views:
                        with st.spinner("Black-Litterman..."):
                            bl_r = compute_black_litterman(
                                price_dict, bl_views, bl_conf,
                                weight_bounds=(min_w, max_w), risk_free_rate=rfr,
                            )
                        st.session_state["bl_result"] = bl_r

    # ── RISULTATI ──────────────────────────────────────────────────────────
    if "fe_result" in st.session_state:
        result   = st.session_state["fe_result"]
        price_dict = st.session_state.get("fe_price_dict", {})
        bl_result  = st.session_state.get("bl_result")

        # Mappa ISIN→nome per label leggibili
        isin_label = {isin: str(_all_fund_pool.get(isin, {}).get("nome", isin))[:35]
                      for isin in price_dict}

        st.markdown("---")
        st.subheader("📊 Risultati Ottimizzazione")

        # KPI
        m_col = st.columns(4)
        ms = result.get("max_sharpe", {})
        if ms and "error" not in ms:
            m_col[0].metric("📈 Rendimento (Max Sharpe)", f"{ms['ret']*100:.2f}%")
            m_col[1].metric("📉 Volatilità (Max Sharpe)", f"{ms['vol']*100:.2f}%")
            m_col[2].metric("⚡ Sharpe Ratio", f"{ms['sharpe']:.3f}")
            if price_dict:
                mdd = estimate_max_drawdown(ms["weights"], price_dict)
                m_col[3].metric("📉 Max Drawdown stimato", f"{mdd:.2f}%")

        # Grafico Frontiera + Monte Carlo
        fig_fe = go.Figure()
        mc = result.get("monte_carlo", pd.DataFrame())
        if not mc.empty:
            fig_fe.add_trace(go.Scatter(
                x=mc["vol"]*100, y=mc["ret"]*100, mode="markers",
                marker=dict(color=mc["sharpe"], colorscale="Viridis",
                            size=4, opacity=0.4, colorbar=dict(title="Sharpe")),
                name="Simulazioni Monte Carlo",
                hovertemplate="Vol: %{x:.1f}%<br>Rend: %{y:.1f}%<extra></extra>",
            ))
        frontier = result.get("frontier_df", pd.DataFrame())
        if not frontier.empty:
            fig_fe.add_trace(go.Scatter(
                x=frontier["vol"]*100, y=frontier["ret"]*100,
                mode="lines", line=dict(color=NAVY, width=3),
                name="Frontiera Efficiente",
            ))
        for pdata, pname, pcolor in [
            (ms, "Max Sharpe ★", "red"),
            (result.get("min_variance",{}), "Min Varianza ★", "#1A6EBD"),
            (bl_result or {}, "Black-Litterman ★", "green"),
        ]:
            if pdata and "error" not in pdata and "vol" in pdata:
                fig_fe.add_trace(go.Scatter(
                    x=[pdata["vol"]*100], y=[pdata["ret"]*100],
                    mode="markers+text",
                    marker=dict(color=pcolor, size=18, symbol="star"),
                    text=[pname], textposition="top center", name=pname,
                ))
        fig_fe.update_layout(
            title="Frontiera Efficiente — Rischio vs Rendimento",
            xaxis_title="Volatilità annua (%)",
            yaxis_title="Rendimento atteso (%)",
            height=520, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_fe, use_container_width=True)

        # Pesi + correlazioni affiancati
        st.subheader("📋 Composizione portafogli")
        tab_ms2, tab_mv2, tab_bl2 = st.tabs(["⭐ Max Sharpe", "🛡️ Min Varianza", "🔮 Black-Litterman"])

        def _render_weights(pdata: dict, label: str):
            if not pdata or "error" in pdata:
                st.info(f"Portafoglio {label} non disponibile.")
                return
            w = {k: v for k, v in pdata["weights"].items() if v > 0.001}
            # Arricchisci con nomi
            w_rows = []
            for isin, peso in sorted(w.items(), key=lambda x: -x[1]):
                info = _all_fund_pool.get(isin, {})
                w_rows.append({
                    "ISIN": isin,
                    "Nome": str(info.get("nome", isin))[:55],
                    "Classificazione": str(info.get("classificazione", info.get("categoria", ""))),
                    "Peso %": round(peso * 100, 2),
                    "Perf 3Y %": info.get("perf_3y"),
                    "Volatilità %": info.get("volatilita"),
                })
            w_df = pd.DataFrame(w_rows)

            c1, c2 = st.columns([3, 2])
            with c1:
                st.dataframe(w_df, use_container_width=True, hide_index=True,
                             column_config={
                                 "Peso %": st.column_config.ProgressColumn(
                                     "Peso %", min_value=0, max_value=100, format="%.1f%%"),
                                 "Perf 3Y %": st.column_config.NumberColumn(format="%.2f"),
                                 "Volatilità %": st.column_config.NumberColumn(format="%.2f"),
                             })
            with c2:
                pie = px.pie(w_df, values="Peso %", names="Nome", hole=0.35,
                             color_discrete_sequence=px.colors.sequential.Blues_r)
                pie.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0),
                                  showlegend=False)
                pie.update_traces(textposition="inside",
                                  textinfo="percent+label",
                                  textfont_size=10)
                st.plotly_chart(pie, use_container_width=True)

            st.markdown(
                f"**Rendimento atteso:** {pdata.get('ret',0)*100:.2f}% &nbsp;|&nbsp; "
                f"**Volatilità:** {pdata.get('vol',0)*100:.2f}% &nbsp;|&nbsp; "
                f"**Sharpe:** {pdata.get('sharpe',0):.3f}"
            )

        with tab_ms2: _render_weights(ms, "Max Sharpe")
        with tab_mv2: _render_weights(result.get("min_variance",{}), "Min Varianza")
        with tab_bl2: _render_weights(bl_result or {}, "Black-Litterman")

        # Matrice correlazioni — SEMPRE visibile (non in expander)
        if len(price_dict) >= 2:
            st.markdown("---")
            st.subheader("🔗 Matrice Correlazioni")
            prices_df = pd.DataFrame(price_dict).pct_change().dropna()
            prices_df.columns = [isin_label.get(c, c) for c in prices_df.columns]
            corr = prices_df.corr()
            fig_corr = px.imshow(
                corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                text_auto=".2f",
                title="Correlazioni storiche tra strumenti selezionati",
            )
            fig_corr.update_layout(height=max(350, len(corr)*40))
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("Valori vicini a +1: alta correlazione (si muovono insieme). Vicini a -1: diversificazione efficace.")

        # BL dettaglio
        if bl_result and "error" not in bl_result and "bl_returns" in bl_result:
            st.markdown("---")
            st.subheader("📐 Black-Litterman — Rendimenti Posteriori vs Prior")
            bl_rets = bl_result["bl_returns"]
            mu_prior = result.get("mu", {})
            bl_compare = pd.DataFrame({
                "Asset": [isin_label.get(k, k) for k in bl_rets],
                "Prior (%)": [mu_prior.get(k, 0)*100 for k in bl_rets],
                "BL Posteriore (%)": [v*100 for v in bl_rets.values()],
            })
            fig_bl = go.Figure()
            fig_bl.add_bar(x=bl_compare["Asset"], y=bl_compare["Prior (%)"],
                           name="Prior", marker_color="steelblue")
            fig_bl.add_bar(x=bl_compare["Asset"], y=bl_compare["BL Posteriore (%)"],
                           name="BL", marker_color="#E8603C")
            fig_bl.update_layout(barmode="group", height=350,
                                  yaxis_title="Rendimento atteso (%)")
            st.plotly_chart(fig_bl, use_container_width=True)

        # Export
        st.markdown("---")
        exp_c1, exp_c2 = st.columns(2)
        if ms and "error" not in ms:
            metrics = {
                "Rendimento atteso (%)": f"{ms.get('ret',0)*100:.2f}",
                "Volatilità (%)": f"{ms.get('vol',0)*100:.2f}",
                "Sharpe Ratio": f"{ms.get('sharpe',0):.3f}",
                "Generato": datetime.now().strftime("%d/%m/%Y %H:%M"),
            }
            excel_bytes = export_portfolio_excel(
                ms["weights"], metrics,
                fund_df=df_unified if not df_unified.empty else None,
                price_dict=price_dict, title="Max Sharpe",
            )
            exp_c1.download_button(
                "📥 Esporta Excel (Max Sharpe)", data=excel_bytes,
                file_name=f"portafoglio_max_sharpe_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            pdf_bytes = export_portfolio_pdf(ms["weights"], metrics, title="Portafoglio Max Sharpe")
            if pdf_bytes:
                exp_c2.download_button(
                    "📄 Esporta PDF (Max Sharpe)", data=pdf_bytes,
                    file_name=f"portafoglio_max_sharpe_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                )


# ===========================================================================
# PORTAFOGLIO QUALITÀ
# ===========================================================================
elif nav == "⭐ Portafoglio Qualità":
    st.title("⭐ Portafoglio Qualità")
    st.markdown("Selezione automatica dei migliori fondi per Score Qualità, suddivisi per bucket (Azionario, Obbligazionario, Bilanciato…).")

    # ── CONFIGURAZIONE ─────────────────────────────────────────────────────
    cfg_c1, cfg_c2 = st.columns([1, 2])
    with cfg_c1:
        profilo = st.selectbox("Profilo di rischio", list(PROFILI.keys()),
                               index=list(PROFILI.keys()).index(st.session_state["profilo"]))
        st.session_state["profilo"] = profilo
        fondi_per_bucket = st.slider("Fondi per bucket", 2, 8,
                                      st.session_state["fondi_per_bucket"])
        st.session_state["fondi_per_bucket"] = fondi_per_bucket

    with cfg_c2:
        st.markdown(f"**Allocazioni target — {profilo}** (modificabili)")
        alloc = PROFILI[profilo].copy()
        alloc_adj = {}
        cols_alloc = st.columns(len(alloc))
        for i, (bucket, pct) in enumerate(alloc.items()):
            alloc_adj[bucket] = cols_alloc[i].number_input(
                f"{bucket} (%)", 0, 100, pct, 5, key=f"alloc_{bucket}")
        total_alloc = sum(alloc_adj.values())
        if total_alloc != 100:
            st.warning(f"⚠️ Totale: {total_alloc}% — deve essere 100%")
        else:
            st.success(f"✅ Totale: {total_alloc}%")

    st.markdown("---")

    # ── DEBUG BUCKET ────────────────────────────────────────────────────────
    from utils.constraints import classify_bucket
    if not df_unified.empty:
        df_unified["_bucket_preview"] = df_unified["classificazione"].apply(classify_bucket)
        bucket_dist = df_unified["_bucket_preview"].value_counts()
        with st.expander("📊 Distribuzione fondi per bucket (verifica classificazione)"):
            st.dataframe(bucket_dist.reset_index().rename(
                columns={"_bucket_preview": "Bucket", "count": "N. fondi"}),
                use_container_width=True, hide_index=True)
            st.caption("Se Azionario è vuoto → le classificazioni inferite non contengono le parole chiave attese.")

    # ── COSTRUZIONE PORTAFOGLIO ─────────────────────────────────────────────
    with st.spinner("Costruzione portafoglio per Score Qualità..."):
        portfolio_buckets = build_portfolio_quality(
            df_unified, profilo=profilo, fondi_per_bucket=fondi_per_bucket
        )

    # ── VISUALIZZAZIONE BUCKET PER BUCKET ──────────────────────────────────
    BUCKET_COLORS = {
        "Azionario":     "#1A2C54",
        "Obbligazionario": "#2E6DA4",
        "Bilanciato":    "#5B9BD5",
        "Monetario":     "#70AD47",
        "Alternativo":   "#ED7D31",
    }
    all_porto_rows = []

    for bucket, df_bucket in portfolio_buckets.items():
        peso_bucket = alloc_adj.get(bucket, 0)
        color = BUCKET_COLORS.get(bucket, NAVY)
        st.markdown(
            f"<h3 style='color:{color}'>{'▣'} {bucket} — {peso_bucket}%</h3>",
            unsafe_allow_html=True,
        )
        if df_bucket is None or df_bucket.empty:
            st.warning(
                f"Nessun fondo trovato per **{bucket}**. "
                f"Verifica la distribuzione bucket qui sopra o rilassa i vincoli."
            )
            continue

        # Tabella principale con tutti i dettagli
        cols_show = ["isin", "nome", "classificazione", "casa",
                     "perf_1y", "perf_3y", "volatilita",
                     "rating_fida", "score_qualita", "_peso_fondo", "retrocessione"]
        cols_show = [c for c in cols_show if c in df_bucket.columns]
        display = df_bucket[cols_show].copy()
        # Formattiamo il nome a max 60 char
        if "nome" in display.columns:
            display["nome"] = display["nome"].astype(str).str[:60]

        st.dataframe(display, use_container_width=True, hide_index=True,
                     column_config={
                         "isin": st.column_config.TextColumn("ISIN", width="small"),
                         "nome": st.column_config.TextColumn("Fondo", width="large"),
                         "classificazione": st.column_config.TextColumn("Classificazione"),
                         "casa": st.column_config.TextColumn("Casa"),
                         "perf_1y": st.column_config.NumberColumn("Perf 1Y %", format="%.2f"),
                         "perf_3y": st.column_config.NumberColumn("Perf 3Y %", format="%.2f"),
                         "volatilita": st.column_config.NumberColumn("Vol %", format="%.2f"),
                         "rating_fida": st.column_config.NumberColumn("★ FIDA", format="%d"),
                         "score_qualita": st.column_config.ProgressColumn(
                             "Score", min_value=0, max_value=20, format="%.2f"),
                         "_peso_fondo": st.column_config.NumberColumn("Peso %", format="%.1f"),
                         "retrocessione": st.column_config.NumberColumn("Retro %", format="%.2f"),
                     })

        # Grafico score orizzontale
        if "score_qualita" in df_bucket.columns and len(df_bucket) > 0:
            _sc_df = df_bucket.copy()
            _sc_df["_label"] = _sc_df["nome"].astype(str).str[:35]
            fig_sc = px.bar(
                _sc_df.sort_values("score_qualita"),
                x="score_qualita", y="_label",
                orientation="h",
                color="score_qualita",
                color_continuous_scale=[[0, "#AED6F1"], [1, color]],
                title=f"Score Qualità — {bucket}",
                labels={"_label": "", "score_qualita": "Score"},
            )
            fig_sc.update_layout(height=max(200, len(df_bucket)*45),
                                  margin=dict(l=0, r=0, t=35, b=0),
                                  showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_sc, use_container_width=True)

        for _, r in df_bucket.iterrows():
            all_porto_rows.append({
                "Bucket": bucket, "ISIN": r.get("isin",""),
                "Fondo": str(r.get("nome",""))[:55],
                "Classificazione": r.get("classificazione",""),
                "Casa": r.get("casa",""),
                "Perf 1Y %": r.get("perf_1y"),
                "Perf 3Y %": r.get("perf_3y"),
                "Volatilità %": r.get("volatilita"),
                "★ FIDA": r.get("rating_fida"),
                "Score": r.get("score_qualita"),
                "Peso %": r.get("_peso_fondo"),
                "Retro %": r.get("retrocessione"),
            })

    # ── RIEPILOGO PORTAFOGLIO COMPLETO ─────────────────────────────────────
    if all_porto_rows:
        st.markdown("---")
        st.subheader("📋 Portafoglio Completo")
        df_porto = pd.DataFrame(all_porto_rows)

        pie_c, tbl_c = st.columns([1, 2])
        with pie_c:
            bw = df_porto.groupby("Bucket")["Peso %"].sum().reset_index()
            fig_pie_q = px.pie(bw, values="Peso %", names="Bucket", hole=0.38,
                               color_discrete_sequence=list(BUCKET_COLORS.values()),
                               title="Allocazione macro")
            fig_pie_q.update_layout(height=320, margin=dict(l=0,r=0,t=35,b=0))
            st.plotly_chart(fig_pie_q, use_container_width=True)
        with tbl_c:
            st.dataframe(
                df_porto[["Bucket","Fondo","Peso %","Score","★ FIDA","Perf 3Y %"]],
                use_container_width=True, hide_index=True, height=320,
                column_config={
                    "Peso %": st.column_config.NumberColumn(format="%.1f"),
                    "Score": st.column_config.NumberColumn(format="%.2f"),
                    "Perf 3Y %": st.column_config.NumberColumn(format="%.2f"),
                }
            )

        # Heatmap correlazioni
        with st.expander("📊 Heatmap Correlazioni"):
            asset_list_q = [
                {"isin": r["ISIN"], "perf_1y": r.get("Perf 1Y %"),
                 "perf_3y": r.get("Perf 3Y %"), "perf_2022": None,
                 "perf_2023": None, "perf_2024": None}
                for _, r in df_porto.iterrows()
            ]
            with st.spinner("Recupero serie storiche per correlazioni..."):
                pd_dict_q = get_multiple_nav(asset_list_q, period="3y")
            if len(pd_dict_q) >= 2:
                prices_q_df = pd.DataFrame(pd_dict_q).pct_change().dropna()
                # Label leggibili
                isin_to_nome = {r["ISIN"]: r["Fondo"][:25] for _, r in df_porto.iterrows()}
                prices_q_df.columns = [isin_to_nome.get(c, c) for c in prices_q_df.columns]
                corr_q = prices_q_df.corr()
                fig_cq = px.imshow(corr_q, color_continuous_scale="RdBu_r",
                                    zmin=-1, zmax=1, text_auto=".2f")
                fig_cq.update_layout(height=max(350, len(corr_q)*38))
                st.plotly_chart(fig_cq, use_container_width=True)
            else:
                st.info("Dati storici non sufficienti. Le correlazioni saranno disponibili dopo il fetch NAV.")

        # Export
        st.markdown("---")
        exp_c1, exp_c2 = st.columns(2)
        weights_q = {r["ISIN"]: (r["Peso %"] or 0) / 100 for _, r in df_porto.iterrows()}
        metrics_q = {"Profilo": profilo, "Totale fondi": len(df_porto),
                     "Score medio": f"{df_porto['Score'].mean():.3f}",
                     "Generato": datetime.now().strftime("%d/%m/%Y %H:%M")}
        excel_q = export_portfolio_excel(weights_q, metrics_q, fund_df=df_unified,
                                          title=f"Portafoglio Qualità {profilo}")
        exp_c1.download_button("📥 Esporta Excel", data=excel_q,
            file_name=f"portafoglio_qualita_{profilo.lower()}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        pdf_q = export_portfolio_pdf(weights_q, metrics_q,
                                      title=f"Portafoglio Qualità — {profilo}")
        if pdf_q:
            exp_c2.download_button("📄 Esporta PDF", data=pdf_q,
                file_name=f"portafoglio_qualita_{profilo.lower()}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf")


# ===========================================================================
# ETF UNIVERSE
# ===========================================================================
elif nav == "🌐 ETF Universe":
    st.title("🌐 ETF Universe — Lista C")

    # Carica
    with st.spinner("Caricamento ETF Universe..."):
        try:
            df_etf = _load_etf_universe_cached()
        except Exception as e:
            st.error(f"Errore caricamento ETF: {e}")
            df_etf = pd.DataFrame()

    # Aggiungi ISINs
    with st.expander("➕ Aggiungi strumenti"):
        new_isins_txt = st.text_area(
            "ISINs aggiuntivi (uno per riga)",
            height=100,
            key="new_etf_isins",
        )
        if st.button("🔄 Fetch e aggiungi"):
            new_list = [x.strip() for x in new_isins_txt.strip().split("\n") if x.strip()]
            if new_list:
                with st.spinner(f"Recupero dati per {len(new_list)} ISINs..."):
                    progress_ph = st.empty()
                    def _prog(i, tot, isin):
                        progress_ph.progress(i / tot, f"Fetching {isin}...")
                    df_etf = fetch_etf_universe(
                        extra_isins=new_list,
                        progress_callback=_prog,
                    )
                    _load_etf_universe_cached.clear()
                progress_ph.empty()
                st.success(f"Aggiunti/aggiornati {len(new_list)} ISINs.")

    if df_etf.empty:
        st.info("Nessun ETF in lista. Clicca 'Fetch e aggiungi' per caricare la lista hardcoded.")
    else:
        # Filtri
        col1, col2, col3 = st.columns(3)
        cats = ["Tutte"] + sorted(df_etf["categoria"].dropna().unique().tolist()) if "categoria" in df_etf.columns else ["Tutte"]
        cat_filter = col1.selectbox("Categoria", cats)
        search_etf = col2.text_input("Cerca nome/ISIN")

        df_display = df_etf.copy()
        if cat_filter != "Tutte" and "categoria" in df_display.columns:
            df_display = df_display[df_display["categoria"] == cat_filter]
        if search_etf:
            mask = df_display.apply(
                lambda c: c.astype(str).str.contains(search_etf, case=False, na=False)
            ).any(axis=1)
            df_display = df_display[mask]

        col3.metric("ETF visualizzati", len(df_display))

        # Legenda fonti dati
        with st.expander("ℹ️ Fonti dati ETF"):
            st.markdown("""
| Campo | Fonte | Note |
|-------|-------|------|
| **Nome / Categoria** | Dataset statico hardcoded | Verificato manualmente su emittente |
| **TER %** | KID/KIID ufficiali emittente | Aggiornamento giugno 2025 |
| **Perf 1Y/3Y/5Y %** | **yfinance** (prezzi reali) | Calcolata da serie storica mensile su Borsa Milano/Xetra |
| **AUM mln €** | Stima indicativa | Non verificato — usare per ordinamento relativo |
| **Ticker** | Mappa ISIN→ticker hardcoded | 85/85 ETF coperti |
""")

        cols_etf = [c for c in ["isin", "ticker", "nome", "categoria", "ter",
                                 "perf_1y", "perf_3y", "perf_5y",
                                 "_fonte_perf", "_fonte_ter"]
                    if c in df_display.columns]
        col_config_etf = {
            "isin": st.column_config.TextColumn("ISIN", width="small"),
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "nome": st.column_config.TextColumn("Nome", width="large"),
            "categoria": st.column_config.TextColumn("Categoria"),
            "ter": st.column_config.NumberColumn("TER %", format="%.2f"),
            "perf_1y": st.column_config.NumberColumn("Perf 1Y %", format="%.2f"),
            "perf_3y": st.column_config.NumberColumn("Perf 3Y %/ann", format="%.2f"),
            "perf_5y": st.column_config.NumberColumn("Perf 5Y %/ann", format="%.2f"),
            "_fonte_perf": st.column_config.TextColumn("Fonte perf.", width="small"),
            "_fonte_ter": st.column_config.TextColumn("Fonte TER", width="small"),
        }
        st.dataframe(df_display[cols_etf], column_config=col_config_etf,
                     use_container_width=True, height=500)

        # Grafico distribuzione per categoria
        if "categoria" in df_etf.columns:
            cat_counts = df_etf["categoria"].value_counts().reset_index()
            cat_counts.columns = ["Categoria", "Conteggio"]
            fig_cat = px.bar(
                cat_counts, x="Conteggio", y="Categoria",
                orientation="h",
                color="Conteggio",
                color_continuous_scale="Blues",
                title="ETF per Categoria",
            )
            fig_cat.update_layout(height=400, showlegend=False,
                                   margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_cat, use_container_width=True)

        # Log errori
        from utils.etf_fetcher import ETF_ERRORS_LOG
        if ETF_ERRORS_LOG.exists():
            with st.expander("⚠️ Log errori JustETF"):
                errors = ETF_ERRORS_LOG.read_text(encoding="utf-8")
                st.code(errors[-2000:] if len(errors) > 2000 else errors)


# ===========================================================================
# IMPOSTAZIONI
# ===========================================================================
elif nav == "⚙️ Impostazioni":
    st.title("⚙️ Impostazioni")

    with st.expander("🎛️ Parametri ottimizzazione", expanded=True):
        col1, col2 = st.columns(2)
        st.session_state["risk_free_rate"] = col1.number_input(
            "Risk-free rate (%)", 0.0, 10.0,
            float(st.session_state["risk_free_rate"]), 0.1
        )
        st.session_state["min_weight"] = col1.slider("Peso minimo asset (%)", 0, 20,
                                                       st.session_state["min_weight"])
        st.session_state["max_weight"] = col2.slider("Peso massimo asset (%)", 10, 100,
                                                       st.session_state["max_weight"])
        st.session_state["opt_period"] = col2.selectbox(
            "Periodo storico",
            ["1Y", "3Y", "5Y"],
            index=["1Y", "3Y", "5Y"].index(st.session_state["opt_period"])
        )

    with st.expander("🗑️ Gestione Cache"):
        col1, col2, col3 = st.columns(3)
        if col1.button("Svuota cache NAV"):
            from utils.nav_fetcher import NAV_CACHE_FILE
            if NAV_CACHE_FILE.exists():
                NAV_CACHE_FILE.unlink()
            st.success("Cache NAV eliminata.")

        if col2.button("Svuota cache ETF"):
            from utils.etf_fetcher import ETF_CACHE_FILE, ETF_UNIVERSE_FILE
            for f in [ETF_CACHE_FILE, ETF_UNIVERSE_FILE]:
                if f.exists():
                    f.unlink()
            _load_etf_universe_cached.clear()
            st.success("Cache ETF eliminata.")

        if col3.button("Svuota cache preselection"):
            _load_all_data.clear()
            _load_preselection.clear()
            st.success("Cache preselection eliminata.")

    with st.expander("ℹ️ Informazioni"):
        st.markdown(f"""
        **Portafogli Efficienti v1.0**

        - **Fondi Terzi:** `tabella_fondi_arricchita.xlsx`
        - **Fondi Azimut:** `fondi_azimut_isin_completo_RATED.xlsx`
        - **ETF Universe:** `data/etf_universe.xlsx` (auto-generato)
        - **Cache NAV:** `data/nav_cache.json`

        Stack: Streamlit · PyPortfolioOpt · Plotly · yfinance · cloudscraper

        Dati in modalità Demo: {"✅ Sì" if st.session_state.get("demo_mode") else "❌ No"}
        """)
