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
from utils.optimizer import (
    compute_efficient_frontier, compute_black_litterman,
    estimate_max_drawdown, compute_bl_auto_views,
)
from utils.exporter import (
    export_portfolio_excel, export_portfolio_pdf,
    export_advisorelite_csv, export_advisorelite_excel,
)

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
    /* ── Sidebar fissa e sempre scrollabile ── */
    [data-testid="stSidebar"] {{
        background-color: {NAVY};
        overflow-y: scroll !important;   /* scroll = barra SEMPRE visibile */
        overflow-x: hidden !important;
    }}
    /* Il div interno deve poter crescere oltre lo schermo */
    [data-testid="stSidebar"] > div:first-child {{
        overflow-y: scroll !important;
        height: 100% !important;
        padding-bottom: 2rem;
    }}
    section[data-testid="stSidebar"] {{
        overflow-y: scroll !important;
    }}
    /* Scrollbar sidebar: sempre visibile, colore chiaro su sfondo navy */
    [data-testid="stSidebar"]::-webkit-scrollbar,
    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {{
        width: 8px !important;
        display: block !important;
    }}
    [data-testid="stSidebar"]::-webkit-scrollbar-track,
    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-track {{
        background: rgba(255,255,255,0.15) !important;
        border-radius: 4px;
    }}
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb,
    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {{
        background: rgba(255,255,255,0.55) !important;
        border-radius: 4px;
        min-height: 40px;
    }}
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover,
    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb:hover {{
        background: rgba(255,255,255,0.85) !important;
    }}
    /* Nasconde la freccia collasso sidebar */
    [data-testid="collapsedControl"],
    button[kind="header"],
    [data-testid="stSidebarCollapseButton"] {{
        display: none !important;
    }}
    /* Testo generico sidebar bianco — solo testo semplice, NON input */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: white !important;
    }}
    /* Radio button labels */
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio div {{
        color: white !important;
    }}
    /* Slider label e valore */
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {{
        color: white !important;
    }}
    /* Selectbox: lascia il testo dell'input visibile (sfondo bianco → testo scuro) */
    [data-testid="stSidebar"] .stSelectbox label {{
        color: white !important;
    }}
    [data-testid="stSidebar"] .stSelectbox [data-testid="stWidgetLabel"] {{
        color: white !important;
    }}
    /* Expander title in sidebar */
    [data-testid="stSidebar"] .streamlit-expanderHeader {{
        color: white !important;
        background-color: rgba(255,255,255,0.08) !important;
        border-radius: 6px;
    }}
    /* Warning/success in sidebar */
    [data-testid="stSidebar"] .stAlert p {{
        color: inherit !important;
    }}
    /* Scrollbar sidebar visibile */
    [data-testid="stSidebar"]::-webkit-scrollbar {{
        width: 6px;
    }}
    [data-testid="stSidebar"]::-webkit-scrollbar-track {{
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
    }}
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb {{
        background: rgba(255,255,255,0.4);
        border-radius: 4px;
    }}
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {{
        background: rgba(255,255,255,0.7);
    }}
    /* ── Scrollbar dataframe ── */
    [data-testid="stDataFrame"] > div {{
        overflow-x: auto !important;
        overflow-y: auto !important;
    }}
    [data-testid="stDataFrame"] ::-webkit-scrollbar {{
        height: 8px;
        width: 8px;
    }}
    [data-testid="stDataFrame"] ::-webkit-scrollbar-track {{
        background: #f1f1f1;
        border-radius: 4px;
    }}
    [data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {{
        background: {NAVY};
        border-radius: 4px;
    }}
    [data-testid="stDataFrame"] ::-webkit-scrollbar-thumb:hover {{
        background: #2E5090;
    }}
    /* ── Scrollbar globale pagina ── */
    ::-webkit-scrollbar {{ width: 7px; height: 7px; }}
    ::-webkit-scrollbar-track {{ background: #f5f5f5; }}
    ::-webkit-scrollbar-thumb {{
        background: #9aaac2;
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: {NAVY}; }}
    /* ── Card metriche ── */
    .metric-card {{
        background: {LIGHT_GRAY};
        border-left: 4px solid {NAVY};
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
    }}
    /* ── Header tabelle sticky ── */
    .stDataFrame thead th {{
        background-color: {NAVY} !important;
        color: white !important;
        position: sticky;
        top: 0;
        z-index: 1;
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
    _la_isins = lista_a["isin"].dropna().tolist() if not lista_a.empty else []
    lista_b = build_lista_b(df_unified, exclude_isins=_la_isins)
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


# ── AUTO-REFRESH ETF (notifica se dati > 24h) ──────────────────────────────
def _check_etf_auto_refresh():
    from pathlib import Path as _P
    _etf_f = _P("data/etf_universe.xlsx")
    if not _etf_f.exists() or st.session_state.get("_etf_refresh_notified"):
        return
    _age_h = (datetime.now() -
              datetime.fromtimestamp(_etf_f.stat().st_mtime)).total_seconds() / 3600
    if _age_h > 24:
        st.session_state["_etf_refresh_notified"] = True
        st.toast("⏰ ETF Universe: dati aggiornati da >24h. "
                 "Vai su ETF Universe → 🔄 per aggiornare.", icon="ℹ️")

_check_etf_auto_refresh()


# ---------------------------------------------------------------------------
# SIDEBAR — FILE UPLOAD + NAVIGAZIONE
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## 📊 {PAGE_TITLE}")
    st.markdown("---")

    # ── STATO DATI CARICATI (sempre visibile, sopra i file uploader) ─────
    _ts_t = st.session_state.get("_terzi_upload_ts")
    _ts_a = st.session_state.get("_azimut_upload_ts")
    _nm_t = st.session_state.get("_terzi_upload_name", "Fondi Terzi")
    _nm_a = st.session_state.get("_azimut_upload_name", "Fondi Azimut")

    if _ts_t or _ts_a:
        # Dati caricati — mostra riepilogo evidente
        _last_ts = max(t for t in [_ts_t, _ts_a] if t is not None)
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.12);border-radius:8px;"
            f"padding:10px 12px;margin-bottom:8px;'>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.7);'>ULTIMO CARICAMENTO</div>"
            f"<div style='font-size:15px;font-weight:bold;color:white;'>"
            f"{_last_ts.strftime('%d/%m/%Y %H:%M')}</div>"
            + (f"<div style='font-size:10px;color:rgba(255,255,255,0.6);margin-top:2px;'>"
               f"{'✅ ' + _nm_t[:28] if _ts_t else ''}"
               f"{'<br>✅ ' + _nm_a[:28] if _ts_a else ''}</div>" )
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:rgba(255,180,0,0.18);border-radius:8px;"
            "padding:8px 12px;margin-bottom:8px;font-size:12px;color:#FFD080;'>"
            "⚠️ Nessun file caricato<br><span style='font-size:10px;opacity:0.8'>"
            "Apri «Carica file dati» qui sotto</span></div>",
            unsafe_allow_html=True,
        )

    # Upload file Excel
    with st.expander("📂 Carica file dati", expanded=(_ts_t is None or _ts_a is None)):
        st.caption("I dati restano in memoria fino al prossimo caricamento.")
        up_terzi = st.file_uploader(
            "Fondi Terzi",
            type=["xlsx"],
            key="up_terzi",
            help="tabella_fondi_arricchita.xlsx",
        )
        up_azimut = st.file_uploader(
            "Fondi Azimut",
            type=["xlsx"],
            key="up_azimut",
            help="fondi_azimut_isin_completo_RATED.xlsx",
        )
        _has_new = (up_terzi is not None or up_azimut is not None)
        if _has_new:
            if st.button("🔄 Ricarica dati con nuovi file", use_container_width=True,
                         type="primary"):
                _load_all_data.clear()
                _load_preselection.clear()
                st.rerun()
        if st.button("🗑️ Dimentica file caricati", use_container_width=True,
                     help="Cancella i dati dalla memoria e torna in modalità Demo"):
            for _k in ["_terzi_bytes_cached","_azimut_bytes_cached",
                       "_terzi_upload_ts","_azimut_upload_ts",
                       "_terzi_upload_name","_azimut_upload_name"]:
                st.session_state.pop(_k, None)
            _load_all_data.clear()
            _load_preselection.clear()
            st.rerun()

    st.markdown("---")


# ---------------------------------------------------------------------------
# CARICAMENTO DATI — con persistenza bytes in session_state
# ---------------------------------------------------------------------------
# Se l'utente ha caricato nuovi file → salva bytes + timestamp
try:
    _up_t = st.session_state.get("up_terzi")
    if _up_t is not None:
        _new_bytes = _up_t.getvalue()
        if _new_bytes:
            st.session_state["_terzi_bytes_cached"]   = _new_bytes
            st.session_state["_terzi_upload_ts"]      = datetime.now()
            st.session_state["_terzi_upload_name"]    = _up_t.name
except Exception:
    pass

try:
    _up_a = st.session_state.get("up_azimut")
    if _up_a is not None:
        _new_bytes = _up_a.getvalue()
        if _new_bytes:
            st.session_state["_azimut_bytes_cached"]  = _new_bytes
            st.session_state["_azimut_upload_ts"]     = datetime.now()
            st.session_state["_azimut_upload_name"]   = _up_a.name
except Exception:
    pass

# Usa i bytes dalla cache persistente (rimangono anche se il file uploader viene chiuso)
_terzi_bytes  = st.session_state.get("_terzi_bytes_cached")
_azimut_bytes = st.session_state.get("_azimut_bytes_cached")

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
        st.warning("⚠️ Modalità Demo")
    else:
        _tot = len(df_terzi) + len(df_azimut)
        _ts_disp = st.session_state.get("_terzi_upload_ts") or st.session_state.get("_azimut_upload_ts")
        _data_str = _ts_disp.strftime("%d/%m/%Y") if _ts_disp else "—"
        st.success(f"✅ {_tot:,} fondi  ·  {_data_str}")
    st.markdown("---")
    nav = st.radio(
        "Navigazione",
        options=[
            "🏠 Home",
            "📈 Frontiera Efficiente",
            "⭐ Portafoglio Qualità",
            "🔀 Comparatore",
            "🌐 ETF Universe",
            "📖 Guida",
            "⚙️ Impostazioni",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ⚙️ Impostazioni rapide")
    st.session_state["risk_free_rate"] = st.slider(
        "Risk-free rate (%)", 0.0, 5.0,
        float(st.session_state["risk_free_rate"]), 0.1,
    )
    st.session_state["opt_period"] = st.selectbox(
        "Periodo ottimizzazione",
        ["1Y", "3Y", "5Y"],
        index=["1Y", "3Y", "5Y"].index(st.session_state["opt_period"]),
    )

    st.markdown("---")
    # ── REBOOT CACHE ─────────────────────────────────────────────────────
    with st.expander("🔁 Cache & Portafogli salvati"):
        if st.button("🗑️ Svuota tutta la cache", use_container_width=True,
                     help="Forza il ricaricamento di fondi, ETF e liste. Cancella anche nav_cache.json"):
            import os as _os2
            from pathlib import Path as _P2
            _load_all_data.clear()
            _load_preselection.clear()
            _load_etf_universe_cached.clear()
            # Cancella cache su disco (nav + etf) — contengono serie con vecchi valori
            for _cf2 in ["data/nav_cache.json","data/etf_cache.json","data/preselection_cache.json"]:
                _fp2 = _P2(_cf2)
                if _fp2.exists():
                    try: _os2.remove(_fp2)
                    except Exception: pass
            for _k in ["fe_result","fe_price_dict","bl_result","fe_selected_isins",
                        "fe_ac_map","fe_ac_target"]:
                st.session_state.pop(_k, None)
            st.toast("✅ Cache completamente svuotata (Streamlit + file su disco)")
            st.rerun()

        st.markdown("---")
        # ── SALVATAGGIO PORTAFOGLIO ───────────────────────────────────────
        import json as _json
        st.caption("**Salva / Carica portafoglio**")

        def _build_save_payload() -> dict:
            """Serializza lo stato corrente dei portafogli in JSON."""
            payload = {
                "version": "1.0",
                "saved_at": datetime.now().isoformat(),
                "fe_selected_isins": st.session_state.get("fe_selected_isins", []),
                "fe_ac_target": st.session_state.get("fe_ac_target", {}),
                "fe_ac_map": st.session_state.get("fe_ac_map", {}),
                "profilo": st.session_state.get("profilo", "Equilibrato"),
                "fondi_per_bucket": st.session_state.get("fondi_per_bucket", 4),
                "risk_free_rate": st.session_state.get("risk_free_rate", 2.5),
                "opt_period": st.session_state.get("opt_period", "3Y"),
                "min_weight": st.session_state.get("min_weight", 3),
                "max_weight": st.session_state.get("max_weight", 30),
                "pq_locks": list(st.session_state.get("pq_locks", set())),
                "pq_replacements": st.session_state.get("pq_replacements", {}),
            }
            # Includi pesi Max Sharpe se disponibili
            fe_res = st.session_state.get("fe_result", {})
            ms = fe_res.get("max_sharpe", {})
            if ms and "error" not in ms:
                payload["last_portfolio_weights"] = dict(ms["weights"])
                payload["last_portfolio_metrics"] = {
                    "ret": ms.get("ret"), "vol": ms.get("vol"), "sharpe": ms.get("sharpe")
                }
            return payload

        save_json = _json.dumps(_build_save_payload(), indent=2, ensure_ascii=False)
        st.download_button(
            "💾 Scarica portafoglio (.json)",
            data=save_json.encode("utf-8"),
            file_name=f"portafoglio_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )

        up_portfolio = st.file_uploader("📂 Carica portafoglio salvato",
                                         type=["json"], key="up_portfolio")
        if up_portfolio and st.button("↩️ Ripristina", use_container_width=True):
            try:
                payload = _json.loads(up_portfolio.getvalue().decode("utf-8"))
                for k in ["fe_selected_isins","fe_ac_target","fe_ac_map",
                           "profilo","fondi_per_bucket","risk_free_rate",
                           "opt_period","min_weight","max_weight",
                           "pq_replacements"]:
                    if k in payload:
                        st.session_state[k] = payload[k]
                if "pq_locks" in payload:
                    st.session_state["pq_locks"] = set(payload["pq_locks"])
                st.toast(f"✅ Portafoglio del {payload.get('saved_at','?')[:10]} ripristinato")
                st.rerun()
            except Exception as _e:
                st.error(f"Errore nel caricamento: {_e}")


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
    col3.metric("Lista Generalisti", len(lista_a),
                help="Top 100 fondi generalisti selezionati per Score Qualità (su tutti i fondi terzi)")
    col4.metric("Lista Tematici", len(lista_b),
                help="Top 100 fondi tematici/specializzati selezionati per Score Qualità")

    st.markdown("---")
    st.subheader("Liste Preselezionate")
    tab_a, tab_b = st.tabs(["📋 Fondi Generalisti", "🎯 Fondi Tematici"])

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
    st.caption(
        "Configura allocazione, numero di strumenti e vincoli — "
        "l'app seleziona automaticamente i migliori fondi/ETF e calcola "
        "Max Sharpe, Min Volatilità e Black-Litterman in un clic."
    )


    # ── POOL GLOBALE (interno) ───────────────────────────────────────────────
    from utils.etf_tickers import (get_dividend_stocks_df, DIVIDEND_STOCKS,
                                    ITALIAN_STOCKS, ISIN_TO_TICKER)
    from utils.etf_static import ETF_STATIC as _ETF_STATIC_LIST

    _all_fund_pool: dict = {}
    _etf_nome_map = {e["isin"]: e["nome"] for e in _ETF_STATIC_LIST}

    def _pool_add(isin: str, info: dict):
        if isin and isin not in _all_fund_pool:
            _all_fund_pool[isin] = info

    def _pool_from_df(df: pd.DataFrame):
        for _, r in df.iterrows():
            isin = str(r.get("isin", r.get("ISIN", ""))).strip()
            if isin:
                _pool_add(isin, r.to_dict())

    if not lista_a.empty: _pool_from_df(lista_a)
    if not lista_b.empty: _pool_from_df(lista_b)

    try:
        df_etf_fe = _load_etf_universe_cached()
        _pool_from_df(df_etf_fe)
    except Exception:
        from utils.etf_static import get_static_etf_df
        df_etf_fe = get_static_etf_df()
        _pool_from_df(df_etf_fe)

    for _isin, _info in ITALIAN_STOCKS.items():
        _pool_add(_isin, {"isin": _isin, "nome": _info["nome"],
                          "classificazione": f"Azione \u2014 {_info['settore']}",
                          "ticker": _info["ticker"]})
    for _isin, _info in DIVIDEND_STOCKS.items():
        _pool_add(_isin, {"isin": _isin, "nome": _info["nome"],
                          "classificazione": f"Dividendo \u2014 {_info['settore']} ({_info['paese']})",
                          "ticker": _info["ticker"]})

    # ETF fallback per asset class
    _ETF_FALLBACK = {
        "Azioni":        ["IE00B4L5Y983","IE00BK5BQT80","IE00B5BMR087",
                          "LU0908500753","IE00BKM4GZ66"],
        "Obbligazioni":  ["IE00B4WXJJ64","IE00B3F81R35","IE00B66F4759",
                          "IE00B2NPKV68","IE00B3T9LM79"],
        "Bilanciato":    ["IE00B4L5Y983","IE00B4WXJJ64"],
        "Materie Prime": ["IE00BD6FTQ80","LU1829218749","GB00B15KXQ89"],
    }

    import re as _re
    _CLASS_TO_MACRO = [
        ("Materie Prime", ["materie prime","commodity","commodities","energy",
                           "metals","oro","gold","petrolio","oil"]),
        ("Obbligazioni",  ["obbligazionari","obbligazionario","bond",
                           "fixed income","reddito fisso","high yield",
                           "corporate bond","government bond","duration"]),
        ("Bilanciato",    ["bilanciati","bilanciato","balanced","flessibili",
                           "flessibile","multi-asset","allocation","ritorno assoluto"]),
        ("Azioni",        ["azionari","azionario","equity","azioni","stock",
                           "tematici","tematico","small cap","large cap","dividend"]),
    ]

    def _macro_from_class(cl):
        cl = str(cl).lower()
        for macro, kws in _CLASS_TO_MACRO:
            for kw in kws:
                if _re.search(r"(?<![a-z])" + _re.escape(kw) + r"(?![a-z])", cl):
                    return macro
        return None

    @st.cache_data(ttl=3600, show_spinner=False)
    def _get_azimut_ranked(azimut_json: str) -> list:
        import io as _io2
        from utils.scoring import compute_scores_df as _css2
        from utils.data_loader import build_unified_fund_df as _buf
        try:
            _az_df = pd.read_json(_io2.StringIO(azimut_json), orient="records")
            if _az_df.empty: return []
            _az_u = _buf(pd.DataFrame(), _az_df)
            if _az_u.empty: return []
            _az_u = _css2(_az_u)
            return _az_u.sort_values("score_qualita", ascending=False)["isin"].dropna().tolist()
        except Exception:
            return []

    _MONETARY_KW = ["money market","monetari","monetario","cash",
                    "insticash","overnight","liquidit","lvnav","vnav"]

    def _is_monetary(cl: str, nome: str) -> bool:
        t = (str(cl) + " " + str(nome)).lower()
        return any(k in t for k in _MONETARY_KW)

    def _pick_assets(macro: str, n: int, prefer_f: bool = True,
                     allow_etf: bool = True) -> list:
        """
        Seleziona asset per bucket macro.
        I fondi dall'Excel sono SEMPRE tentati come sorgente primaria se df_unified
        è caricato. I parametri controllano solo la preferenza e il supplemento ETF:
        - prefer_f=True : fondi prima degli ETF (se entrambi disponibili)
        - allow_etf=True : gli ETF completano la selezione se i fondi non bastano
        """
        # Recupera candidati fondi da df_unified (sempre, se disponibile)
        fund_candidates = []
        if not df_unified.empty:
            from utils.scoring import compute_scores_df
            _fu = compute_scores_df(df_unified.copy())
            _fu["_macro_auto"] = _fu["classificazione"].apply(_macro_from_class)
            _fu = _fu[_fu["_macro_auto"] == macro]
            _fu = _fu[~_fu.apply(
                lambda r: _is_monetary(r.get("classificazione",""), r.get("nome","")), axis=1
            )]
            _fu = _fu.sort_values("score_qualita", ascending=False)
            for _, _r in _fu.iterrows():
                _isin_r = str(_r.get("isin",""))
                if _isin_r:
                    _pool_add(_isin_r, _r.to_dict())
            _seen: set = set()
            for _, r in _fu.iterrows():
                casa = str(r.get("casa",""))
                if casa and casa in _seen: continue
                fund_candidates.append(r["isin"])
                if casa: _seen.add(casa)
                if len(fund_candidates) >= n * 3: break

        etf_candidates = _ETF_FALLBACK.get(macro, []) if allow_etf else []

        if prefer_f or not allow_etf:
            # Fondi prima; ETF solo se abilitati e mancano slot
            selected = fund_candidates[:n]
            if allow_etf:
                for e in etf_candidates:
                    if e not in selected and len(selected) < n:
                        selected.append(e)
        else:
            # Mix paritetico: metà fondi, metà ETF (o tutto fondi se ETF mancano)
            if fund_candidates:
                half = max(n // 2, 1)
                selected = fund_candidates[:half]
                for e in etf_candidates:
                    if e not in selected and len(selected) < n:
                        selected.append(e)
            else:
                # Nessun fondo → usa ETF puri come fallback
                selected = etf_candidates[:n]
        return selected

    def _inject_azimut(selected: list, n_min: int,
                       az_sel: list | None = None,
                       ob_sel: list | None = None) -> tuple:
        """
        Sostituisce n_min fondi terzi con fondi Azimut (il totale NON aumenta).
        Deduplicazione per nome fondo (classi diverse dello stesso fondo → prende la migliore).
        """
        if n_min <= 0:
            return selected, []
        if df_azimut.empty:
            return selected, []

        import re as _re
        from utils.data_loader import AZIMUT_COLS as _ACOLS
        from utils.scoring import compute_score as _cscore

        _c_isin = _ACOLS.get("isin", "ISIN")
        _c_nome = _ACOLS.get("nome", "FONDO AZIMUT")
        _c_cl   = _ACOLS.get("classificazione", "CLASSIFICAZIONE FIDA")
        _c_p3   = _ACOLS.get("perf_3y", "PERF 3Y")
        _c_p1   = _ACOLS.get("perf_1y", "PERF 1Y")
        _c_fida = _ACOLS.get("stelle_fida", "STELLE FIDA")

        if _c_isin not in df_azimut.columns:
            return selected, []

        # Chiave di deduplicazione: prime 4 parole significative del nome
        _AZ_STOP = {"a","b","c","d","e","f","p","r","i","ii","iii","iv",
                    "acc","eur","usd","gbp","chf","inc","cap","ret","dist",
                    "class","cl","fund","sicav","az"}
        def _az_key(nome):
            words = _re.sub(r"[^a-z0-9\s]", " ", str(nome).lower()).split()
            mw = [w for w in words if w not in _AZ_STOP and len(w) > 1]
            return " ".join(mw[:4])

        # ── Costruisce pool Azimut deduplicato per nome fondo ─────────────
        _best_by_name: dict = {}
        for _, _r in df_azimut.iterrows():
            _isin_r = str(_r.get(_c_isin, "") or "").strip()
            if not _isin_r or len(_isin_r) < 8:
                continue
            _nome_r = str(_r.get(_c_nome, _isin_r) or _isin_r)
            try:
                _p3   = float(_r.get(_c_p3, 0) or 0)
                _p1   = float(_r.get(_c_p1, 0) or 0)
                _fida = _r.get(_c_fida)
                _sc   = _cscore(_p3, _p1, volatility=0.0, fida_stars=_fida)
            except Exception:
                _sc = 0.0
            _cl    = str(_r.get(_c_cl, "") or "")
            _macro = _macro_from_class(_cl) or "Altro"
            _entry = {
                "isin": _isin_r, "nome": _nome_r, "casa": "Azimut",
                "classificazione": _cl,
                "perf_3y": float(_r.get(_c_p3, 0) or 0),
                "perf_1y": float(_r.get(_c_p1, 0) or 0),
                "volatilita": None, "rating_fida": _fida,
                "score_qualita": _sc, "_source": "azimut", "_macro": _macro,
            }
            _k = _az_key(_nome_r)
            if _k not in _best_by_name or _sc > _best_by_name[_k]["score_qualita"]:
                _best_by_name[_k] = _entry

        _az_pool = sorted(_best_by_name.values(), key=lambda x: x["score_qualita"], reverse=True)
        if not _az_pool:
            return selected, []

        # ── Conta Azimut già presenti in selected ─────────────────────────
        _az_isins   = {d["isin"] for d in _az_pool}
        _az_already = [i for i in selected if i in _az_isins]
        _need       = n_min - len(_az_already)

        _result = list(selected)
        _forced = list(_az_already[:n_min])

        # Aggiorna pool per quelli già presenti
        for _entry in _az_pool:
            if _entry["isin"] in set(selected):
                _pool_add(_entry["isin"], _entry)

        if _need <= 0:
            return _result, _forced

        # ── Quota per bucket (Azioni / Obbligazioni) ──────────────────────
        _n_az_bucket = len(az_sel or [])
        _n_ob_bucket = len(ob_sel or [])
        _tot_req     = max(_n_az_bucket + _n_ob_bucket, 1)
        _n_az_to_add = round(_need * _n_az_bucket / _tot_req)
        _n_ob_to_add = _need - _n_az_to_add

        def _replace_with_azimut(bucket_list, quota, macro_filter):
            """Sostituisce 'quota' fondi terzi del bucket con fondi Azimut del macro corrispondente."""
            _q = quota
            _added_local = 0
            # Fondi terzi sostituibili: in bucket_list, non Azimut, non già forzati
            _replaceable = [i for i in (bucket_list or [])
                            if i in _result and i not in _az_isins and i not in _forced]
            for _entry in _az_pool:
                if _q <= 0 or not _replaceable:
                    break
                if _entry["isin"] in _result:
                    continue
                if _entry["_macro"] != macro_filter:
                    continue
                # Rimuovi l'ultimo terzo del bucket (il meno prioritario)
                _out = _replaceable.pop()
                _result.remove(_out)
                _result.append(_entry["isin"])
                _forced.append(_entry["isin"])
                _pool_add(_entry["isin"], _entry)
                _q -= 1
                _added_local += 1
            return _added_local

        # Conta quanti Azimut disponibili per macro PRIMA di sostituire
        _az_avail_az = sum(1 for e in _az_pool if e["_macro"] == "Azioni" and e["isin"] not in _result)
        _az_avail_ob = sum(1 for e in _az_pool if e["_macro"] == "Obbligazioni" and e["isin"] not in _result)

        # Ridistribuisce le quote se Azimut non ha abbastanza fondi in un bucket:
        # i "mancanti" obbligazionari vengono compensati con più azioni Azimut
        # (sostituendo posti AZIONI, non obbligazioni → le obbligazioni restano intatte)
        _real_ob_add = min(_n_ob_to_add, _az_avail_ob)
        _real_az_add = min(_n_az_to_add + (_n_ob_to_add - _real_ob_add), _az_avail_az)

        _added  = _replace_with_azimut(az_sel, _real_az_add, "Azioni")
        _added += _replace_with_azimut(ob_sel, _real_ob_add, "Obbligazioni")

        # Quota residua (Azimut di qualsiasi macro → sostituisce SOLO posti azioni,
        # per non toccare le obbligazioni selezionate)
        _still_need = _need - _added
        if _still_need > 0:
            _az_replaceable = [i for i in (az_sel or [])
                               if i in _result and i not in _az_isins and i not in _forced]
            if not _az_replaceable:  # nessun posto azioni disponibile → prende qualsiasi
                _az_replaceable = [i for i in _result if i not in _az_isins and i not in _forced
                                   and i not in (ob_sel or [])]  # evita obbligazioni
            for _entry in _az_pool:
                if _still_need <= 0 or not _az_replaceable:
                    break
                if _entry["isin"] in _result:
                    continue
                _out = _az_replaceable.pop()
                _result.remove(_out)
                _result.append(_entry["isin"])
                _forced.append(_entry["isin"])
                _pool_add(_entry["isin"], _entry)
                _still_need -= 1

        # Aggiorna sorgente pool per tutti gli iniettati
        for _isin in _forced:
            if _isin not in _all_fund_pool:
                _rw = df_unified[df_unified["isin"] == _isin] if not df_unified.empty else pd.DataFrame()
                if not _rw.empty:
                    _pool_add(_isin, _rw.iloc[0].to_dict())
            if _isin in _all_fund_pool:
                _all_fund_pool[_isin]["_source"] = "azimut"
                _all_fund_pool[_isin]["casa"] = "Azimut"

        return _result, _forced

    # Conteggio fondi Azimut disponibili (deduplicato per nome fondo)
    _n_az_avail = 0
    try:
        if not df_azimut.empty:
            import re as _re2
            _AZ_STOP2 = {"a","b","c","d","e","f","p","r","i","ii","iii","iv",
                         "acc","eur","usd","gbp","chf","inc","cap","ret","dist",
                         "class","cl","fund","sicav","az"}
            _c_nome2 = "FONDO AZIMUT"
            if _c_nome2 in df_azimut.columns:
                _names = df_azimut[_c_nome2].dropna().astype(str)
                def _az_key2(n):
                    w = _re2.sub(r"[^a-z0-9\s]"," ",n.lower()).split()
                    return " ".join([x for x in w if x not in _AZ_STOP2 and len(x)>1][:4])
                _n_az_avail = _names.apply(_az_key2).nunique()
            else:
                _n_az_avail = len(df_azimut)
        elif not df_unified.empty and "_source" in df_unified.columns:
            _n_az_avail = int((df_unified["_source"]=="azimut").sum())
    except Exception:
        _n_az_avail = 0 if df_azimut.empty else len(df_azimut)

    # =========================================================================
    # FORM GUIDATO
    # =========================================================================
    _prev = st.session_state.get("_fe_form_vals", {})

    with st.form("fe_guided_form", clear_on_submit=False):

        # Step 1: Allocazione
        st.markdown("**1 \u2014 Allocazione target (%)**")
        _ac1, _ac2, _ac3 = st.columns(3)
        _pct_az = _ac1.number_input("Azioni %",        0, 100, _prev.get("pct_az", 60), 5)
        _pct_ob = _ac2.number_input("Obbligazioni %",  0, 100, _prev.get("pct_ob", 30), 5)
        _pct_mp = _ac3.number_input("Materie Prime %", 0, 100, _prev.get("pct_mp", 10), 5)
        _pct_bi = 0   # rimosso: i bilanciati rientrano nell'allocazione obbligazionaria
        _total_pct = int(_pct_az) + int(_pct_ob) + int(_pct_mp)
        _tc, _ = st.columns([1, 3])
        if _total_pct == 100:
            _tc.success(f"Totale: {_total_pct}%  OK")
        else:
            _tc.warning(f"Totale: {_total_pct}%  (deve essere 100%)")

        st.markdown("---")

        # Step 2: Strumenti
        st.markdown("**2 \u2014 Strumenti**")
        _sc1, _sc2, _sc3 = st.columns(3)
        _n_az_str = _sc1.number_input("Strumenti azionari",       2, 12, _prev.get("n_az_str", 4), 1)
        _n_ob_str = _sc2.number_input("Strumenti obbligazionari", 2, 12, _prev.get("n_ob_str", 4), 1)
        _n_mp_str = _sc3.number_input("Strumenti Mat. Prime",     0,  6, _prev.get("n_mp_str", 2), 1,
                                       disabled=(int(_pct_mp)==0))
        _n_bi_str = 0   # bilanciato rimosso
        _sf1, _sf2, _sf3 = st.columns(3)
        _az_label = f"Min fondi Azimut  ({_n_az_avail} disponibili)" if _n_az_avail else "Min fondi Azimut"
        _n_min_az = _sf1.number_input(_az_label, 0, max(_n_az_avail, 20),
                                       _prev.get("n_min_az", 0), 1,
                                       help="I migliori per Score Qualita' vengono inclusi automaticamente")
        _use_fondi  = _sf2.checkbox("Fondi prima degli ETF", value=_prev.get("use_fondi", True),
                                    help="I fondi caricati dall'Excel vengono selezionati prima degli ETF. "
                                         "Disattiva per un mix paritetico fondi/ETF.")
        _use_etf    = _sf2.checkbox("Includi ETF come supplemento", value=_prev.get("use_etf", True),
                                    help="Se non ci sono abbastanza fondi per riempire gli slot richiesti, "
                                         "vengono aggiunti ETF di mercato. Disattiva per usare solo fondi dall'Excel.")
        _use_azioni = _sf3.checkbox("Includi azioni FTSE MIB", value=_prev.get("use_azioni", False))

        st.markdown("---")

        # Step 3: Parametri
        st.markdown("**3 \u2014 Parametri ottimizzazione**")
        _pc1, _pc2, _pc3 = st.columns(3)
        _min_w_pct = _pc1.slider("Peso minimo per strumento %",  1, 15,
                                  _prev.get("min_w_pct", 3),  1)
        _max_w_pct = _pc2.slider("Peso massimo per strumento %", 10, 50,
                                  _prev.get("max_w_pct", 30), 5)
        _periodo   = _pc3.selectbox("Periodo storico (ETF/azioni)",
                                     ["1Y", "3Y", "5Y"],
                                     index=["1Y","3Y","5Y"].index(_prev.get("periodo","3Y")))
        st.caption(
            "Black-Litterman viene calcolato automaticamente usando lo Score Qualita' dei fondi. "
            "Non sono necessari input manuali."
        )

        # Bottoni
        _btn1, _btn2 = st.columns([3, 1])
        _submitted  = _btn1.form_submit_button(
            "Costruisci e calcola portafoglio",
            type="primary", use_container_width=True,
            disabled=(_total_pct != 100),
        )
        _reset_btn = _btn2.form_submit_button("Reset", use_container_width=True)

    # Selezione manuale avanzata
    with st.expander("🔍 Selezione manuale avanzata", expanded=False):
        st.caption("Sfoglia le liste fondi e aggiungi direttamente alla selezione corrente.")

        _adv_tab1, _adv_tab2, _adv_tab3, _adv_tab4 = st.tabs([
            "📋 Lista Generalisti", "🎯 Lista Tematici", "🔵 Fondi Azimut", "✏️ ISIN manuale"
        ])

        def _adv_add_isins(isins_to_add: list):
            """Aggiunge ISIN alla selezione corrente e al pool."""
            _cur = list(st.session_state.get("fe_selected_isins", []))
            _added_n = 0
            for _isin_a in isins_to_add:
                if _isin_a and _isin_a not in _cur:
                    _cur.append(_isin_a)
                    _added_n += 1
            st.session_state["fe_selected_isins"] = _cur
            return _added_n

        # ── Tab 1: Lista Generalisti ──────────────────────────────────────
        with _adv_tab1:
            st.caption("Top 100 fondi generalisti/globali per Score Qualità.")
            try:
                from utils.constraints import build_lista_generalisti
                _gen_df = build_lista_generalisti(df_unified, n=100) if not df_unified.empty else pd.DataFrame()
                if not _gen_df.empty:
                    _gen_opts = []
                    for _, _r in _gen_df.iterrows():
                        _isin_g = str(_r.get("isin",""))
                        _nome_g = str(_r.get("nome",""))[:50]
                        _cl_g   = str(_r.get("classificazione",""))[:30]
                        _sc_g   = float(_r.get("score_qualita",0) or 0)
                        _p3_g   = _r.get("perf_3y")
                        _p3_str = f" | 3Y:{_p3_g:.1f}%" if _p3_g is not None else ""
                        _gen_opts.append((_isin_g, f"{_nome_g}  [{_cl_g}  score:{_sc_g:.1f}{_p3_str}]"))
                    _gen_sel = st.multiselect(
                        "Seleziona fondi generalisti da aggiungere",
                        options=[o[0] for o in _gen_opts],
                        format_func=lambda x: next((o[1] for o in _gen_opts if o[0]==x), x),
                        key="adv_gen_sel",
                    )
                    if st.button("➕ Aggiungi generalisti selezionati", key="adv_gen_add"):
                        for _ig in _gen_sel:
                            _rw = _gen_df[_gen_df["isin"]==_ig]
                            if not _rw.empty:
                                _pool_add(_ig, _rw.iloc[0].to_dict())
                        n = _adv_add_isins(_gen_sel)
                        st.toast(f"✅ {n} fondi aggiunti")
                        st.rerun()
                else:
                    st.info("Lista generalisti non disponibile (carica i file fondi).")
            except Exception as _e:
                st.warning(f"Lista generalisti: {_e}")

        # ── Tab 2: Lista Tematici ─────────────────────────────────────────
        with _adv_tab2:
            st.caption("Top 100 fondi tematici/specializzati per Score Qualità.")
            try:
                from utils.constraints import build_lista_tematici
                _tem_df = build_lista_tematici(df_unified, n=100) if not df_unified.empty else pd.DataFrame()
                if not _tem_df.empty:
                    _tem_opts = []
                    for _, _r in _tem_df.iterrows():
                        _isin_t = str(_r.get("isin",""))
                        _nome_t = str(_r.get("nome",""))[:50]
                        _cl_t   = str(_r.get("classificazione",""))[:30]
                        _sc_t   = float(_r.get("score_qualita",0) or 0)
                        _p3_t   = _r.get("perf_3y")
                        _p3_str2 = f" | 3Y:{_p3_t:.1f}%" if _p3_t is not None else ""
                        _tem_opts.append((_isin_t, f"{_nome_t}  [{_cl_t}  score:{_sc_t:.1f}{_p3_str2}]"))
                    _tem_sel = st.multiselect(
                        "Seleziona fondi tematici da aggiungere",
                        options=[o[0] for o in _tem_opts],
                        format_func=lambda x: next((o[1] for o in _tem_opts if o[0]==x), x),
                        key="adv_tem_sel",
                    )
                    if st.button("➕ Aggiungi tematici selezionati", key="adv_tem_add"):
                        for _it in _tem_sel:
                            _rw = _tem_df[_tem_df["isin"]==_it]
                            if not _rw.empty:
                                _pool_add(_it, _rw.iloc[0].to_dict())
                        n = _adv_add_isins(_tem_sel)
                        st.toast(f"✅ {n} fondi aggiunti")
                        st.rerun()
                else:
                    st.info("Lista tematici non disponibile (carica i file fondi).")
            except Exception as _e:
                st.warning(f"Lista tematici: {_e}")

        # ── Tab 3: Fondi Azimut (top 50 per Score Qualità) ───────────────
        with _adv_tab3:
            st.caption(
                "Top 50 fondi Azimut per Score Qualità "
                "(colonna AUM non disponibile nel file — usato Score come proxy)."
            )
            try:
                if not df_azimut.empty:
                    from utils.data_loader import AZIMUT_COLS as _ACOLS2
                    from utils.scoring import compute_score as _cs2
                    _c_isin3 = _ACOLS2.get("isin","ISIN")
                    _c_nome3 = _ACOLS2.get("nome","FONDO AZIMUT")
                    _c_cl3   = _ACOLS2.get("classificazione","CLASSIFICAZIONE FIDA")
                    _c_p3b   = _ACOLS2.get("perf_3y","PERF 3Y")
                    _c_p1b   = _ACOLS2.get("perf_1y","PERF 1Y")
                    _c_fb    = _ACOLS2.get("stelle_fida","STELLE FIDA")

                    import re as _re3
                    _AZ_STOP3 = {"a","b","c","d","e","f","p","r","i","ii","iii",
                                 "acc","eur","usd","inc","cap","ret","class","cl","az"}
                    def _az_key3(n):
                        w = _re3.sub(r"[^a-z0-9\s]"," ",str(n).lower()).split()
                        return " ".join([x for x in w if x not in _AZ_STOP3 and len(x)>1][:4])

                    _az_top_rows, _seen3 = [], {}
                    for _, _r in df_azimut.iterrows():
                        _isin3 = str(_r.get(_c_isin3,"") or "").strip()
                        if not _isin3 or len(_isin3)<8: continue
                        _nome3 = str(_r.get(_c_nome3,_isin3) or _isin3)
                        _k3 = _az_key3(_nome3)
                        try:
                            _p3v = float(_r.get(_c_p3b,0) or 0)
                            _p1v = float(_r.get(_c_p1b,0) or 0)
                            _sc3 = _cs2(_p3v, _p1v, volatility=0.0, fida_stars=_r.get(_c_fb))
                        except Exception:
                            _sc3 = 0.0
                        _cl3v = str(_r.get(_c_cl3,"") or "")
                        _entry3 = {"isin":_isin3,"nome":_nome3,"classificazione":_cl3v,
                                   "perf_3y":float(_r.get(_c_p3b,0) or 0),
                                   "perf_1y":float(_r.get(_c_p1b,0) or 0),
                                   "score_qualita":_sc3,"_source":"azimut","casa":"Azimut"}
                        if _k3 not in _seen3 or _sc3 > _seen3[_k3]["score_qualita"]:
                            _seen3[_k3] = _entry3
                    _az_top50 = sorted(_seen3.values(), key=lambda x: x["score_qualita"], reverse=True)[:50]

                    _az_opts = []
                    for _e in _az_top50:
                        _p3s = f" | 3Y:{_e['perf_3y']:.1f}%" if _e.get("perf_3y") else ""
                        _az_opts.append((_e["isin"],
                            f"{_e['nome'][:45]}  [{_e['classificazione'][:28]}  score:{_e['score_qualita']:.1f}{_p3s}]"))

                    _az_sel2 = st.multiselect(
                        "Seleziona fondi Azimut da aggiungere",
                        options=[o[0] for o in _az_opts],
                        format_func=lambda x: next((o[1] for o in _az_opts if o[0]==x), x),
                        key="adv_az_sel",
                    )
                    if st.button("➕ Aggiungi Azimut selezionati", key="adv_az_add"):
                        for _ia in _az_sel2:
                            _ea = next((e for e in _az_top50 if e["isin"]==_ia), None)
                            if _ea:
                                _pool_add(_ia, _ea)
                        n = _adv_add_isins(_az_sel2)
                        st.toast(f"✅ {n} fondi Azimut aggiunti")
                        st.rerun()
                else:
                    st.info("Catalogo Azimut non caricato. Carica il file nella sidebar.")
            except Exception as _e:
                st.warning(f"Lista Azimut: {_e}")

        # ── Tab 4: ISIN manuale + rimozione ──────────────────────────────
        with _adv_tab4:
            _adv_raw = st.text_area("ISIN o Ticker aggiuntivi (uno per riga)", height=80,
                                     placeholder="ENI.MI\nAAPL\nIE00B4L5Y983",
                                     key="fe_adv_raw")
            _adv_remove = st.multiselect(
                "Rimuovi dalla selezione corrente",
                options=st.session_state.get("fe_selected_isins", []),
                format_func=lambda x: f"{x} — {str(_all_fund_pool.get(x,{}).get('nome',x))[:45]}",
                key="fe_adv_remove",
            )
            _adv_c1, _adv_c2 = st.columns(2)
            if _adv_c1.button("Aggiungi ISIN", key="adv_add"):
                from utils.nav_fetcher import classify_asset_type
                for _line in (_adv_raw or "").strip().split("\n"):
                    _tok = _line.strip().upper()
                    if _tok:
                        _cur = st.session_state.get("fe_selected_isins", [])
                        if _tok not in _cur:
                            _cur.append(_tok)
                            st.session_state["fe_selected_isins"] = _cur
                        _pool_add(_tok, {"isin": _tok, "nome": _tok,
                                         "classificazione": classify_asset_type(_tok)})
                st.rerun()
            if _adv_c2.button("Rimuovi selezionati", key="adv_rem") and _adv_remove:
                st.session_state["fe_selected_isins"] = [
                    i for i in st.session_state.get("fe_selected_isins", [])
                    if i not in _adv_remove
                ]
                st.rerun()

    # ── View Black-Litterman manuali ─────────────────────────────────────
    with st.expander("🔮 View Black-Litterman (opzionale)", expanded=False):
        st.caption(
            "Inserisci le tue aspettative di rendimento per singoli strumenti. "
            "Lascia vuoto per usare le view automatiche (basate su Score Qualità). "
            "Le view si applicano come rendimento extra atteso rispetto al mercato."
        )
        _bl_sel = st.session_state.get("fe_selected_isins", [])
        _prev_bl_views = st.session_state.get("fe_bl_views_manual", {})
        if not _bl_sel:
            st.info("Esegui prima la selezione strumenti (premi 'Calcola portafoglio').")
        else:
            _bl_new_views = {}
            _bl_new_confs = {}
            _bl_cols = st.columns(2)
            for _bi, _bisin in enumerate(_bl_sel):
                _bname = str(_all_fund_pool.get(_bisin, {}).get("nome", _bisin))[:38]
                _bcol  = _bl_cols[_bi % 2]
                _prev_view = _prev_bl_views.get(_bisin, {}).get("view", 0.0)
                _prev_conf = _prev_bl_views.get(_bisin, {}).get("conf", 50)
                _bview = _bcol.number_input(
                    f"{_bname}",
                    min_value=-50.0, max_value=100.0, value=float(_prev_view), step=0.5,
                    format="%.1f", key=f"bl_view_{_bisin}",
                    help="Rendimento atteso annuo (%) — 0 = neutro / nessuna view"
                )
                _bconf = _bcol.slider(
                    "Confidenza %", 10, 100, int(_prev_conf), 10,
                    key=f"bl_conf_{_bisin}",
                )
                if abs(_bview) > 0.01:
                    _bl_new_views[_bisin] = _bview / 100
                    _bl_new_confs[_bisin] = _bconf / 100

            _bl_save_col, _bl_reset_col = st.columns(2)
            if _bl_save_col.button("💾 Salva view BL", key="bl_save"):
                st.session_state["fe_bl_views_manual"] = {
                    _bisin: {"view": _bl_new_views.get(_bisin, 0), "conf": int(_bl_new_confs.get(_bisin,0)*100)}
                    for _bisin in _bl_sel if _bisin in _bl_new_views
                }
                st.session_state["fe_bl_views_dict"]  = _bl_new_views
                st.session_state["fe_bl_confs_dict"]  = _bl_new_confs
                st.toast(f"✅ {len(_bl_new_views)} view salvate — ricalcola per applicarle")
            if _bl_reset_col.button("🗑️ Azzera view", key="bl_reset_views"):
                for _k2 in ["fe_bl_views_manual","fe_bl_views_dict","fe_bl_confs_dict"]:
                    st.session_state.pop(_k2, None)
                st.toast("View BL azzerate — si useranno le view automatiche")

    # Leggi views manuali per passarle all'optimizer
    if st.session_state.get("fe_bl_views_dict"):
        use_bl  = True
        bl_views = st.session_state["fe_bl_views_dict"]
        bl_conf  = st.session_state.get("fe_bl_confs_dict", {})

    # Reset
    if _reset_btn:
        for _k in ["fe_result","fe_price_dict","bl_result","bl_views_used",
                   "fe_selected_isins","fe_ac_map","fe_ac_target","_fe_form_vals",
                   "fe_bl_views_manual","fe_bl_views_dict","fe_bl_confs_dict"]:
            st.session_state.pop(_k, None)
        st.toast("Reset completato")
        st.rerun()

    # Variabili per il blocco optimizer (sotto)
    run = False
    _prev_vals = st.session_state.get("_fe_form_vals", {})
    min_w = _prev_vals.get("min_w_pct", 3) / 100
    max_w = _prev_vals.get("max_w_pct", 30) / 100
    sel_isins = st.session_state.get("fe_selected_isins", [])
    forced_include_sel: list = []
    use_bl = False
    bl_views: dict = {}
    bl_conf: dict = {}

    if _submitted and _total_pct == 100:
        st.session_state["_fe_form_vals"] = {
            "pct_az": int(_pct_az), "pct_ob": int(_pct_ob), "pct_mp": int(_pct_mp),
            "n_az_str": int(_n_az_str), "n_ob_str": int(_n_ob_str), "n_mp_str": int(_n_mp_str),
            "n_min_az": int(_n_min_az),
            "use_fondi": bool(_use_fondi), "use_etf": bool(_use_etf),
            "use_azioni": bool(_use_azioni),
            "min_w_pct": int(_min_w_pct), "max_w_pct": int(_max_w_pct),
            "periodo": str(_periodo),
        }
        min_w = int(_min_w_pct) / 100
        max_w = int(_max_w_pct) / 100
        st.session_state["opt_period"] = str(_periodo)

        with st.spinner("Selezione strumenti..."):
            _az_sel = _pick_assets("Azioni",        int(_n_az_str), _use_fondi, _use_etf) if int(_pct_az)>0 else []
            _ob_sel = _pick_assets("Obbligazioni",  int(_n_ob_str), _use_fondi, _use_etf) if int(_pct_ob)>0 else []
            _bi_sel = []   # bilanciato non più campo separato
            _mp_sel = _pick_assets("Materie Prime", int(_n_mp_str), False,       _use_etf) if (int(_pct_mp)>0 and int(_n_mp_str)>0) else []
            if _use_azioni:
                _it = [i for i in ITALIAN_STOCKS if not i.startswith("IT_BTP")][:4]
                _az_sel = list(dict.fromkeys(_az_sel + _it))
            _all_sel = list(dict.fromkeys(_az_sel + _ob_sel + _bi_sel + _mp_sel))
            _n_min_az_int = int(_n_min_az)
            _all_sel, _az_forced = _inject_azimut(
                _all_sel, _n_min_az_int, az_sel=_az_sel, ob_sel=_ob_sel
            )

        # Debug Azimut (solo se richiesto)
        if _n_min_az_int > 0:
            _az_in_sel = [i for i in _all_sel if _all_fund_pool.get(i, {}).get("_source") == "azimut"]
            _az_status_col, _ = st.columns([2, 1])
            if _az_in_sel:
                _az_status_col.success(
                    f"✅ {len(_az_in_sel)} fondo/i Azimut incluso/i: "
                    + ", ".join(str(_all_fund_pool.get(i,{}).get("nome",i))[:30] for i in _az_in_sel[:3])
                )
            else:
                _az_detail = (
                    f"df_azimut vuoto" if df_azimut.empty
                    else f"df_azimut: {len(df_azimut)} righe, ISIN col: "
                         f"{'ISIN' if 'ISIN' in df_azimut.columns else list(df_azimut.columns)[:4]}, "
                         f"forced={_az_forced}"
                )
                _az_status_col.warning(
                    f"⚠️ Nessun fondo Azimut trovato (richiesti {_n_min_az_int}). "
                    f"Dettaglio: {_az_detail}"
                )

        st.session_state["fe_selected_isins"] = _all_sel
        # Costruisce ac_map includendo i fondi Azimut iniettati
        # (non presenti in _az_sel/_ob_sel originali → li mappa dal loro _macro nel pool)
        _ac_map_base = (
            {i: "Azioni"        for i in _az_sel} |
            {i: "Obbligazioni"  for i in _ob_sel} |
            {i: "Bilanciato"    for i in _bi_sel} |
            {i: "Materie Prime" for i in _mp_sel}
        )
        _macro_to_ac = {"Azioni": "Azioni", "Obbligazioni": "Obbligazioni",
                        "Materie Prime": "Materie Prime", "Bilanciato": "Bilanciato"}
        for _isin_ac in _all_sel:
            if _isin_ac not in _ac_map_base:
                _pm = _all_fund_pool.get(_isin_ac, {}).get("_macro", "")
                _ac_map_base[_isin_ac] = _macro_to_ac.get(_pm, "Azioni")
        st.session_state["fe_ac_map"] = _ac_map_base
        st.session_state["fe_ac_target"] = {
            k: v/100 for k, v in [
                ("Azioni", int(_pct_az)), ("Obbligazioni", int(_pct_ob)),
                ("Materie Prime", int(_pct_mp)),
            ] if v > 0
        }
        sel_isins          = _all_sel
        forced_include_sel = _az_forced
        run                = True

        if sel_isins:
            _etf_set = {e["isin"] for e in _ETF_STATIC_LIST}
            _prev_rows = []
            for _isin in sel_isins:
                _info = _all_fund_pool.get(_isin, {})
                _fonte = ("Azimut" if _info.get("_source")=="azimut"
                          else "ETF" if _isin in _etf_set else "Terzi")
                # Per fondi Azimut iniettati (non in az_sel/ob_sel originali) usa il macro dal pool
                if _isin in set(_az_sel or []):
                    _macro_lbl = "Azioni"
                elif _isin in set(_ob_sel or []):
                    _macro_lbl = "Obbligazioni"
                elif _isin in set(_bi_sel or []):
                    _macro_lbl = "Bilanciato"
                elif _isin in set(_mp_sel or []):
                    _macro_lbl = "Mat. Prime"
                else:
                    # Azimut iniettato in sostituzione: recupera macro dal pool
                    _pool_macro = _all_fund_pool.get(_isin, {}).get("_macro", "")
                    _macro_lbl = _pool_macro if _pool_macro else "Mat. Prime"
                _prev_rows.append({
                    "Asset Class": _macro_lbl, "Fonte": _fonte, "ISIN": _isin,
                    "Nome": str(_info.get("nome", _isin))[:52],
                    "Score": round(float(_info.get("score_qualita", 0) or 0), 2),
                    "Perf 3Y %": _info.get("perf_3y"),
                    "FIDA": _info.get("rating_fida"),
                })
            st.dataframe(
                pd.DataFrame(_prev_rows), use_container_width=True, hide_index=True,
                column_config={
                    "Score":     st.column_config.ProgressColumn("Score", min_value=0, max_value=20, format="%.2f"),
                    "Perf 3Y %": st.column_config.NumberColumn(format="%.1f"),
                    "FIDA":      st.column_config.NumberColumn(format="%d"),
                    "Fonte":     st.column_config.TextColumn("Fonte", width="small"),
                },
                height=min(480, len(_prev_rows)*38+50),
            )
            _n_az_sel = sum(1 for r in _prev_rows if r["Fonte"]=="Azimut")
            if _n_az_sel:
                st.caption(f"{_n_az_sel} fondi Azimut inclusi")

    # ── (fine sezione input) ─────────────────────────────────────────────────

    if run:
        if len(sel_isins) < 3:
            st.error("Seleziona almeno 3 strumenti.")
        else:
            with st.spinner(f"Recupero dati storici per {len(sel_isins)} strumenti..."):
                period_map = {"1Y": "1y", "3Y": "3y", "5Y": "5y"}
                period = period_map.get(st.session_state["opt_period"], "3y")
                # Indice lookup perf: df_unified (fondi) + ETF universe (ETF)
                _du_idx = {}
                if not df_unified.empty and "isin" in df_unified.columns:
                    for _, _du_r in df_unified.iterrows():
                        _isin_r = str(_du_r.get("isin",""))
                        if _isin_r:
                            _du_idx[_isin_r] = _du_r.to_dict()
                # Aggiunge dati ETF universe (ticker + perf se disponibili)
                try:
                    _etf_cached = _load_etf_universe_cached()
                    for _, _er in _etf_cached.iterrows():
                        _ei = str(_er.get("isin",""))
                        if _ei and _ei not in _du_idx:
                            _du_idx[_ei] = _er.to_dict()
                        elif _ei:
                            # Aggiunge ticker se mancante
                            if not _du_idx[_ei].get("ticker") and _er.get("ticker"):
                                _du_idx[_ei]["ticker"] = _er.get("ticker")
                except Exception:
                    pass

                def _get_perf(isin: str, key: str):
                    """Cerca perf: pool → df_unified → ETF universe."""
                    for _src in [_all_fund_pool.get(isin, {}), _du_idx.get(isin, {})]:
                        v = _src.get(key)
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            try:
                                fv = float(v)
                                if fv != 0.0:   # 0.0 esplicito è spesso placeholder
                                    return fv
                            except Exception:
                                pass
                    return None

                asset_list = []
                for isin in sel_isins:
                    info = _all_fund_pool.get(isin, {})
                    asset_list.append({
                        "isin": isin,
                        "ticker": info.get("ticker") or _du_idx.get(isin, {}).get("ticker"),
                        "perf_1y":   _get_perf(isin, "perf_1y"),
                        "perf_3y":   _get_perf(isin, "perf_3y"),
                        "perf_ytd":  _get_perf(isin, "perf_ytd"),
                        "perf_2022": _get_perf(isin, "perf_2022"),
                        "perf_2023": _get_perf(isin, "perf_2023"),
                        "perf_2024": _get_perf(isin, "perf_2024"),
                    })
                price_dict = get_multiple_nav(asset_list, period=period)

                # Filtra serie piatte (var ≈ 0) prima dell'ottimizzazione
                _flat_isins = []
                _good_price_dict = {}
                for _p_isin, _p_ser in price_dict.items():
                    if isinstance(_p_ser, pd.Series) and len(_p_ser) >= 6:
                        _variance = float(_p_ser.pct_change().dropna().var())
                        if _variance < 1e-8:
                            _flat_isins.append(_p_isin)
                        else:
                            _good_price_dict[_p_isin] = _p_ser
                    else:
                        _good_price_dict[_p_isin] = _p_ser

                if _flat_isins:
                    price_dict = _good_price_dict
                    # Info discreta: solo expander collassato, non warning rosso
                    with st.expander(
                        f"ℹ️ {len(_flat_isins)} strument{'o' if len(_flat_isins)==1 else 'i'} "
                        f"senza dati storici (esclus{'o' if len(_flat_isins)==1 else 'i'})",
                        expanded=False
                    ):
                        _flat_names = [str(_all_fund_pool.get(i,{}).get("nome",i))[:40]
                                       for i in _flat_isins]
                        for _fn in _flat_names[:8]:
                            st.caption(f"• {_fn}")
                        st.caption(
                            "Causa: dati non disponibili su yfinance o Morningstar. "
                            "Usa 🗑️ Svuota cache se il problema persiste."
                        )

            # Conta asset viabili: con serie storica O con stats da Excel
            _n_with_series = len(price_dict)
            _n_with_stats  = sum(
                1 for i in sel_isins
                if i not in price_dict
                and _all_fund_pool.get(i, {}).get("perf_3y") is not None
            )
            _n_viable = _n_with_series + _n_with_stats

            if _n_viable < 3:
                st.error(
                    f"Dati insufficienti: {_n_with_series} serie storiche + {_n_with_stats} "
                    f"fondi da Excel = {_n_viable} strumenti viabili (min 3).\n\n"
                    "**Suggerimenti:**\n"
                    "- Carica il file Excel fondi terze parti (deve contenere perf_3y e volatilita)\n"
                    "- Abilita 'Includi ETF' per aggiungere strumenti con dati yfinance certi\n"
                    "- Clicca **🗑️ Svuota cache** nella sidebar"
                )
            else:
                with st.spinner("Ottimizzazione portafoglio..."):
                    rfr = st.session_state["risk_free_rate"] / 100
                    # Vincoli settoriali — usa tutti gli asset selezionati (non solo price_dict)
                    ac_map   = st.session_state.get("fe_ac_map", {})
                    ac_target= st.session_state.get("fe_ac_target", {})
                    sector_constraints = None
                    if ac_map and ac_target:
                        sc_mapper = {k: v for k, v in ac_map.items() if k in sel_isins}
                        if sc_mapper:
                            sector_constraints = {
                                "mapper": sc_mapper,
                                "lower": {k: max(0, v - 0.10) for k, v in ac_target.items()},
                                "upper": {k: min(1, v + 0.10) for k, v in ac_target.items()},
                            }

                    result = compute_efficient_frontier(
                        price_dict, weight_bounds=(min_w, max_w),
                        risk_free_rate=rfr,
                        forced_include=forced_include_sel or None,
                        sector_constraints=sector_constraints,
                        assets_info=_all_fund_pool,
                        selected_isins=sel_isins,
                    )
                if "error" in result:
                    st.error(f"Errore ottimizzazione: {result['error']}")
                else:
                    lbl = " (con vincoli asset class)" if sector_constraints else ""
                    st.success(f"Ottimizzazione completata su {_n_viable} strumenti{lbl}.")
                    st.session_state["fe_result"] = result
                    st.session_state["fe_price_dict"] = price_dict

                    # ── Black-Litterman sempre calcolato ──────────────────
                    # Views manuali se inserite, altrimenti auto da Score Qualità
                    with st.spinner("Black-Litterman (views auto da Score Qualità)..."):
                        _bl_views_used  = bl_views if (use_bl and bl_views) else {}
                        _bl_confs_used  = bl_conf  if (use_bl and bl_views) else {}
                        if not _bl_views_used:
                            # Auto-views da Score Qualità
                            _base_mu = result.get("mu", {})
                            _bl_views_used, _bl_confs_used = compute_bl_auto_views(
                                _all_fund_pool, base_returns=_base_mu
                            )
                        if _bl_views_used:
                            bl_r = compute_black_litterman(
                                price_dict, _bl_views_used, _bl_confs_used,
                                weight_bounds=(min_w, max_w),
                                risk_free_rate=rfr,
                                assets_info=_all_fund_pool,
                                forced_include=forced_include_sel or None,
                                sector_constraints=sector_constraints,
                                selected_isins=sel_isins,
                            )
                            st.session_state["bl_result"] = bl_r
                            st.session_state["bl_views_used"] = _bl_views_used

    # ── RISULTATI ──────────────────────────────────────────────────────────
    if "fe_result" in st.session_state:
        result   = st.session_state["fe_result"]
        price_dict = st.session_state.get("fe_price_dict", {})
        bl_result  = st.session_state.get("bl_result")

        # Mappa ISIN→nome per label leggibili (tutti gli asset selezionati, non solo price_dict)
        _all_result_isins = set(price_dict.keys()) | set(st.session_state.get("fe_selected_isins", []))
        isin_label = {isin: str(_all_fund_pool.get(isin, {}).get("nome", isin))[:35]
                      for isin in _all_result_isins}

        st.markdown("---")
        st.subheader("📊 Risultati Ottimizzazione")
        st.caption(
            "⚠️ **Nota**: Rendimento e Volatilità sono stime del modello basate sulle "
            "performance storiche a 3 anni dei fondi selezionati — non sono previsioni garantite. "
            "Il rendimento è il ritorno cumulativo atteso basato sui dati storici, non annualizzato."
        )

        # KPI
        m_col = st.columns(4)
        ms = result.get("max_sharpe", {})
        if ms and "error" not in ms:
            m_col[0].metric("📈 Rendimento stimato (3Y cumulativo)", f"{ms['ret']*100:.2f}%")
            m_col[1].metric("📉 Volatilità annua stimata", f"{ms['vol']*100:.2f}%")
            m_col[2].metric("⚡ Sharpe Ratio", f"{ms['sharpe']:.3f}")
            if price_dict:
                mdd = estimate_max_drawdown(ms["weights"], price_dict)
                m_col[3].metric("📉 Max Drawdown stimato", f"{mdd:.2f}%")

        # ── Controllo qualità dati ───────────────────────────────────────
        _ms_err  = ms.get("error", "") if ms else "nessun risultato"
        _ms_ret  = ms.get("ret", 0) if ms and "error" not in ms else 0
        _ms_vol  = ms.get("vol", 0) if ms and "error" not in ms else 0
        _ms_sh   = ms.get("sharpe", 0) if ms and "error" not in ms else 0

        # Degenere = dati piatti O errore ottimizzatore
        # Degenere solo se c'è errore esplicito o rendimento praticamente nullo (< 0.1%)
        _is_degenerate = ("error" in (ms or {})) or abs(_ms_ret) < 0.001

        if _is_degenerate:
            if _ms_err:
                _msg = (
                    f"🚫 **Errore ottimizzazione**: {_ms_err}\n\n"
                    "Possibili cause:\n"
                    "- Troppo pochi asset con dati validi (min 3)\n"
                    "- Matrice di covarianza singolare (asset troppo correlati o serie piatte)\n"
                    "- Vincoli impossibili (peso min × N asset > 100%)\n\n"
                    "**Soluzione:** riduci i vincoli di peso o aggiungi più strumenti diversificati."
                )
            else:
                _msg = (
                    f"🚫 **Dati non validi** — Rendimento {_ms_ret*100:.2f}% · Sharpe {_ms_sh:.2f}\n\n"
                    "Le serie storiche dei fondi sono piatte.\n"
                    "Clicca **Svuota cache** qui sotto, poi ricalcola."
                )
            st.error(_msg)
            _rc1, _rc2, _rc3 = st.columns(3)
            if _rc1.button("1️⃣ Svuota cache", type="primary", use_container_width=True):
                import os as _os
                from pathlib import Path as _P
                # Svuota cache Streamlit
                _load_all_data.clear()
                _load_preselection.clear()
                _load_etf_universe_cached.clear()
                # IMPORTANTE: cancella anche nav_cache.json su disco
                # (contiene serie sintetiche con valori decimali errati)
                for _cache_file in ["data/nav_cache.json", "data/etf_cache.json"]:
                    _cf = _P(_cache_file)
                    if _cf.exists():
                        try:
                            _os.remove(_cf)
                        except Exception:
                            pass
                # Pulisce risultati optimizer
                for _k in ["fe_result","fe_price_dict","bl_result","fe_selected_isins"]:
                    st.session_state.pop(_k, None)
                st.toast("✅ Cache completamente svuotata (incluso nav_cache.json)")
                st.rerun()
            _rc2.info("2️⃣ I file Excel **rimangono in memoria** — non serve ricaricarli")
            _rc3.info("3️⃣ Ri-seleziona i fondi e ricalcola")
            st.stop()  # Non mostrare il grafico degenere

        # ── Calcolo portafoglio manuale (da session_state) ───────────────
        _manual_weights_raw = st.session_state.get("fe_manual_weights", {})
        _mu_dict  = result.get("mu", {})
        _cov_dict = result.get("cov", {})
        _manual_point = None
        if _manual_weights_raw and _mu_dict and _cov_dict:
            try:
                import numpy as _np
                _man_assets = [i for i in _manual_weights_raw
                               if i in _mu_dict and _manual_weights_raw[i] > 0]
                if len(_man_assets) >= 2:
                    _w = _np.array([_manual_weights_raw[i]/100 for i in _man_assets])
                    _w = _w / _w.sum()  # normalizza
                    _mu_arr = _np.array([_mu_dict[i] for i in _man_assets])
                    _cov_df = pd.DataFrame(_cov_dict).reindex(index=_man_assets, columns=_man_assets).fillna(0)
                    _cov_mat = _cov_df.values
                    _man_ret = float(_w @ _mu_arr)
                    _man_vol = float(_np.sqrt(_w @ _cov_mat @ _w))
                    _man_sharpe = round((_man_ret - rfr) / _man_vol, 3) if _man_vol > 0 else 0
                    _manual_point = {
                        "ret": round(_man_ret * 100, 2),
                        "vol": round(_man_vol * 100, 2),
                        "sharpe": _man_sharpe,
                        "weights": {i: round(_w[j]*100,1) for j,i in enumerate(_man_assets)},
                    }
            except Exception:
                _manual_point = None

        # ── Grafico Frontiera Efficiente — stile Markowitz classico ─────
        fig_fe = go.Figure()
        mc       = result.get("monte_carlo", pd.DataFrame())
        frontier = result.get("frontier_df", pd.DataFrame())
        mv_data  = result.get("min_variance", {})

        # Calcolo range assi
        _all_vols, _all_rets = [], []
        if not mc.empty:
            _all_vols += (mc["vol"]*100).tolist()
            _all_rets += (mc["ret"]*100).tolist()
        if not frontier.empty:
            _all_vols += (frontier["vol"]*100).tolist()
            _all_rets += (frontier["ret"]*100).tolist()

        def _axis_range(vals, pad=0.18):
            if not vals: return [0, 1]
            lo, hi = min(vals), max(vals)
            span = max(hi - lo, 0.5)
            return [round(lo - span*pad, 2), round(hi + span*pad, 2)]

        x_range = _axis_range(_all_vols)
        y_range = _axis_range(_all_rets)

        # ── 1. Punti dei singoli asset (portafogli con 100% in 1 asset) ──
        _asset_points = []
        for _isin in price_dict:
            _s = price_dict[_isin]
            if isinstance(_s, pd.Series) and len(_s) >= 12:
                try:
                    _s2 = _s.copy(); _s2.index = pd.to_datetime(_s2.index)
                    _rets = _s2.pct_change().dropna()
                    _ar = float(_rets.mean() * 12 * 100)   # annualizzato %
                    _av = float(_rets.std() * (12**0.5) * 100)  # annualizzata %
                    _nm = str(_all_fund_pool.get(_isin, {}).get("nome", _isin))[:25]
                    _asset_points.append({"isin": _isin, "nome": _nm,
                                          "vol": round(_av,2), "ret": round(_ar,2)})
                except Exception:
                    pass

        if _asset_points:
            _ap_df = pd.DataFrame(_asset_points)
            fig_fe.add_trace(go.Scatter(
                x=_ap_df["vol"], y=_ap_df["ret"],
                mode="markers+text",
                marker=dict(color="#555", size=9, symbol="circle",
                            line=dict(color="white", width=1)),
                text=_ap_df["nome"],
                textposition="top right",
                textfont=dict(size=9, color="#444"),
                name="Singoli asset",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Vol: <b>%{x:.1f}%</b>  Rend: <b>%{y:.1f}%</b>"
                    "<extra></extra>"
                ),
            ))

        # ── 2. Nuvola Monte Carlo — punti grigi piccoli ──────────────────
        if not mc.empty:
            fig_fe.add_trace(go.Scatter(
                x=(mc["vol"]*100).round(2),
                y=(mc["ret"]*100).round(2),
                mode="markers",
                marker=dict(color="rgba(150,160,180,0.35)", size=4),
                name="Portafogli ammissibili",
                hovertemplate="Vol: %{x:.1f}%  Rend: %{y:.1f}%<extra></extra>",
            ))

        # ── 3+4. Frontiera: tratto efficiente (blu) + inefficiente (grigio tratteggiato) ─
        # La frontiera viene DIVISA al punto Min Varianza:
        # - sopra (ret >= mv_ret) → efficiente, linea blu spessa
        # - sotto (ret < mv_ret)  → inefficiente, linea grigia tratteggiata
        _annotations = []
        _fr_ineff_point = None   # punto medio tratto inefficiente per annotation
        if not frontier.empty:
            _fr_all = frontier.sort_values("ret")
            _mv_ret_val = mv_data.get("ret", 0) if (mv_data and "error" not in mv_data) else 0

            _fr_eff   = _fr_all[_fr_all["ret"] >= _mv_ret_val]
            _fr_ineff = _fr_all[_fr_all["ret"] <  _mv_ret_val]

            # Tratto efficiente — blu pieno
            if not _fr_eff.empty:
                fig_fe.add_trace(go.Scatter(
                    x=(_fr_eff["vol"]*100).round(2),
                    y=(_fr_eff["ret"]*100).round(2),
                    mode="lines",
                    line=dict(color=NAVY, width=4),
                    name="Tratto efficiente",
                    hovertemplate="✅ Efficiente | Vol: <b>%{x:.1f}%</b>  Rend: <b>%{y:.1f}%</b><extra></extra>",
                ))
                # Punto 75° percentile per annotation (in zona alta della linea)
                _fr_q75 = _fr_eff.iloc[int(len(_fr_eff) * 0.75)]
                _eff_ann_v = round(float(_fr_q75["vol"]) * 100, 2)
                _eff_ann_r = round(float(_fr_q75["ret"]) * 100, 2)

            # Tratto inefficiente — grigio tratteggiato
            if not _fr_ineff.empty:
                fig_fe.add_trace(go.Scatter(
                    x=(_fr_ineff["vol"]*100).round(2),
                    y=(_fr_ineff["ret"]*100).round(2),
                    mode="lines",
                    line=dict(color="#94A3B8", width=3, dash="dash"),
                    name="Tratto inefficiente",
                    hovertemplate="⚠️ Inefficiente | Vol: %{x:.1f}%  Rend: %{y:.1f}%<extra></extra>",
                ))
                _fr_ineff_mid = _fr_ineff.iloc[len(_fr_ineff)//2]
                _fr_ineff_point = (round(float(_fr_ineff_mid["vol"])*100,2),
                                   round(float(_fr_ineff_mid["ret"])*100,2))

        # ── 5. Punto Min Varianza ─────────────────────────────────────────
        if mv_data and "error" not in mv_data and mv_data.get("vol"):
            _mv_v = round(mv_data["vol"]*100, 2)
            _mv_r = round(mv_data["ret"]*100, 2)
            # Linea tratteggiata orizzontale dal punto Min Var a sinistra
            fig_fe.add_shape(
                type="line",
                x0=x_range[0], x1=_mv_v,
                y0=_mv_r, y1=_mv_r,
                line=dict(color="#64748B", width=1.5, dash="dash"),
            )
            fig_fe.add_trace(go.Scatter(
                x=[_mv_v], y=[_mv_r],
                mode="markers",
                marker=dict(color="#1D4ED8", size=14, symbol="diamond",
                            line=dict(color="white", width=2)),
                name=f"◆ Min Varianza — rischio minimo (Vol {_mv_v:.1f}%  Rend {_mv_r:.1f}%)",
                hovertemplate=(
                    "<b>Minima Varianza</b><br>"
                    "Rischio minimo raggiungibile<br>"
                    "Vol: <b>%{x:.2f}%</b><br>"
                    "Rend: <b>%{y:.2f}%</b>"
                    "<extra></extra>"
                ),
            ))

            # ── Annotazioni: testo a DESTRA con freccia lunga verso la curva ─
            if not frontier.empty and _fr_eff is not None and not _fr_eff.empty:
                # "Tratto EFFICIENTE" → testo sulla destra del grafico, freccia punta alla linea
                _annotations.append(dict(
                    x=_eff_ann_v, y=_eff_ann_r,        # punta freccia = punto sulla frontiera
                    ax=180, ay=-10,                      # testo 180px a destra
                    xref="x", yref="y", axref="pixel", ayref="pixel",
                    text="<b style='color:#1A2C54'>▲ Tratto EFFICIENTE</b><br>"
                         "<span style='font-size:9px;color:#444'>"
                         "Massimo rendimento per ogni dato rischio</span>",
                    showarrow=True,
                    arrowhead=2, arrowwidth=1.5, arrowcolor=NAVY,
                    font=dict(size=10), bgcolor="rgba(240,245,255,0.95)",
                    bordercolor=NAVY, borderwidth=1.5, borderpad=5,
                    align="left",
                ))

            # "Tratto INEFFICIENTE" → testo sulla destra, freccia punta al tratto grigio
            if _fr_ineff_point:
                _annotations.append(dict(
                    x=_fr_ineff_point[0], y=_fr_ineff_point[1],
                    ax=180, ay=20,
                    xref="x", yref="y", axref="pixel", ayref="pixel",
                    text="<i style='color:#64748B'>▼ Tratto INEFFICIENTE</i><br>"
                         "<span style='font-size:9px;color:#888'>"
                         "Stesso rischio, rendimento più basso<br>"
                         "→ nessun investitore razionale lo sceglie</span>",
                    showarrow=True,
                    arrowhead=2, arrowwidth=1.2, arrowcolor="#94A3B8",
                    font=dict(size=9), bgcolor="rgba(248,248,252,0.95)",
                    bordercolor="#94A3B8", borderwidth=1, borderpad=4,
                    align="left",
                ))

        # ── 6. Max Sharpe ─────────────────────────────────────────────────
        if ms and "error" not in ms and ms.get("vol"):
            _ms_v = round(ms["vol"]*100, 2)
            _ms_r = round(ms["ret"]*100, 2)
            _ms_s = round(ms.get("sharpe",0), 2)
            fig_fe.add_trace(go.Scatter(
                x=[_ms_v], y=[_ms_r],
                mode="markers",
                marker=dict(color="#DC2626", size=18, symbol="star",
                            line=dict(color="white", width=2)),
                name=f"Max Sharpe  (Vol {_ms_v:.1f}%  Rend {_ms_r:.1f}%  Sharpe {_ms_s})",
                hovertemplate=(
                    "<b>Max Sharpe</b><br>"
                    "Vol: <b>%{x:.2f}%</b><br>"
                    "Rend: <b>%{y:.2f}%</b><br>"
                    f"Sharpe: <b>{_ms_s}</b>"
                    "<extra></extra>"
                ),
            ))
            _annotations.append(dict(
                x=_ms_v, y=_ms_r,
                ax=55, ay=-45,
                xref="x", yref="y", axref="pixel", ayref="pixel",
                text=f"<b style='color:#DC2626'>Max Sharpe</b><br>"
                     f"Rend {_ms_r:.1f}% · Vol {_ms_v:.1f}%<br>"
                     f"Sharpe {_ms_s}",
                showarrow=True,
                arrowhead=2, arrowwidth=1.5, arrowcolor="#DC2626",
                font=dict(size=10), bgcolor="white",
                bordercolor="#DC2626", borderwidth=1.5, borderpad=4,
            ))

        # ── 7. Black-Litterman (se presente) ────────────────────────────
        if bl_result and "error" not in bl_result and bl_result.get("vol"):
            _bl_v = round(bl_result["vol"]*100, 2)
            _bl_r = round(bl_result["ret"]*100, 2)
            fig_fe.add_trace(go.Scatter(
                x=[_bl_v], y=[_bl_r],
                mode="markers",
                marker=dict(color="#16A34A", size=15, symbol="pentagon",
                            line=dict(color="white", width=2)),
                name=f"Black-Litterman  (Vol {_bl_v:.1f}%  Rend {_bl_r:.1f}%)",
                hovertemplate=(
                    "<b>Black-Litterman</b><br>"
                    "Vol: <b>%{x:.2f}%</b>  Rend: <b>%{y:.2f}%</b>"
                    "<extra></extra>"
                ),
            ))

        # ── 8. Il tuo portafoglio manuale ────────────────────────────────
        if _manual_point:
            _mp_v = _manual_point["vol"]
            _mp_r = _manual_point["ret"]
            _mp_s = _manual_point["sharpe"]
            fig_fe.add_trace(go.Scatter(
                x=[_mp_v], y=[_mp_r],
                mode="markers",
                marker=dict(color="#F97316", size=18, symbol="circle",
                            line=dict(color="white", width=2.5)),
                name=f"🟠 Il tuo portafoglio (Vol {_mp_v:.1f}%  Rend {_mp_r:.1f}%  Sharpe {_mp_s})",
                hovertemplate=(
                    "<b>Il tuo portafoglio</b><br>"
                    "Vol: <b>%{x:.2f}%</b><br>"
                    "Rend: <b>%{y:.2f}%</b><br>"
                    f"Sharpe: <b>{_mp_s}</b>"
                    "<extra></extra>"
                ),
            ))
            _annotations.append(dict(
                x=_mp_v, y=_mp_r,
                ax=-65, ay=50,
                xref="x", yref="y", axref="pixel", ayref="pixel",
                text=f"<b style='color:#F97316'>Il tuo portafoglio</b><br>"
                     f"Rend {_mp_r:.1f}% · Vol {_mp_v:.1f}%<br>"
                     f"Sharpe {_mp_s}",
                showarrow=True,
                arrowhead=2, arrowwidth=1.5, arrowcolor="#F97316",
                font=dict(size=10), bgcolor="white",
                bordercolor="#F97316", borderwidth=1.5, borderpad=4,
            ))

        # ── Layout ────────────────────────────────────────────────────────
        fig_fe.update_layout(
            title=dict(
                text="<b>Frontiera Efficiente</b>  "
                     "<span style='font-size:12px;color:#666'>E(Rp) vs σ</span>",
                font=dict(size=16, color=NAVY), x=0.03,
            ),
            xaxis=dict(
                title=dict(text="Rischio — Volatilità annua σ (%)", font=dict(size=12)),
                tickformat=".1f", ticksuffix="%",
                gridcolor="#EEF1F6", showgrid=True, zeroline=False,
                range=x_range, tickfont=dict(size=11),
            ),
            yaxis=dict(
                title=dict(text="Rendimento atteso E(Rp) (%)", font=dict(size=12)),
                tickformat=".1f", ticksuffix="%",
                gridcolor="#EEF1F6", showgrid=True, zeroline=False,
                range=y_range, tickfont=dict(size=11),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=600,
            annotations=_annotations,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.22,
                xanchor="center", x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#DDD", borderwidth=1,
                font=dict(size=10),
            ),
            margin=dict(t=60, b=120, l=65, r=30),
            hoverlabel=dict(bgcolor="white", font=dict(size=11)),
        )
        st.plotly_chart(fig_fe, use_container_width=True, config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "toImageButtonOptions": {"format": "png", "width": 1200, "height": 700},
        })

        # Legenda esplicativa
        with st.expander("📖 Come leggere il grafico", expanded=False):
            st.markdown("""
**Asse X — Rischio (Volatilità σ):** quanto oscilla il portafoglio. Più è a destra, più è rischioso.

**Asse Y — Rendimento atteso E(Rp):** il rendimento annualizzato stimato. Più è in alto, meglio.

---

**🔵 Tratto EFFICIENTE** *(linea blu continua, sopra il diamante)*
> Per ogni livello di rischio scelto, offre il **massimo rendimento possibile**.
> L'ottimizzatore punta sempre su questa zona. Il punto ★ Max Sharpe è il rapporto rendimento/rischio migliore.

**⚫ Tratto INEFFICIENTE** *(linea tratteggiata, sotto il diamante)*
> Portafogli con lo **stesso rischio ma rendimento inferiore** a quelli efficienti.
> Un investitore razionale non li sceglie mai: c'è sempre un'alternativa migliore sul tratto superiore.

**◆ Min Varianza** *(diamante blu)*
> Il portafoglio col **rischio più basso in assoluto** tra tutti i possibili.
> Separa la zona efficiente (sopra) da quella inefficiente (sotto).

**⭐ Max Sharpe** *(stella rossa)*
> Il portafoglio col miglior **rapporto rendimento/rischio** (Indice di Sharpe).
> È il punto selezionato dall'app come portafoglio principale.

**Nuvola di punti grigi:** migliaia di portafogli casuali (simulazione Monte Carlo).
Mostrano lo spazio di tutte le combinazioni possibili degli strumenti selezionati.
""")

        # ── Sezione "Il tuo portafoglio" ─────────────────────────────────
        st.markdown("---")
        with st.expander("🟠 Costruisci il tuo portafoglio e confrontalo con la frontiera", expanded=False):
            _assets_for_manual = result.get("assets", sel_isins or [])
            if not _assets_for_manual:
                st.info("Esegui prima il calcolo della frontiera.")
            else:
                st.markdown(
                    "Imposta i pesi desiderati per ogni strumento. "
                    "Il punto 🟠 apparirà sul grafico (ricarica dopo aver salvato i pesi)."
                )
                # Form con sliders
                with st.form("manual_portfolio_form"):
                    _n_man = len(_assets_for_manual)
                    _default_w = 100.0 / _n_man
                    _prev_manual = st.session_state.get("fe_manual_weights", {})
                    _new_weights = {}
                    # Mostra slider in 2 colonne
                    _mc1, _mc2 = st.columns(2)
                    for _idx, _isin in enumerate(_assets_for_manual):
                        _nome_m = str(_all_fund_pool.get(_isin, {}).get("nome", _isin))[:40]
                        _prev_w = _prev_manual.get(_isin, round(_default_w, 1))
                        _col = _mc1 if _idx % 2 == 0 else _mc2
                        _new_weights[_isin] = _col.slider(
                            f"{_nome_m}",
                            min_value=0.0, max_value=100.0,
                            value=float(_prev_w), step=0.5,
                            key=f"man_w_{_isin}",
                        )
                    _tot_manual = sum(_new_weights.values())
                    st.caption(f"Totale pesi: **{_tot_manual:.1f}%** {'✅' if abs(_tot_manual-100)<0.5 else '⚠️ normalizzati automaticamente a 100%'}")
                    _save_manual = st.form_submit_button("💾 Salva pesi e aggiorna grafico", type="primary")
                    if _save_manual:
                        st.session_state["fe_manual_weights"] = _new_weights
                        st.rerun()

                # Tabella di confronto (se punto manuale disponibile)
                if _manual_point:
                    st.subheader("📊 Confronto: Il tuo portafoglio vs Ottimizzati")
                    _ms_cmp = {"Rendimento": f"{ms['ret']*100:.2f}%" if ms and 'ret' in ms else "-",
                               "Volatilità": f"{ms['vol']*100:.2f}%" if ms and 'vol' in ms else "-",
                               "Sharpe": f"{ms.get('sharpe',0):.3f}" if ms else "-"}
                    _mv_cmp_d = result.get("min_variance", {})
                    _mv_cmp = {"Rendimento": f"{_mv_cmp_d['ret']*100:.2f}%" if _mv_cmp_d and 'ret' in _mv_cmp_d else "-",
                               "Volatilità": f"{_mv_cmp_d['vol']*100:.2f}%" if _mv_cmp_d and 'vol' in _mv_cmp_d else "-",
                               "Sharpe": f"{_mv_cmp_d.get('sharpe',0):.3f}" if _mv_cmp_d else "-"}
                    _cmp_df = pd.DataFrame({
                        "🟠 Il tuo portafoglio": {
                            "Rendimento atteso": f"{_manual_point['ret']:.2f}%",
                            "Volatilità (rischio)": f"{_manual_point['vol']:.2f}%",
                            "Indice di Sharpe": str(_manual_point["sharpe"]),
                        },
                        "⭐ Max Sharpe (ottimizzato)": {
                            "Rendimento atteso": _ms_cmp["Rendimento"],
                            "Volatilità (rischio)": _ms_cmp["Volatilità"],
                            "Indice di Sharpe": _ms_cmp["Sharpe"],
                        },
                        "◆ Min Varianza (ottimizzato)": {
                            "Rendimento atteso": _mv_cmp["Rendimento"],
                            "Volatilità (rischio)": _mv_cmp["Volatilità"],
                            "Indice di Sharpe": _mv_cmp["Sharpe"],
                        },
                    })
                    st.dataframe(_cmp_df, use_container_width=True)

                    # Pesi del portafoglio manuale
                    st.markdown("**Composizione del tuo portafoglio (normalizzata):**")
                    _man_w_rows = []
                    for _isin, _w_pct in sorted(_manual_point["weights"].items(), key=lambda x: -x[1]):
                        _info_m = _all_fund_pool.get(_isin, {})
                        _man_w_rows.append({
                            "ISIN": _isin,
                            "Nome": str(_info_m.get("nome", _isin))[:50],
                            "Peso %": _w_pct,
                        })
                    st.dataframe(pd.DataFrame(_man_w_rows), use_container_width=True, hide_index=True)

                    _btn_reset = st.button("🗑️ Rimuovi il tuo portafoglio dal grafico")
                    if _btn_reset:
                        st.session_state.pop("fe_manual_weights", None)
                        st.rerun()

        # Pesi + correlazioni affiancati
        st.subheader("📋 Composizione portafogli")
        tab_ms2, tab_mv2, tab_bl2 = st.tabs(["⭐ Max Sharpe", "🛡️ Min Varianza", "🔮 Black-Litterman"])

        def _render_weights(pdata: dict, label: str, chart_key: str = ""):
            if not pdata or "error" in pdata:
                st.info(f"Portafoglio {label} non disponibile.")
                return
            w = {k: v for k, v in pdata["weights"].items() if v > 0.001}

            # Helper: cerca info fondo in pool → df_unified → df_azimut (fallback)
            def _get_info(isin):
                info = dict(_all_fund_pool.get(isin, {}))
                nome = info.get("nome", "")
                # Fallback su df_unified se nome mancante o uguale all'ISIN
                if (not nome or nome == isin) and not df_unified.empty and "isin" in df_unified.columns:
                    _row = df_unified[df_unified["isin"] == isin]
                    if not _row.empty:
                        r = _row.iloc[0]
                        info.update({k: r[k] for k in r.index if pd.notna(r[k])})
                # Fallback su df_azimut
                nome = info.get("nome", "")
                if (not nome or nome == isin) and not df_azimut.empty:
                    from utils.data_loader import AZIMUT_COLS as _AC
                    _c_isin2 = _AC.get("isin","ISIN")
                    _c_nome2 = _AC.get("nome","FONDO AZIMUT")
                    if _c_isin2 in df_azimut.columns:
                        _az_row = df_azimut[df_azimut[_c_isin2].astype(str).str.strip() == isin]
                        if not _az_row.empty:
                            info["nome"] = str(_az_row.iloc[0].get(_c_nome2, isin))
                            info["classificazione"] = str(_az_row.iloc[0].get(_AC.get("classificazione",""), ""))
                return info

            w_rows = []
            for isin, peso in sorted(w.items(), key=lambda x: -x[1]):
                info = _get_info(isin)
                nome = str(info.get("nome", isin) or isin)
                if nome == isin:
                    nome = isin  # mostra ISIN solo se proprio non trovato
                w_rows.append({
                    "ISIN": isin,
                    "Nome": nome[:55],
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
                                 "Perf 3Y %": st.column_config.NumberColumn(
                                     "Perf 3Y %", format="%.1f",
                                     help="Rendimento cumulativo 3 anni (fonte: file Excel)"),
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
                st.plotly_chart(pie, use_container_width=True, key=f"pie_{chart_key}")

            st.markdown(
                f"**Rendimento atteso:** {pdata.get('ret',0)*100:.2f}% &nbsp;|&nbsp; "
                f"**Volatilità:** {pdata.get('vol',0)*100:.2f}% &nbsp;|&nbsp; "
                f"**Sharpe:** {pdata.get('sharpe',0):.3f}"
            )

        with tab_ms2: _render_weights(ms, "Max Sharpe", "ms")
        with tab_mv2: _render_weights(result.get("min_variance",{}), "Min Varianza", "mv")
        with tab_bl2: _render_weights(bl_result or {}, "Black-Litterman", "bl")

        # ── Matrice Correlazioni ─────────────────────────────────────────
        if len(price_dict) >= 2:
            st.markdown("---")
            st.subheader("🔗 Matrice Correlazioni")
            try:
                # Ricostruisce DataFrame con indice datetime corretto
                # (le Series in session_state possono perdere il dtype dell'indice)
                _series_list = {}
                for _k, _s in price_dict.items():
                    if isinstance(_s, pd.Series):
                        _s2 = _s.copy()
                        _s2.index = pd.to_datetime(_s2.index)
                        _series_list[_k] = _s2
                    else:
                        # fallback se serializzato come dict
                        _s2 = pd.Series(_s)
                        _s2.index = pd.to_datetime(_s2.index)
                        _series_list[_k] = _s2

                prices_df = (
                    pd.DataFrame(_series_list)
                    .sort_index()
                    .ffill()
                    .dropna(how="all")
                )
                returns_df = prices_df.pct_change().dropna()

                if returns_df.empty or len(returns_df) < 3:
                    st.warning(
                        f"Serie troppo corte per calcolare correlazioni "
                        f"({len(returns_df)} periodi disponibili dopo allineamento). "
                        "Prova con un periodo più lungo nelle impostazioni."
                    )
                elif len(returns_df.columns) < 2:
                    st.info("Servono almeno 2 strumenti con dati sovrapposti.")
                else:
                    # Rinomina colonne con nomi leggibili
                    returns_df.columns = [
                        isin_label.get(c, c)[:28] for c in returns_df.columns
                    ]
                    corr = returns_df.corr().round(2)
                    n_assets = len(corr)
                    cell_h = max(30, min(55, 420 // n_assets))
                    fig_corr = px.imshow(
                        corr,
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        text_auto=True,
                        aspect="auto",
                        title=None,
                    )
                    fig_corr.update_traces(
                        textfont=dict(size=max(9, min(13, 140 // n_assets))),
                        hovertemplate="%{x}<br>%{y}<br>Correlazione: %{z:.2f}<extra></extra>",
                    )
                    fig_corr.update_layout(
                        height=max(300, n_assets * cell_h + 80),
                        margin=dict(l=0, r=10, t=10, b=0),
                        coloraxis_colorbar=dict(
                            title="Corr.", thickness=12, len=0.8,
                            tickvals=[-1, -0.5, 0, 0.5, 1],
                        ),
                        xaxis=dict(tickfont=dict(size=10)),
                        yaxis=dict(tickfont=dict(size=10)),
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

                    # Legenda interpretativa
                    lc1, lc2, lc3 = st.columns(3)
                    lc1.markdown("🔴 **vicino a +1** — si muovono insieme (bassa diversificazione)")
                    lc2.markdown("⚪ **vicino a 0** — movimenti indipendenti")
                    lc3.markdown("🔵 **vicino a -1** — movimenti opposti (ottima diversificazione)")

                    # Tabella coppie con correlazione più alta/bassa
                    _pairs = []
                    cols_c = list(corr.columns)
                    for i in range(len(cols_c)):
                        for j in range(i+1, len(cols_c)):
                            _pairs.append({
                                "Asset A": cols_c[i], "Asset B": cols_c[j],
                                "Correlazione": corr.iloc[i, j],
                            })
                    if _pairs:
                        _pairs_df = pd.DataFrame(_pairs).sort_values("Correlazione")
                        with st.expander("📊 Coppie più e meno correlate"):
                            _low = _pairs_df.head(3)
                            _high = _pairs_df.tail(3).iloc[::-1]
                            c_low, c_high = st.columns(2)
                            c_low.markdown("**Meno correlate (migliore diversificazione):**")
                            c_low.dataframe(_low, use_container_width=True, hide_index=True,
                                            column_config={"Correlazione": st.column_config.NumberColumn(format="%.2f")})
                            c_high.markdown("**Più correlate (simili):**")
                            c_high.dataframe(_high, use_container_width=True, hide_index=True,
                                             column_config={"Correlazione": st.column_config.NumberColumn(format="%.2f")})

            except Exception as _corr_err:
                st.error(f"Errore nel calcolo delle correlazioni: {_corr_err}")

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

        # ── EXPORT ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📥 Esporta")

        _mv_res  = result.get("min_variance", {})
        _bl_res  = bl_result or {}
        _bl_views_used = st.session_state.get("bl_views_used", {})
        _today   = datetime.now().strftime("%Y%m%d")

        def _make_metrics(pdata: dict, label: str) -> dict:
            if not pdata or "error" in pdata:
                return {}
            return {
                "Portafoglio": label,
                "Rendimento atteso (%)": f"{pdata.get('ret',0)*100:.2f}",
                "Volatilità (%)":        f"{pdata.get('vol',0)*100:.2f}",
                "Sharpe Ratio":          f"{pdata.get('sharpe',0):.3f}",
                "Generato":              datetime.now().strftime("%d/%m/%Y %H:%M"),
            }

        ms_metrics  = _make_metrics(ms, "Max Sharpe")
        mv_metrics  = _make_metrics(_mv_res, "Min Volatilità")
        bl_metrics  = _make_metrics(_bl_res, "Black-Litterman")

        # Converti grafico in PNG
        _fe_png = None
        try:
            from utils.exporter import plotly_to_png
            _fe_png = plotly_to_png(fig_fe)
        except Exception:
            pass

        # ── Riga 1: PDF completo + Excel standard ─────────────────────────
        ex1, ex2 = st.columns(2)

        if ms and "error" not in ms:
            pdf_bytes = export_portfolio_pdf(
                weights=ms["weights"],
                metrics=ms_metrics,
                title="Report Portafogli Efficienti",
                chart_bytes=_fe_png,
                fund_df=df_unified if not df_unified.empty else None,
                fund_pool=_all_fund_pool,
                weights_minvol=_mv_res.get("weights") if "error" not in _mv_res else None,
                metrics_minvol=mv_metrics or None,
                weights_bl=_bl_res.get("weights") if "error" not in _bl_res else None,
                metrics_bl=bl_metrics or None,
                bl_views=_bl_views_used or None,
            )
            if pdf_bytes:
                ex1.download_button(
                    "📄 PDF completo (3 portafogli)",
                    data=pdf_bytes,
                    file_name=f"report_portafogli_{_today}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            excel_bytes = export_portfolio_excel(
                ms["weights"], ms_metrics,
                fund_df=df_unified if not df_unified.empty else None,
                price_dict=price_dict, title="Max Sharpe",
            )
            ex2.download_button(
                "📊 Excel (Max Sharpe)",
                data=excel_bytes,
                file_name=f"portafoglio_max_sharpe_{_today}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        # ── Riga 2: AdvisorElite ──────────────────────────────────────────
        st.markdown("**AdvisorElite**")
        ae1, ae2, ae3, ae4 = st.columns(4)

        def _ae_buttons(col, label: str, pdata: dict, suffix: str):
            if not pdata or "error" in pdata:
                col.info(f"{label}\nnon disponibile")
                return
            w = {k: v for k, v in pdata["weights"].items() if v > 0.0001}
            csv_b = export_advisorelite_csv(w)
            col.download_button(
                f"📋 CSV\n{label}",
                data=csv_b,
                file_name=f"advisorelite_{suffix}_{_today}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        def _ae_excel_buttons(col, label: str, pdata: dict, suffix: str):
            if not pdata or "error" in pdata:
                return
            w = {k: v for k, v in pdata["weights"].items() if v > 0.0001}
            xls_b = export_advisorelite_excel(
                w, portfolio_name=label, fund_pool=_all_fund_pool
            )
            if xls_b:
                col.download_button(
                    f"📗 Excel\n{label}",
                    data=xls_b,
                    file_name=f"advisorelite_{suffix}_{_today}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

        _ae_buttons(ae1,      "Max Sharpe",        ms,      "max_sharpe")
        _ae_buttons(ae2,      "Min Volatilità",     _mv_res, "min_vol")
        _ae_buttons(ae3,      "Black-Litterman",    _bl_res, "bl")
        _ae_excel_buttons(ae4, "Max Sharpe",        ms,      "max_sharpe")

        st.caption(
            "📌 Il file CSV AdvisorElite contiene ISIN e % peso (somma = 100). "
            "Il file Excel segue il formato virtual-positions-template."
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
        # Fondi Azimut — campo sempre editabile, cerca nel catalogo completo
        _n_az_pq_avail = 0
        try:
            if not df_azimut.empty:
                _n_az_pq_avail = len(df_azimut)
            elif not df_unified.empty and "_source" in df_unified.columns:
                _n_az_pq_avail = int((df_unified["_source"] == "azimut").sum())
        except Exception:
            pass
        n_min_az_pq = st.number_input(
            "Min fondi Azimut",
            min_value=0, max_value=50,
            value=0, step=1, key="pq_n_min_azimut",
            help=(
                f"Garantisce almeno N fondi Azimut nel portafoglio finale "
                f"({_n_az_pq_avail} nel catalogo). "
                "Ricerca nel catalogo completo Azimut per Score Qualità, "
                "distribuiti nei bucket più adatti per classificazione."
            ),
            disabled=False,
        )
        if _n_az_pq_avail > 0:
            st.caption(f"📘 {_n_az_pq_avail} fondi Azimut nel catalogo")
        else:
            st.caption("⚠️ Carica il file Azimut dalla sidebar")

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
        with st.expander("🔍 Distribuzione fondi per bucket", expanded=False):
            st.dataframe(bucket_dist.reset_index().rename(
                columns={"_bucket_preview": "Bucket", "count": "N. fondi"}),
                use_container_width=True, hide_index=True)
            st.caption("Se un bucket è vuoto: le classificazioni inferite non matchano le keyword. Apri per diagnosticare.")

    # ── STATO LOCK / SOSTITUZIONI ───────────────────────────────────────────
    if "pq_locks" not in st.session_state:
        st.session_state["pq_locks"] = set()
    if "pq_replacements" not in st.session_state:
        st.session_state["pq_replacements"] = {}   # {(bucket, isin_old): isin_new}

    # ── COSTRUZIONE PORTAFOGLIO ─────────────────────────────────────────────
    with st.spinner("Costruzione portafoglio per Score Qualità..."):
        portfolio_buckets = build_portfolio_quality(
            df_unified, profilo=profilo, fondi_per_bucket=fondi_per_bucket
        )

    # ── INIEZIONE FONDI AZIMUT ──────────────────────────────────────────────
    if n_min_az_pq > 0:
        # Costruisce lista Azimut ordinata per score dal catalogo completo
        _pq_az_ranked = []
        try:
            from utils.scoring import compute_scores_df as _pq_css
            from utils.data_loader import build_unified_fund_df as _pq_buf
            _az_src_pq = pd.DataFrame()
            if not df_azimut.empty:
                _az_src_pq = _pq_buf(pd.DataFrame(), df_azimut)
            elif not df_unified.empty and "_source" in df_unified.columns:
                _az_src_pq = df_unified[df_unified["_source"] == "azimut"].copy()
            if not _az_src_pq.empty:
                _az_src_pq = _pq_css(_az_src_pq)
                _az_src_pq = _az_src_pq.sort_values("score_qualita", ascending=False)
                _pq_az_ranked = _az_src_pq["isin"].dropna().tolist()
        except Exception:
            pass

        # Conta quanti fondi Azimut sono già presenti nel portafoglio
        _az_in_pq = set()
        for _bk_df in portfolio_buckets.values():
            if _bk_df is not None and not _bk_df.empty and "isin" in _bk_df.columns:
                for _isin_chk in _bk_df["isin"].tolist():
                    if _isin_chk in set(_pq_az_ranked):
                        _az_in_pq.add(_isin_chk)

        _az_need = int(n_min_az_pq) - len(_az_in_pq)

        # Bucket disponibili nel portafoglio corrente
        _available_buckets = [k for k, v in portfolio_buckets.items()
                               if v is not None and not v.empty]

        if _az_need > 0 and _pq_az_ranked:
            _added_az = 0
            for _az_isin in _pq_az_ranked:
                if _added_az >= _az_need:
                    break
                if _az_isin in _az_in_pq:
                    continue

                _az_row = _az_src_pq[_az_src_pq["isin"] == _az_isin]
                if _az_row.empty:
                    continue

                _az_class = str(_az_row.iloc[0].get("classificazione", ""))
                _az_bucket = classify_bucket(_az_class)

                # Se il bucket non esiste nel portafoglio, scegli il più vicino disponibile
                if _az_bucket not in _available_buckets or _az_bucket == "Altro":
                    # Priorità: Azionario > Obbligazionario > Bilanciato > primo disponibile
                    for _fallback in ["Azionario", "Obbligazionario", "Bilanciato"]:
                        if _fallback in _available_buckets:
                            _az_bucket = _fallback
                            break
                    else:
                        _az_bucket = _available_buckets[0] if _available_buckets else "Azionario"

                # Crea entry fondo Azimut con colonne minime necessarie
                _az_entry = _az_row.iloc[0].to_dict()
                _az_entry["_peso_bucket"] = alloc_adj.get(_az_bucket, 10)
                _az_df_new = pd.DataFrame([_az_entry])

                # Aggiunge al bucket e ricalcola pesi equi
                if _az_bucket in portfolio_buckets and portfolio_buckets[_az_bucket] is not None:
                    _existing = portfolio_buckets[_az_bucket]
                    _combined = pd.concat([_az_df_new, _existing], ignore_index=True)
                    _combined = _combined.drop_duplicates(subset=["isin"])
                    _n_new = len(_combined)
                    _combined["_peso_fondo"] = round(alloc_adj.get(_az_bucket, 10) / _n_new, 1)
                    portfolio_buckets[_az_bucket] = _combined
                    if _az_bucket not in _available_buckets:
                        _available_buckets.append(_az_bucket)
                else:
                    _az_df_new["_peso_fondo"] = alloc_adj.get(_az_bucket, 10)
                    portfolio_buckets[_az_bucket] = _az_df_new
                    _available_buckets.append(_az_bucket)

                _az_in_pq.add(_az_isin)
                _added_az += 1

            if _added_az > 0:
                _az_names = [str(_az_src_pq[_az_src_pq["isin"]==i]["nome"].values[0])[:30]
                             if not _az_src_pq[_az_src_pq["isin"]==i].empty else i
                             for i in list(_az_in_pq)[:3]]
                st.info(
                    f"🔵 **{_added_az} fondi Azimut aggiunti** al portafoglio: "
                    f"{', '.join(_az_names)}"
                    + (f" e altri {len(_az_in_pq)-3}" if len(_az_in_pq) > 3 else "")
                )
            elif n_min_az_pq > 0:
                st.warning(f"⚠️ Richiesti {n_min_az_pq} fondi Azimut ma catalogo vuoto o "
                           f"file non caricato. Carica il file Azimut dalla sidebar.")

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

        # Lock indicator
        locked_in_bucket = [i for i in st.session_state["pq_locks"]
                            if any(df_bucket["isin"] == i)]
        if locked_in_bucket:
            st.caption(f"🔒 Bloccati: {', '.join(locked_in_bucket)}")

        _pq_col_cfg = {
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
        }
        st.dataframe(display, use_container_width=True, hide_index=True,
                     column_config=_pq_col_cfg)

        # ── LOCK / SOSTITUZIONE ──────────────────────────────────────────
        with st.expander(f"🔄 Sostituisci / Blocca fondo — {bucket}", expanded=False):
            _bucket_isins = df_bucket["isin"].tolist() if "isin" in df_bucket.columns else []
            _bk = bucket.replace(" ", "_")

            # Lock
            _lock_sel = st.multiselect(
                "🔒 Blocca fondi (rimangono indipendentemente dallo score)",
                options=_bucket_isins,
                default=[i for i in _bucket_isins if i in st.session_state["pq_locks"]],
                format_func=lambda x: f"{x} — {str(df_unified[df_unified['isin']==x]['nome'].values[0])[:45] if not df_unified[df_unified['isin']==x].empty else x}",
                key=f"pq_lock_{_bk}",
            )
            if st.button("Salva lock", key=f"save_lock_{_bk}", use_container_width=False):
                for i in _bucket_isins:
                    st.session_state["pq_locks"].discard(i)
                st.session_state["pq_locks"].update(_lock_sel)
                st.toast(f"🔒 Lock aggiornati per {bucket}")
                st.rerun()

            st.markdown("---")
            # Sostituzione manuale
            st.caption("Sostituisci un fondo con uno dalla classifica completa del bucket:")
            # Costruisci classifica completa del bucket
            from utils.scoring import compute_scores_df
            from utils.constraints import classify_bucket as _cb
            _full_bucket_df = df_unified.copy()
            _full_bucket_df["_bucket_tmp"] = _full_bucket_df["classificazione"].apply(_cb)
            _full_bucket_df = _full_bucket_df[_full_bucket_df["_bucket_tmp"] == bucket].copy()
            _full_bucket_df = compute_scores_df(_full_bucket_df)
            _full_bucket_df = _full_bucket_df.sort_values("score_qualita", ascending=False)

            if not _full_bucket_df.empty:
                _sub_c1, _sub_c2 = st.columns(2)
                _isin_out = _sub_c1.selectbox(
                    "Fondo da rimuovere",
                    options=_bucket_isins,
                    format_func=lambda x: f"{x} — {str(df_unified[df_unified['isin']==x]['nome'].values[0])[:40] if not df_unified[df_unified['isin']==x].empty else x}",
                    key=f"pq_out_{_bk}",
                )
                _alternatives = _full_bucket_df[
                    ~_full_bucket_df["isin"].isin(_bucket_isins)
                ]["isin"].tolist()[:30]
                _isin_in = _sub_c2.selectbox(
                    "Sostituisci con",
                    options=_alternatives,
                    format_func=lambda x: f"{x} — {str(_full_bucket_df[_full_bucket_df['isin']==x]['nome'].values[0])[:40] if not _full_bucket_df[_full_bucket_df['isin']==x].empty else x}",
                    key=f"pq_in_{_bk}",
                ) if _alternatives else None

                if _isin_in and st.button(
                    f"↔️ Sostituisci", key=f"pq_sub_{_bk}", type="primary"):
                    st.session_state["pq_replacements"][(bucket, _isin_out)] = _isin_in
                    st.toast(f"✅ {_isin_out} → {_isin_in} nel bucket {bucket}")
                    st.rerun()

                # Mostra sostituzioni attive
                active_subs = {k: v for k, v in st.session_state["pq_replacements"].items()
                               if k[0] == bucket}
                if active_subs:
                    st.caption("Sostituzioni attive:")
                    for (_, old), new in active_subs.items():
                        sc1, sc2 = st.columns([4, 1])
                        sc1.markdown(f"~~{old}~~ → **{new}**")
                        if sc2.button("✕", key=f"rm_sub_{old}_{_bk}"):
                            del st.session_state["pq_replacements"][(bucket, old)]
                            st.rerun()
            else:
                st.info("Non ci sono fondi alternativi disponibili per questo bucket.")

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
        pdf_q = export_portfolio_pdf(
            weights_q, metrics_q,
            title=f"Portafoglio Qualità — {profilo}",
            fund_df=df_unified if not df_unified.empty else None,
            fund_pool={r["ISIN"]: {
                "nome": r["Fondo"], "classificazione": r["Classificazione"],
                "perf_3y": r.get("Perf 3Y %"), "rating_fida": r.get("★ FIDA"),
            } for _, r in df_porto.iterrows()},
        )
        if pdf_q:
            exp_c2.download_button("📄 Esporta PDF", data=pdf_q,
                file_name=f"portafoglio_qualita_{profilo.lower()}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf")


# ===========================================================================
# COMPARATORE PORTAFOGLI
# ===========================================================================
elif nav == "🔀 Comparatore":
    st.title("🔀 Comparatore Portafogli")
    st.markdown(
        "Confronta fianco a fianco Max Sharpe, Min Varianza, Portafoglio Qualità "
        "e un portafoglio personalizzato."
    )

    # Raccoglie portafogli disponibili
    _comp_portfolios: dict[str, dict] = {}

    fe_res = st.session_state.get("fe_result", {})
    if fe_res.get("max_sharpe") and "error" not in fe_res["max_sharpe"]:
        _comp_portfolios["Max Sharpe"] = fe_res["max_sharpe"]
    if fe_res.get("min_variance") and "error" not in fe_res["min_variance"]:
        _comp_portfolios["Min Varianza"] = fe_res["min_variance"]
    if st.session_state.get("bl_result") and "error" not in st.session_state["bl_result"]:
        _comp_portfolios["Black-Litterman"] = st.session_state["bl_result"]

    # Portafoglio Qualità — costruito al volo se dati disponibili
    if not df_unified.empty:
        try:
            _pq_buckets = build_portfolio_quality(
                df_unified, profilo=st.session_state.get("profilo","Equilibrato"),
                fondi_per_bucket=st.session_state.get("fondi_per_bucket",4)
            )
            _pq_weights = {}
            for _bk, _bdf in _pq_buckets.items():
                if _bdf is not None and not _bdf.empty:
                    for _, _r in _bdf.iterrows():
                        _pq_weights[_r.get("isin","")] = (_r.get("_peso_fondo",0) or 0) / 100
            if _pq_weights:
                _comp_portfolios["Qualità"] = {"weights": _pq_weights, "ret": None, "vol": None, "sharpe": None}
        except Exception:
            pass

    if not _comp_portfolios:
        st.info(
            "Nessun portafoglio calcolato ancora. "
            "Vai su **Frontiera Efficiente** e calcola almeno un portafoglio, "
            "oppure usa **Portafoglio Qualità**."
        )
    else:
        st.success(f"Portafogli disponibili: {', '.join(_comp_portfolios.keys())}")

        # ── TABELLA METRICHE ─────────────────────────────────────────────
        st.subheader("📊 Metriche a confronto")
        metrics_rows = []
        for pname, pdata in _comp_portfolios.items():
            n_assets = sum(1 for v in pdata["weights"].values() if v and v > 0.001)
            metrics_rows.append({
                "Portafoglio": pname,
                "Rendimento atteso %": round(pdata["ret"]*100, 2) if pdata.get("ret") else "—",
                "Volatilità %":        round(pdata["vol"]*100, 2) if pdata.get("vol") else "—",
                "Sharpe Ratio":        round(pdata["sharpe"], 3)  if pdata.get("sharpe") else "—",
                "N. asset":            n_assets,
            })
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

        # ── PESI AFFIANCATI ──────────────────────────────────────────────
        st.subheader("⚖️ Composizione a confronto")
        n_cols = len(_comp_portfolios)
        comp_cols = st.columns(n_cols)

        # Pool nomi
        _pool_names: dict = {}
        try:
            from utils.etf_tickers import ITALIAN_STOCKS, DIVIDEND_STOCKS, ISIN_TO_TICKER
            from utils.etf_static import ETF_STATIC as _ESL
            for e in _ESL: _pool_names[e["isin"]] = e["nome"]
            for i, d in ITALIAN_STOCKS.items(): _pool_names[i] = d["nome"]
            for i, d in DIVIDEND_STOCKS.items(): _pool_names[i] = d["nome"]
            if not df_unified.empty:
                for _, r in df_unified.iterrows():
                    if r.get("isin"): _pool_names[r["isin"]] = str(r.get("nome",""))[:45]
        except Exception:
            pass

        for col, (pname, pdata) in zip(comp_cols, _comp_portfolios.items()):
            w = {k: v for k, v in pdata["weights"].items() if v and v > 0.001}
            w_rows = [{"Nome": _pool_names.get(isin, isin)[:35], "Peso %": round(v*100,1)}
                      for isin, v in sorted(w.items(), key=lambda x: -x[1])]
            with col:
                st.markdown(f"**{pname}**")
                st.dataframe(pd.DataFrame(w_rows), use_container_width=True,
                             hide_index=True, height=300)
                fig_c = px.pie(pd.DataFrame(w_rows), values="Peso %", names="Nome",
                               hole=0.35, color_discrete_sequence=px.colors.sequential.Blues_r)
                fig_c.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0),
                                    showlegend=False)
                fig_c.update_traces(textposition="inside", textinfo="percent",
                                    textfont_size=9)
                st.plotly_chart(fig_c, use_container_width=True)

        # ── GRAFICO RADAR METRICHE ────────────────────────────────────────
        _quant = [(n, d) for n, d in _comp_portfolios.items()
                  if d.get("ret") and d.get("vol") and d.get("sharpe")]
        if len(_quant) >= 2:
            st.subheader("🎯 Radar: rendimento vs rischio vs sharpe")
            fig_rad = go.Figure()
            cats = ["Rendimento %", "1/Volatilità (sicurezza)", "Sharpe"]
            max_r  = max(d["ret"]*100 for _, d in _quant) or 1
            max_sh = max(d["sharpe"]  for _, d in _quant) or 1
            min_v  = min(d["vol"]*100 for _, d in _quant) or 0.01

            for pname, pdata in _quant:
                vals = [
                    pdata["ret"]*100 / max_r * 10,
                    min_v / (pdata["vol"]*100) * 10,
                    pdata["sharpe"] / max_sh * 10,
                ]
                fig_rad.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=cats + [cats[0]],
                    fill="toself", name=pname, opacity=0.7,
                ))
            fig_rad.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=True, height=380,
            )
            st.plotly_chart(fig_rad, use_container_width=True)

        # ── EXPORT COMPARATORE ────────────────────────────────────────────
        st.markdown("---")
        if st.button("📥 Esporta confronto Excel"):
            import io as _io
            buf = _io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                pd.DataFrame(metrics_rows).to_excel(writer, sheet_name="Metriche", index=False)
                for pname, pdata in _comp_portfolios.items():
                    w = {k: v for k, v in pdata["weights"].items() if v and v > 0.001}
                    w_df = pd.DataFrame([
                        {"ISIN": k, "Nome": _pool_names.get(k,k)[:45], "Peso %": round(v*100,2)}
                        for k, v in sorted(w.items(), key=lambda x: -x[1])
                    ])
                    sheet_name = pname[:31]
                    w_df.to_excel(writer, sheet_name=sheet_name, index=False)
            st.download_button(
                "📥 Scarica confronto",
                data=buf.getvalue(),
                file_name=f"confronto_portafogli_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ===========================================================================
# ETF UNIVERSE
# ===========================================================================
elif nav == "🌐 ETF Universe":
    st.title("🌐 ETF Universe — Lista C")

    # Carica — con fallback immediato su dati statici
    with st.spinner("Caricamento ETF Universe..."):
        try:
            df_etf = _load_etf_universe_cached()
        except Exception as e:
            df_etf = pd.DataFrame()

        # Se vuoto o senza colonne chiave → usa statico immediatamente
        if df_etf.empty or "nome" not in df_etf.columns:
            from utils.etf_static import get_static_etf_df
            from utils.etf_tickers import ISIN_TO_TICKER, TER_VERIFIED
            df_etf = get_static_etf_df()
            df_etf["ticker"] = df_etf["isin"].map(ISIN_TO_TICKER)
            df_etf["ter"] = df_etf["isin"].map(TER_VERIFIED)
            df_etf["_fonte_ter"] = df_etf["ter"].apply(lambda x: "KID verificato" if pd.notna(x) else "n/d")
            df_etf["_fonte_perf"] = "n/d (carica con 🔄)"

        # Converti valori numerici (potrebbe essere stringa da Excel)
        for _nc in ["ter", "perf_1y", "perf_3y", "perf_5y"]:
            if _nc in df_etf.columns:
                df_etf[_nc] = pd.to_numeric(df_etf[_nc], errors="coerce")

    # Toolbar
    tb1, tb2, tb3 = st.columns([2, 1, 1])
    perf_ok = int(df_etf["perf_1y"].notna().sum()) if "perf_1y" in df_etf.columns else 0
    tb1.caption(f"**{len(df_etf)} ETF** in lista · TER verificato: 85/85 · Perf yfinance: {perf_ok}/85")

    if tb2.button("🔄 Aggiorna rendimenti (yfinance)", use_container_width=True):
        _load_etf_universe_cached.clear()
        from utils.etf_fetcher import _build_from_yfinance_and_static
        import os
        from pathlib import Path as _P
        cache_f = _P("data/etf_universe.xlsx")
        if cache_f.exists(): os.remove(cache_f)

        prog_bar  = st.progress(0.0, text="Inizializzazione…")
        prog_text = st.empty()

        def _etf_progress(pct: float, msg: str):
            prog_bar.progress(min(pct, 1.0), text=msg)
            prog_text.caption(msg)

        df_etf = _build_from_yfinance_and_static(progress_callback=_etf_progress)
        prog_bar.empty(); prog_text.empty()
        ok = int(df_etf["perf_1y"].notna().sum()) if "perf_1y" in df_etf.columns else 0
        st.success(f"✅ Completato: {ok}/85 ETF con dati reali da yfinance")
        st.rerun()

    with tb3.expander("➕ Aggiungi ISIN"):
        new_isins_txt = st.text_area("ISINs (uno per riga)", height=80, key="new_etf_isins")
        if st.button("Aggiungi"):
            new_list = [x.strip() for x in new_isins_txt.strip().split("\n") if x.strip()]
            if new_list:
                with st.spinner(f"Recupero {len(new_list)} ISINs..."):
                    df_etf = fetch_etf_universe(extra_isins=new_list)
                    _load_etf_universe_cached.clear()
                st.success(f"Aggiunti {len(new_list)} ISINs.")

    # Filtri
    if True:
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

        # Calcola perf e volatilità on-demand via yfinance per ETF visibili
        # (solo se i dati mancano nel file cache)
        _etf_need_perf = df_display[
            df_display["perf_1y"].isna() & df_display["ticker"].notna()
        ]["ticker"].dropna().tolist() if "perf_1y" in df_display.columns else []

        if _etf_need_perf:
            with st.spinner(f"Calcolo rendimenti per {len(_etf_need_perf)} ETF…"):
                try:
                    import yfinance as _yf
                    _hist = _yf.download(
                        " ".join(_etf_need_perf[:30]),  # max 30 alla volta
                        period="3y", interval="1mo",
                        auto_adjust=True, progress=False, threads=False
                    )
                    if not _hist.empty:
                        _close = _hist["Close"] if hasattr(_hist.columns, "levels") else _hist
                        for _tk in _etf_need_perf[:30]:
                            if _tk not in _close.columns:
                                continue
                            _s = _close[_tk].dropna()
                            if len(_s) < 12:
                                continue
                            _p1 = (_s.iloc[-1]/_s.iloc[-13]-1)*100 if len(_s)>=13 else None
                            _p3 = (_s.iloc[-1]/_s.iloc[-37]-1)*100/3 if len(_s)>=37 else None
                            _vol= float(_s.pct_change().dropna().std()*(12**0.5)*100)
                            # Aggiorna df_display
                            _idx = df_display[df_display["ticker"]==_tk].index
                            if len(_idx):
                                if _p1: df_display.loc[_idx, "perf_1y"] = round(_p1,2)
                                if _p3: df_display.loc[_idx, "perf_3y"] = round(_p3,2)
                                df_display.loc[_idx, "volatilita"] = round(_vol,2)
                                df_display.loc[_idx, "_fonte_perf"] = "yfinance"
                except Exception:
                    pass

        # Converti colonne numeriche (potrebbero essere stringhe dalla cache)
        for _nc in ["ter","perf_1y","perf_3y","perf_5y","volatilita"]:
            if _nc in df_display.columns:
                df_display[_nc] = pd.to_numeric(df_display[_nc], errors="coerce")

        cols_etf = [c for c in ["isin","ticker","nome","categoria","ter",
                                 "perf_1y","perf_3y","volatilita"]
                    if c in df_display.columns]
        col_config_etf = {
            "isin":       st.column_config.TextColumn("ISIN", width="small"),
            "ticker":     st.column_config.TextColumn("Ticker", width="small"),
            "nome":       st.column_config.TextColumn("Nome", width="large"),
            "categoria":  st.column_config.TextColumn("Categoria"),
            "ter":        st.column_config.NumberColumn("TER %", format="%.2f"),
            "perf_1y":    st.column_config.NumberColumn("Rend 1Y %", format="%.1f"),
            "perf_3y":    st.column_config.NumberColumn("Rend 3Y %/a", format="%.1f"),
            "volatilita": st.column_config.NumberColumn("Vol % (ann)", format="%.1f"),
        }
        st.dataframe(df_display[cols_etf], column_config=col_config_etf,
                     use_container_width=True, height=500)
        st.caption("Rendimenti e volatilità calcolati da yfinance (prezzi mensili). "
                   "Blank = ticker non trovato su Yahoo Finance.")

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
# GUIDA
# ===========================================================================
elif nav == "📖 Guida":
    st.title("📖 Guida — Come funziona l'app")
    st.markdown(
        "Questa pagina spiega le **regole che governano l'app**, "
        "come vengono costruiti i portafogli e cosa significano i termini tecnici."
    )

    # ── TAB PRINCIPALI ──────────────────────────────────────────────────────
    g1, g2, g3, g4, g5 = st.tabs([
        "📋 Le Liste e i Dati",
        "⭐ Score Qualità & Portafoglio Qualità",
        "📈 Frontiera Efficiente",
        "🔮 Black-Litterman",
        "📐 Regole di Diversificazione",
    ])

    # ── TAB 1: LE LISTE ─────────────────────────────────────────────────────
    with g1:
        st.subheader("Le sorgenti dati")
        st.markdown("""
L'app lavora su **tre categorie** di strumenti, ognuna con una fonte diversa:

| Lista | Contenuto | Fonte |
|-------|-----------|-------|
| **Fondi Generalisti** | Fondi terzi + Azimut con classificazione globale/bilanciata/flessibile | File Excel caricato in sidebar |
| **Fondi Tematici** | Fondi specializzati (emergenti, settoriali, tematici, high yield…) | File Excel caricato in sidebar |
| **Lista C — ETF** | 85 ETF/ETC/ETN selezionati + titoli italiani + dividend stocks | Dataset statico + yfinance per rendimenti |
""")

        with st.expander("Come viene costruita Lista A (100 fondi generalisti)"):
            st.markdown("""
**Criteri di eleggibilità:**
- Classificazione FIDA contiene almeno una keyword: *Globali, Globale, Bilanciati, Flessibili, Ritorno Assoluto, Multi-Asset, Allocation, Azionari Globali, Obbligazionari Globali*
- Performance 3Y disponibile e non nulla
- Rating FIDA ≥ 3 stelle (se disponibile) — i fondi senza rating passano comunque
- Performance 3Y annualizzata ≥ 0%

**Vincoli di selezione (top 100):**
- Max **3 fondi** per casa di gestione
- Max **2 fondi** per sottoclassificazione FIDA
- Max **1 fondo** con la stessa "radice strategia" (prime 3 parole significative del nome — evita ACC e MINC della stessa strategia)
- Ordinamento: **Score Qualità** decrescente (vedi tab successivo)
- Tiebreaker a parità di score: vince il fondo con **retrocessione più alta**
""")

        with st.expander("Come viene costruita Lista B (100 fondi tematici)"):
            st.markdown("""
**Criteri di eleggibilità:**
- Classificazione FIDA **non** contiene le keyword generaliste (esclusi da Lista A)
- Include: Frontier Markets, Emergenti specifici, Settoriali (Tech, Healthcare, Biotech), Tematici (ESG, Robotica, Acqua), High Yield, CoCo/AT1, Convertibili, ABS, Sukuk, ecc.
- Performance 1Y disponibile

**Vincoli:**
- Max **2 fondi** per casa di gestione
- Max **1 fondo** per sotto-tema specifico
""")

        with st.expander("Lista C — ETF e come vengono aggiornati i prezzi"):
            st.markdown("""
**Composizione:**
85 ETF/ETC/ETN selezionati manualmente, organizzati per categoria:
Azioni Mondo, Azioni USA, Azioni Europa, Azioni Emergenti, Obbligazioni Governativi EUR,
Obbligazioni Societari, High Yield, Obbligazioni Emergenti, iBonds, BTP/Italia, Materie Prime.

**Più:** 35 titoli FTSE MIB italiani e 30 dividend stocks internazionali (TDIV/EUDV).

**TER:** verificato da KID/KIID ufficiali dell'emittente (giugno 2025).

**Rendimenti:** scaricati in tempo reale da **Yahoo Finance** tramite una mappa ISIN→ticker
verificata (es. `IE00B4L5Y983` → `SWDA.MI`). Copertura: 78/85 ETF.

**Aggiornamento:** automaticamente ogni 24 ore al primo caricamento; manualmente con
il pulsante "🔄 Aggiorna rendimenti" nella pagina ETF Universe.
""")

        with st.expander("Fonti dati per le serie storiche (ottimizzazione)"):
            st.markdown("""
Per calcolare la **frontiera efficiente** servono serie storiche di prezzi/NAV.
L'app le recupera con questa cascata:

1. **Cache locale** (24h) — evita chiamate ripetute
2. **Morningstar API** — per i fondi con ISIN noto (usa cloudscraper per aggirare bot-detection)
3. **FondiDoc.it** — fallback per i fondi italiani
4. **Yahoo Finance** — per ETF (via ticker map) e azioni quotate (ticker diretto)
5. **Serie sintetica** — costruita dai rendimenti annuali disponibili (2022, 2023, 2024, YTD),
   interpolando mensilmente. Meno precisa ma funziona sempre.

> ⚠️ Le serie sintetiche producono risultati meno affidabili nell'ottimizzazione
> rispetto alle serie reali. Preferire ETF (yfinance) quando possibile.
""")

    # ── TAB 2: SCORE QUALITÀ ────────────────────────────────────────────────
    with g2:
        st.subheader("Score Qualità — la formula")
        st.markdown("""
Lo **Score Qualità** è il punteggio che l'app usa per classificare i fondi e costruire
il Portafoglio Qualità. È un indicatore sintetico che combina rendimento, efficienza
e consistenza.
""")

        st.code("""
# FORMULA SCORE QUALITÀ
base_score = (perf_3y_ann × 0.50)           # 50%: rendimento triennale annualizzato
           + (perf_3y_ann / volatilità × 0.30)  # 30%: efficienza (Sharpe proxy)
           + (perf_1y × 0.20)               # 20%: rendimento ultimo anno

# Moltiplicatore stelle FIDA
fida_mult = { 5★: 1.30,  4★: 1.15,  3★: 1.00,  2★: 0.90,  1★: 0.80,  n/d: 1.00 }
score = base_score × fida_mult

# Penalità rendimento triennale negativo
if perf_3y_ann < 0:  score × 0.50

# Penalità anni molto negativi (< -10%)
bad_years = numero anni (2022, 2023, 2024) con perf < -10%
consistency_bonus = max(0.40,  1.0 - bad_years × 0.15)
score × consistency_bonus

# Tiebreaker a parità di score (±5%): vince il fondo con retrocessione più alta
""", language="python")

        st.markdown("""
**Interpretazione:**
- Score **> 10**: fondo eccellente, ottimo rapporto rendimento/rischio
- Score **5–10**: buono, nella media alta
- Score **0–5**: nella media
- Score **negativo**: rendimento triennale negativo (il moltiplicatore 0.5 abbatte lo score)
""")

        st.subheader("Portafoglio Qualità — come funziona")
        st.markdown("""
Il Portafoglio Qualità **non** usa l'ottimizzazione quantitativa. Seleziona i fondi
migliori per **Score Qualità** rispettando vincoli di diversificazione.

**Flusso:**

1. Scegli il **profilo di rischio** → definisce le % target per macro asset class

| Profilo | Obbligazioni | Bilanciato | Azionario | Monetario | Alternativo |
|---------|-------------|-----------|-----------|-----------|-------------|
| Conservativo | 60% | 15% | 10% | 15% | — |
| Equilibrato | 40% | 25% | 30% | — | 5% |
| Accrescitivo | 25% | 15% | 55% | — | 5% |
| Dinamico | 15% | — | 75% | — | 10% |

2. Per ogni **bucket** (Azionario, Obbligazionario, Bilanciato…) l'app:
   - Filtra i fondi per classificazione FIDA
   - Ordina per Score Qualità decrescente
   - Applica vincoli di diversificazione → seleziona i top N fondi

3. Il peso di ogni fondo = peso del bucket ÷ numero di fondi nel bucket
""")

        with st.expander("Vincoli di diversificazione nel Portafoglio Qualità"):
            st.markdown("""
Per ogni bucket:
- Max **1 fondo** per casa di gestione
- Max **1 fondo** per sottoclassificazione FIDA
- Max **1 fondo** con la stessa radice strategia (prime 3 parole significative del nome)
- Max **2 fondi** dalla stessa macro-area geografica (US, Europe, Emerging, Japan, Asia, Global, Italy)
- Almeno **1 fondo** con "Global/Globale/Internazionale" nella classificazione per ogni bucket
""")

    # ── TAB 3: FRONTIERA EFFICIENTE ─────────────────────────────────────────
    with g3:
        st.subheader("Cos'è la Frontiera Efficiente")
        st.markdown("""
La **Frontiera Efficiente** è il concetto centrale della Teoria Moderna del Portafoglio
(Markowitz, 1952). È la curva che mostra tutti i portafogli che offrono il **massimo
rendimento per ogni dato livello di rischio** (o il minimo rischio per ogni dato rendimento).

Qualsiasi portafoglio che si trova **al di sotto** della curva è sub-ottimale:
esiste un portafoglio sulla frontiera che offre lo stesso rendimento con meno rischio,
o più rendimento con lo stesso rischio.
""")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**Come la costruisce l'app:**
1. Recupera serie storiche per tutti gli asset selezionati
2. Calcola rendimenti attesi (media storica annualizzata)
3. Calcola matrice di covarianza con **Ledoit-Wolf shrinkage** (più stabile della covarianza campionaria)
4. Ottimizza con **PyPortfolioOpt** su 50 punti da Min Varianza a Max Rendimento
5. Genera 5.000 portafogli casuali (Monte Carlo) per visualizzare lo spazio possibile
""")
        with col2:
            st.info("""
**Limitazione importante:**
I rendimenti attesi sono basati su dati **storici** (3 anni di default).
I rendimenti passati non sono predittivi di quelli futuri.

La frontiera è utile per capire le **relazioni di rischio/rendimento**
tra gli asset, non per fare previsioni precise.
""")

        st.markdown("---")
        st.subheader("⭐ Max Sharpe")
        st.markdown("""
Il **portafoglio Max Sharpe** massimizza l'**Indice di Sharpe**:

$$\\text{Sharpe} = \\frac{\\text{Rendimento atteso} - \\text{Risk-free rate}}{\\text{Volatilità}}$$

È il portafoglio che offre il **miglior rapporto tra rendimento in eccesso e rischio**.
Se il risk-free rate è 2.5% e un portafoglio rende 12% con 8% di volatilità:
Sharpe = (12 - 2.5) / 8 = **1.19**

**Quando usarlo:** quando vuoi massimizzare il rendimento per unità di rischio.
Tipicamente adatto a profili Equilibrato/Accrescitivo.

> Sul grafico: **stella rossa 🔴**
""")

        st.subheader("🛡️ Min Varianza")
        st.markdown("""
Il **portafoglio Min Varianza** (o Minimum Variance Portfolio) minimizza la
**volatilità del portafoglio**, indipendentemente dal rendimento.

Sfrutta la **correlazione tra asset**: combinando asset con correlazione bassa o negativa
si ottiene un portafoglio complessivo meno volatile della media dei suoi componenti.

**Quando usarlo:** quando la priorità è la stabilità del valore. Adatto a profili
Conservativo/Equilibrato, o quando si è vicini a una necessità di liquidare.

> Sul grafico: **diamante blu 🔵** — è sempre il punto più a sinistra della frontiera
""")

        with st.expander("Come impostare i vincoli nell'ottimizzazione"):
            st.markdown("""
**Peso minimo (default 3%):** evita posizioni troppo piccole (transaction costs).
**Peso massimo (default 30%):** impone diversificazione, evita concentrazione eccessiva.

**Forza inclusione:** inserisce un asset con peso minimo garantito (utile per BTP/ETF specifici).
**Min fondi Azimut:** garantisce la presenza di almeno N fondi Azimut nel portafoglio finale.

**Vincoli asset class:** attivati dall'Auto-composizione — il portafoglio deve rispettare
le % target per macro classe (±10% di tolleranza).
""")

        with st.expander("Paraametri configurabili (sidebar)"):
            st.markdown("""
| Parametro | Default | Significato |
|-----------|---------|-------------|
| **Risk-free rate** | 2.5% | Tasso privo di rischio usato nel calcolo dello Sharpe |
| **Periodo ottimizzazione** | 3Y | Finestra storica per calcolo rendimenti e covarianza |
| **Peso minimo** | 3% | Peso minimo per ogni asset nel portafoglio |
| **Peso massimo** | 30% | Peso massimo per ogni asset nel portafoglio |
""")

    # ── TAB 4: BLACK-LITTERMAN ──────────────────────────────────────────────
    with g4:
        st.subheader("Black-Litterman — Intuizione")
        st.markdown("""
Il modello **Black-Litterman** (Fischer Black e Robert Litterman, Goldman Sachs 1990)
risolve un problema pratico del modello di Markowitz: i rendimenti attesi storici
sono **molto instabili** e producono portafogli concentrati e poco intuitivi.

**L'idea:** invece di partire solo dai dati storici, si parte dai **pesi di mercato**
(capitalizzazione — quanto il mercato stesso "pensa" che valga ogni asset) e poi
si integrano le **view soggettive** del gestore con una confidenza esplicita.
""")

        col1b, col2b = st.columns(2)
        with col1b:
            st.markdown("""
**Come funziona in pratica:**

1. **Prior (equilibrio):** i pesi di mercato implicano rendimenti attesi di equilibrio
2. **View:** inserisci la tua aspettativa su uno o più asset
   - Es: *"Mi aspetto che gli Emergenti rendano 10% l'anno"*
   - Con confidenza 0.7 (abbastanza sicuro)
3. **Posterior:** BL combina prior e view → nuovi rendimenti attesi "bayesiani"
4. **Ottimizzazione:** Max Sharpe sui rendimenti posteriori
""")
        with col2b:
            st.info("""
**Quando ha senso usarlo:**
- Hai una tesi specifica su un settore o area geografica
- Vuoi "inclinare" il portafoglio verso una view senza abbandonare la diversificazione
- Es: sovrappeso azionario emergente perché credi alla ripresa cinese

**Quando NON usarlo:**
- Non hai view specifiche → usa Max Sharpe
- Le tue view sono già riflesse nel prezzo → il mercato sa già
""")

        st.markdown("""
**Confidenza:** valore da 0.1 a 1.0.
- **1.0** = certezza assoluta sulla view → il portafoglio si avvicina molto alla tua aspettativa
- **0.5** = view moderata → BL bilancia 50/50 tra prior di mercato e tua view
- **0.1** = view molto incerta → il portafoglio rimane vicino all'equilibrio di mercato

> Sul grafico: **pentagono verde 🟢** — appare solo se BL è abilitato con almeno una view
""")

    # ── TAB 5: REGOLE DIVERSIFICAZIONE ─────────────────────────────────────
    with g5:
        st.subheader("Le regole che ci siamo dati")
        st.markdown("""
Queste regole sono state definite esplicitamente per garantire portafogli
**professionalmente diversificati** e non influenzati da bias di selezione.
""")

        with st.expander("📋 Regole Liste A/B (preselection)"):
            st.markdown("""
**Fondi Generalisti:**
- Eleggibilità: classificazione FIDA generalista + perf 3Y disponibile + rating ≥ 3★ (o n/d) + perf 3Y ≥ 0%
- **Max 3 fondi** per casa di gestione
- **Max 2 fondi** per sottoclassificazione FIDA
- **Max 1 fondo** per "radice strategia" (prime 3 parole significative del nome, escludendo share class come ACC/MINC)
- Copertura obbligatoria di almeno 5 macro-aree: Azionario globale, Obbligazionario globale, Bilanciato, Ritorno assoluto, Flessibile

**Fondi Tematici:**
- **Max 2 fondi** per casa di gestione
- **Max 1 fondo** per sotto-tema specifico
""")

        with st.expander("⭐ Regole Portafoglio Qualità"):
            st.markdown("""
Per ogni bucket dell'allocazione target:

- **Max 1 fondo per casa di gestione** — nessun "doppio" della stessa asset manager
- **Max 1 fondo per sottoclassificazione FIDA** — diversificazione tra stili
- **Max 1 fondo per radice strategia** — evita che ACC e MINC della stessa strategia finiscano insieme
- **Max 2 fondi dalla stessa macro-area geografica** — tra US, Europe, Emerging, Japan, Asia, Global, Italy
- **Tiebreaker:** a parità di score (±5%), vince il fondo con retrocessione più alta

Puoi **bloccare** un fondo (rimane nel portafoglio indipendentemente dallo score) e
**sostituire** manualmente un fondo con un alternativo dalla classifica completa del bucket.
""")

        with st.expander("📈 Regole Frontiera Efficiente"):
            st.markdown("""
- **Minimo 3 asset, massimo 30** per avere una frontiera significativa
- **Matrice di covarianza:** Ledoit-Wolf shrinkage (riduce l'instabilità della covarianza campionaria)
- **Rendimenti attesi:** media storica annualizzata (frequenza mensile × 12)
- **Monte Carlo:** 5.000 portafogli casuali per visualizzare lo spazio delle possibilità
- **Frontiera:** 50 punti da Min Varianza a Max Rendimento
- **Vincoli settoriali (Auto-composizione):** ±10% di tolleranza sul target di asset class
""")

        with st.expander("🔮 Regole Black-Litterman"):
            st.markdown("""
- **Prior:** pesi uguali tra gli asset (in assenza di capitalizzazione di mercato per i fondi)
  o pesi di mercato per gli ETF azionari
- **Tau** (parametro di incertezza sul prior): 0.05 (default PyPortfolioOpt)
- **Omega** (matrice di incertezza delle view): proporzionale alla varianza dell'asset
- Le view sono **assolute** (rendimento atteso diretto), non relative
- Confidenza > 0.8 → la view domina; < 0.3 → il prior domina
""")

        with st.expander("📊 Classificazione automatica degli strumenti"):
            st.markdown("""
Quando la colonna "Classificazione FIDA" del file Excel è vuota, l'app **inferisce
la classificazione dal nome del fondo** usando regole keyword:

| Parole chiave nel nome | Classificazione inferita |
|------------------------|--------------------------|
| equity, azionari, stock, dividend, growth fund | Azionari |
| emerging market, frontier market | Azionari Emergenti |
| technology, healthcare, biotech, innovation | Azionari Tematici |
| bond, obbligazionari, fixed income, high yield | Obbligazionari |
| government bond, sovereign | Obbligazionari Governativi |
| money market, monetario, overnight | Monetario |
| balanced, bilanciati, multi-asset, patrimoine | Bilanciati |
| flexible, flessibili, dynamic allocation | Flessibili |
| absolute return, ritorno assoluto, market neutral | Ritorno Assoluto |
| commodity, materie prime, gold, oil | Materie Prime |

L'ordine è importante: *Obbligazionari* viene cercato PRIMA di *Azionari* per evitare
che "obbligazionari" (che contiene "azionari") venga classificato erroneamente.
""")


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
