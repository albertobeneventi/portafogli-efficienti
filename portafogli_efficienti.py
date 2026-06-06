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
    /* ── Sidebar sempre visibile — nasconde il pulsante collasso ── */
    [data-testid="stSidebar"] {{
        background-color: {NAVY};
        overflow-y: auto !important;
    }}
    /* Nasconde la freccia collasso sidebar */
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}
    /* Nasconde anche il pulsante < che appare sul bordo */
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
                     help="Forza il ricaricamento di fondi, ETF e liste"):
            _load_all_data.clear()
            _load_preselection.clear()
            _load_etf_universe_cached.clear()
            # Rimuovi chiavi di risultato dall'optimizer
            for _k in ["fe_result","fe_price_dict","bl_result","fe_selected_isins",
                        "fe_ac_map","fe_ac_target"]:
                st.session_state.pop(_k, None)
            st.toast("✅ Cache svuotata — pagina in ricarica")
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
        "Costruisci un portafoglio ottimizzato. "
        "**Step 1**: scegli gli strumenti (o usa l'auto-selezione per asset class) → "
        "**Step 2**: vincoli → **Step 3**: calcola."
    )

    # ── POOL GLOBALE asset disponibili ─────────────────────────────────────
    from utils.etf_tickers import (get_dividend_stocks_df, DIVIDEND_STOCKS,
                                    ITALIAN_STOCKS, ISIN_TO_TICKER)
    from utils.etf_static import ETF_STATIC as _ETF_STATIC_LIST

    if "fe_selected_isins" not in st.session_state:
        st.session_state["fe_selected_isins"] = []

    # Costruisce pool completo una sola volta per pagina
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
        df_etf_fe = pd.DataFrame()
        # fallback: usa static
        from utils.etf_static import get_static_etf_df
        df_etf_fe = get_static_etf_df()
        _pool_from_df(df_etf_fe)

    # Aggiungi azioni italiane e dividend
    for _isin, _info in ITALIAN_STOCKS.items():
        _pool_add(_isin, {"isin": _isin, "nome": _info["nome"],
                          "classificazione": f"Azione — {_info['settore']}",
                          "ticker": _info["ticker"]})
    for _isin, _info in DIVIDEND_STOCKS.items():
        _pool_add(_isin, {"isin": _isin, "nome": _info["nome"],
                          "classificazione": f"Dividendo — {_info['settore']} ({_info['paese']})",
                          "ticker": _info["ticker"]})

    # helper: label leggibile per un ISIN
    def _label(isin: str) -> str:
        nome = _all_fund_pool.get(isin, {}).get("nome", isin)
        return f"{isin} — {str(nome)[:50]}"

    # ── STEP 1: SELEZIONE ──────────────────────────────────────────────────
    st.subheader("1️⃣ Seleziona strumenti")

    fe_tab_a, fe_tab_b, fe_tab_c, fe_tab_it, fe_tab_div, fe_tab_isin = st.tabs([
        "📋 Lista A", "🎯 Lista B", "🌐 ETF",
        "🇮🇹 Titoli Italiani", "💰 Dividend", "✏️ ISIN/Ticker libero"
    ])

    # ── Funzione generica di selezione con data_editor ──────────────────────
    def _selection_tab(df: pd.DataFrame, tab_key: str, display_cols: list,
                       col_cfg: dict | None = None, empty_msg: str = "Nessun dato."):
        """
        Tabella con checkbox editabile direttamente.
        Usa st.data_editor: spunta ✅ la riga → viene aggiunta alla selezione.
        Pulsante "Applica" conferma le modifiche.
        """
        if df is None or df.empty:
            st.info(empty_msg)
            return

        srch = st.text_input("🔍 Cerca", key=f"srch_{tab_key}",
                              placeholder="nome, ISIN, categoria…")
        show_df = df.copy()
        if srch:
            m = show_df.apply(
                lambda c: c.astype(str).str.contains(srch, case=False, na=False)
            ).any(axis=1)
            show_df = show_df[m]

        # Costruisce df da mostrare con colonna Seleziona editabile
        show_cols = [c for c in display_cols if c in show_df.columns]
        disp = show_df[show_cols].copy().head(200).reset_index(drop=True)
        # Colonna checkbox precompilata con stato attuale
        if "isin" in disp.columns:
            disp.insert(0, "Seleziona",
                        disp["isin"].isin(st.session_state["fe_selected_isins"]))

        # Column config
        cfg: dict = {
            "Seleziona": st.column_config.CheckboxColumn(
                "✅", width="small",
                help="Spunta per aggiungere al portafoglio",
            ),
        }
        # Rendi non-editabili tutte le colonne tranne "Seleziona"
        for c in show_cols:
            if c == "isin":
                cfg[c] = st.column_config.TextColumn("ISIN", width="small", disabled=True)
            elif c == "nome":
                cfg[c] = st.column_config.TextColumn("Nome", width="large", disabled=True)
            elif c not in (col_cfg or {}):
                cfg[c] = st.column_config.TextColumn(c.replace("_"," ").title(), disabled=True)
        # col_cfg sovrascrive (già con disabled=True impostato dal chiamante)
        if col_cfg:
            cfg.update(col_cfg)

        edited = st.data_editor(
            disp,
            use_container_width=True,
            hide_index=True,
            column_config=cfg,
            height=300,
            key=f"de_{tab_key}",
            disabled=[c for c in show_cols],   # solo "Seleziona" è editabile
        )

        # Confronta checkbox prima/dopo per trovare variazioni
        if "Seleziona" in edited.columns and "isin" in edited.columns:
            checked   = set(edited[edited["Seleziona"] == True]["isin"].tolist())
            unchecked = set(edited[edited["Seleziona"] == False]["isin"].tolist())
            prev_sel  = set(st.session_state["fe_selected_isins"])
            new_add   = checked - prev_sel
            new_rem   = unchecked & prev_sel

            if new_add or new_rem:
                btn_lbl = []
                if new_add: btn_lbl.append(f"➕ {len(new_add)} aggiunti")
                if new_rem: btn_lbl.append(f"➖ {len(new_rem)} rimossi")
                if st.button(
                    "✅ Applica selezione  ·  " + "  ".join(btn_lbl),
                    key=f"apply_{tab_key}",
                    type="primary",
                    use_container_width=True,
                ):
                    for isin in new_add:
                        if isin not in st.session_state["fe_selected_isins"]:
                            st.session_state["fe_selected_isins"].append(isin)
                    st.session_state["fe_selected_isins"] = [
                        i for i in st.session_state["fe_selected_isins"]
                        if i not in new_rem
                    ]
                    if new_add:
                        st.toast(f"✅ Aggiunti {len(new_add)} strumenti")
                    if new_rem:
                        st.toast(f"🗑️ Rimossi {len(new_rem)} strumenti")
                    st.rerun()
            else:
                n_sel = len(checked & prev_sel)
                if n_sel:
                    st.caption(f"✅ {n_sel} strument{'o' if n_sel==1 else 'i'} selezionat{'o' if n_sel==1 else 'i'} in questa lista")

    FUND_COLS = ["isin", "nome", "classificazione", "perf_1y", "perf_3y", "volatilita", "rating_fida"]
    ETF_COLS  = ["isin", "nome", "categoria", "ter", "perf_1y", "perf_3y"]
    IT_COLS   = ["isin", "ticker", "nome", "settore"]
    DIV_COLS  = ["isin", "ticker", "nome", "settore", "paese", "div_yield_est"]

    _FUND_CFG = {
        "classificazione": st.column_config.TextColumn("Classificazione", disabled=True),
        "perf_1y":  st.column_config.NumberColumn("Perf 1Y %", format="%.1f", disabled=True),
        "perf_3y":  st.column_config.NumberColumn("Perf 3Y %", format="%.1f", disabled=True),
        "volatilita": st.column_config.NumberColumn("Vol %", format="%.1f", disabled=True),
        "rating_fida": st.column_config.NumberColumn("★ FIDA", format="%d", disabled=True),
    }
    _ETF_CFG = {
        "categoria": st.column_config.TextColumn("Categoria", disabled=True),
        "ter":    st.column_config.NumberColumn("TER %", format="%.2f", disabled=True),
        "perf_1y": st.column_config.NumberColumn("Perf 1Y %", format="%.1f", disabled=True),
        "perf_3y": st.column_config.NumberColumn("Perf 3Y %/a", format="%.1f", disabled=True),
    }
    _IT_CFG = {
        "ticker":  st.column_config.TextColumn("Ticker", width="small", disabled=True),
        "settore": st.column_config.TextColumn("Settore", disabled=True),
    }
    _DIV_CFG = {
        "ticker":  st.column_config.TextColumn("Ticker", width="small", disabled=True),
        "settore": st.column_config.TextColumn("Settore", disabled=True),
        "paese":   st.column_config.TextColumn("Paese", width="small", disabled=True),
        "div_yield_est": st.column_config.NumberColumn("Yield % (stima)", format="%.1f", disabled=True),
    }

    with fe_tab_a:
        if lista_a.empty:
            st.warning("Carica i file Excel dalla sidebar per vedere i fondi generalisti.")
        else:
            _selection_tab(lista_a, "A", FUND_COLS, col_cfg=_FUND_CFG)

    with fe_tab_b:
        if lista_b.empty:
            st.warning("Carica i file Excel dalla sidebar per vedere i fondi tematici.")
        else:
            _selection_tab(lista_b, "B", FUND_COLS, col_cfg=_FUND_CFG)

    with fe_tab_c:
        _selection_tab(df_etf_fe, "C", ETF_COLS, col_cfg=_ETF_CFG,
                       empty_msg="ETF Universe non caricato.")

    with fe_tab_it:
        st.caption("Titoli FTSE MIB — prezzi storici via yfinance (ticker .MI)")
        df_it = pd.DataFrame([
            {"isin": isin, "ticker": d["ticker"], "nome": d["nome"],
             "settore": d["settore"]}
            for isin, d in ITALIAN_STOCKS.items()
            if not isin.startswith("IT_BTP")
        ])
        _pool_from_df(df_it)
        _selection_tab(df_it, "IT", IT_COLS, col_cfg=_IT_CFG,
                       empty_msg="Nessun titolo italiano.")

    with fe_tab_div:
        st.caption("Top 30 azioni non-USA — Fonte: TDIV/EUDV. Yield stimato, verificare su Yahoo Finance.")
        df_div = get_dividend_stocks_df()
        _pool_from_df(df_div)
        _fc1, _fc2 = st.columns(2)
        _fp = _fc1.selectbox("Paese", ["Tutti"]+sorted(df_div["paese"].unique().tolist()), key="dp_paese")
        _fs = _fc2.selectbox("Settore", ["Tutti"]+sorted(df_div["settore"].unique().tolist()), key="dp_sett")
        df_div_f = df_div.copy()
        if _fp != "Tutti": df_div_f = df_div_f[df_div_f["paese"] == _fp]
        if _fs != "Tutti": df_div_f = df_div_f[df_div_f["settore"] == _fs]
        _selection_tab(df_div_f.reset_index(drop=True), "DIV", DIV_COLS, col_cfg=_DIV_CFG)

    with fe_tab_isin:
        st.markdown("""
| Tipo | Esempi | Fonte |
|------|--------|-------|
| ETF (ISIN) | `IE00B4L5Y983` | yfinance ticker map |
| Azioni italiane | `ENI.MI`, `ISP.MI` | yfinance diretto |
| Azioni USA | `AAPL`, `MSFT`, `NVDA` | yfinance diretto |
| Azioni EU | `ADS.DE`, `ASML.AS` | yfinance diretto |
| Fondi (ISIN) | `LU0048578792` | Morningstar/FondiDoc |
| BTP proxy | `IBTS.MI` | yfinance diretto |
""")
        custom_raw = st.text_area("ISIN / Ticker (uno per riga)", height=100,
                                  key="fe_custom_raw",
                                  placeholder="ENI.MI\nAAPL\nNVDA\nIE00B4L5Y983")
        if st.button("➕ Aggiungi", key="add_custom", type="primary"):
            from utils.nav_fetcher import classify_asset_type
            added_info = []
            for line in custom_raw.strip().split("\n"):
                token = line.strip().upper()
                if not token:
                    continue
                if token not in st.session_state["fe_selected_isins"]:
                    st.session_state["fe_selected_isins"].append(token)
                if token not in _all_fund_pool:
                    if token in ITALIAN_STOCKS:
                        d = ITALIAN_STOCKS[token]
                        _pool_add(token, {"isin": token, "nome": d["nome"],
                                          "classificazione": f"Azione — {d['settore']}",
                                          "ticker": d["ticker"]})
                    elif token in _etf_nome_map:
                        _pool_add(token, {"isin": token, "nome": _etf_nome_map[token],
                                          "classificazione": "ETF",
                                          "ticker": ISIN_TO_TICKER.get(token, "")})
                    else:
                        _pool_add(token, {"isin": token, "nome": token,
                                          "classificazione": classify_asset_type(token)})
                added_info.append(f"**{token}** — {_all_fund_pool.get(token,{}).get('nome',token)}")
            if added_info:
                st.toast(f"✅ Aggiunti {len(added_info)} strumenti")
                for m in added_info: st.markdown(f"- {m}")
            st.rerun()

    # ── AUTO-SELEZIONE PER ASSET CLASS ─────────────────────────────────────
    st.markdown("---")
    with st.expander("🎯 Auto-composizione per Asset Class", expanded=False):
        st.markdown(
            "Indica la % target per macro asset class. "
            "L'app selezionerà i **migliori fondi per Score Qualità** (Liste A/B) "
            "e/o gli **ETF rappresentativi** (Lista C), poi ottimizzerà i pesi "
            "rispettando l'allocazione richiesta."
        )

        # ── Sorgente dati ─────────────────────────────────────────────────
        use_funds = st.checkbox(
            "Includi fondi (Liste A/B) oltre agli ETF",
            value=True,
            key="ac_use_funds",
            help="Se attivo usa i fondi per Score Qualità; se disattivo usa solo ETF Lista C",
        )
        prefer_funds = st.checkbox(
            "Preferisci fondi agli ETF (se disponibili)",
            value=True,
            key="ac_prefer_funds",
            disabled=not use_funds,
        )

        # ── Sliders allocazione ───────────────────────────────────────────
        ac1, ac2, ac3 = st.columns(3)
        pct_az  = ac1.slider("Azioni %",         0, 100, 60, 5, key="ac_az")
        pct_ob  = ac2.slider("Obbligazioni %",    0, 100, 30, 5, key="ac_ob")
        pct_mp  = ac3.slider("Materie Prime %",   0, 100, 10, 5, key="ac_mp")
        total_ac = pct_az + pct_ob + pct_mp
        if total_ac != 100:
            st.warning(f"Totale: {total_ac}% — deve essere 100%")
        else:
            st.success(f"✅ {pct_az}% Azioni + {pct_ob}% Obbligazioni + {pct_mp}% Materie Prime")

        n_az = ac1.number_input("N. strumenti azionari",      2, 8, 4, key="n_az")
        n_ob = ac2.number_input("N. strumenti obbligazionari", 2, 8, 4, key="n_ob")
        n_mp = ac3.number_input("N. strumenti commodity",      1, 4, 2, key="n_mp")

        # ── Logica di selezione ───────────────────────────────────────────
        # ETF fallback per asset class
        _ETF_FALLBACK = {
            "Azioni": [
                "IE00B4L5Y983",  # MSCI World
                "IE00BK5BQT80",  # FTSE All-World
                "IE00B5BMR087",  # S&P 500
                "LU0908500753",  # Stoxx 600
                "IE00BKM4GZ66",  # EM IMI
            ],
            "Obbligazioni": [
                "IE00B4WXJJ64",  # Euro Govt
                "IE00B3F81R35",  # EUR Corp
                "IE00B66F4759",  # EUR HY
                "IE00B2NPKV68",  # EM Bond
                "IE00B3T9LM79",  # BTP
            ],
            "Materie Prime": [
                "IE00BD6FTQ80",  # Bloomberg Commodity
                "LU1829218749",  # Commodity ex-Agri
                "GB00B15KXQ89",  # Copper
                "GB00B15KXV33",  # WTI Oil
            ],
        }

        # Mappa classificazione → macro-bucket con word-boundary (evita "azionari" in "obbligazionari")
        import re as _re
        _CLASS_TO_MACRO_ORDERED = [
            # Materie Prime prima (termini univoci)
            ("Materie Prime", ["materie prime", "commodity", "commodities",
                               "energy", "metals", "agriculture", "oro", "gold",
                               "petrolio", "oil", "gas"]),
            # Obbligazioni prima di Azioni (evita match "azionari" ⊂ "obbligazionari")
            ("Obbligazioni",  ["obbligazionari", "obbligazionario", "bond",
                               "fixed income", "reddito fisso", "high yield",
                               "corporate bond", "government bond", "duration",
                               "inflation linked", "monetario", "monetari",
                               "money market", "liquidit"]),
            # Azioni per ultima
            ("Azioni",        ["azionari", "azionario", "equity", "azioni", "stock",
                               "tematici", "tematico", "growth fund",
                               "small cap", "large cap", "dividend"]),
        ]

        def _macro_from_class(classificazione: str) -> str | None:
            cl = str(classificazione).lower()
            for macro, keywords in _CLASS_TO_MACRO_ORDERED:
                for kw in keywords:
                    # Word-boundary: il keyword non deve essere parte di un'altra parola
                    pattern = r"(?<![a-z])" + _re.escape(kw) + r"(?![a-z])"
                    if _re.search(pattern, cl):
                        return macro
            return None

        def _pick_assets(macro: str, n: int,
                          use_f: bool, prefer_f: bool) -> tuple[list, list]:
            """
            Ritorna (lista_isins, lista_macro_label) con i migliori N strumenti
            per il macro-bucket richiesto.
            Prima tenta con i fondi (se use_f), poi completa con ETF fallback.
            """
            selected = []
            labels   = []

            # ── FONDI da df_unified (Liste A/B) ──────────────────────────
            fund_candidates = []
            if use_f and not df_unified.empty:
                from utils.scoring import compute_scores_df
                _fu = df_unified.copy()
                _fu = compute_scores_df(_fu)
                _fu["_macro_auto"] = _fu["classificazione"].apply(_macro_from_class)
                _fu = _fu[_fu["_macro_auto"] == macro].copy()
                _fu = _fu.sort_values("score_qualita", ascending=False)
                # Diversificazione: max 1 per casa
                _seen_casa: set = set()
                for _, r in _fu.iterrows():
                    casa = str(r.get("casa", ""))
                    if casa and casa in _seen_casa:
                        continue
                    fund_candidates.append(r["isin"])
                    if casa:
                        _seen_casa.add(casa)
                    if len(fund_candidates) >= n * 3:
                        break

            # ── ETF fallback ──────────────────────────────────────────────
            etf_candidates = _ETF_FALLBACK.get(macro, [])

            if prefer_f and fund_candidates:
                # Priorità ai fondi, completa con ETF se non bastano
                selected = fund_candidates[:n]
                if len(selected) < n:
                    for e in etf_candidates:
                        if e not in selected and len(selected) < n:
                            selected.append(e)
            elif fund_candidates and not prefer_f:
                # Misto: metà fondi, metà ETF
                half = n // 2
                selected = fund_candidates[:half]
                for e in etf_candidates:
                    if e not in selected and len(selected) < n:
                        selected.append(e)
            else:
                # Solo ETF
                selected = etf_candidates[:n]

            labels = [macro] * len(selected)
            return selected, labels

        # ── Preview anteprima prima di confermare ──────────────────────────
        if total_ac == 100:
            _prev_az, _ = _pick_assets("Azioni", int(n_az), use_funds, prefer_funds)
            _prev_ob, _ = _pick_assets("Obbligazioni", int(n_ob), use_funds, prefer_funds)
            _prev_mp, _ = _pick_assets("Materie Prime", int(n_mp), use_funds, prefer_funds)
            _all_prev = _prev_az + _prev_ob + _prev_mp

            if _all_prev:
                st.markdown("**Anteprima selezione:**")
                _prev_rows = []
                for _isin in _all_prev:
                    _info = _all_fund_pool.get(_isin, {})
                    _nome = str(_info.get("nome", _isin))[:55]
                    _tipo = "ETF" if _isin in {e["isin"] for e in _ETF_STATIC_LIST} else "Fondo"
                    _macro_lbl = ("Azioni" if _isin in _prev_az
                                  else "Obbligazioni" if _isin in _prev_ob
                                  else "Materie Prime")
                    _prev_rows.append({
                        "Asset Class": _macro_lbl,
                        "Tipo": _tipo,
                        "ISIN": _isin,
                        "Nome": _nome,
                        "Score": round(_info.get("score_qualita", 0) or 0, 2),
                    })
                st.dataframe(
                    pd.DataFrame(_prev_rows),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Score Qualità", min_value=0, max_value=20, format="%.2f"),
                    },
                    height=min(400, len(_prev_rows) * 38 + 40),
                )

        if st.button("🎯 Aggiungi alla selezione e ottimizza",
                     key="auto_compose", type="primary",
                     disabled=(total_ac != 100)):
            az_list, az_lbl = _pick_assets("Azioni", int(n_az), use_funds, prefer_funds)
            ob_list, ob_lbl = _pick_assets("Obbligazioni", int(n_ob), use_funds, prefer_funds)
            mp_list, mp_lbl = _pick_assets("Materie Prime", int(n_mp), use_funds, prefer_funds)

            auto_isins = list(dict.fromkeys(az_list + ob_list + mp_list))

            # Aggiungi info nel pool per gli ISIN non ancora presenti
            for _isin in auto_isins:
                if _isin not in _all_fund_pool:
                    _pool_add(_isin, {"isin": _isin, "nome": _isin, "classificazione": ""})

            st.session_state["fe_selected_isins"] = list(dict.fromkeys(
                st.session_state["fe_selected_isins"] + auto_isins
            ))
            st.session_state["fe_ac_target"] = {
                "Azioni": pct_az / 100,
                "Obbligazioni": pct_ob / 100,
                "Materie Prime": pct_mp / 100,
            }
            st.session_state["fe_ac_map"] = (
                {i: "Azioni" for i in az_list} |
                {i: "Obbligazioni" for i in ob_list} |
                {i: "Materie Prime" for i in mp_list}
            )
            n_fondi = sum(1 for i in auto_isins
                          if i not in {e["isin"] for e in _ETF_STATIC_LIST})
            n_etf   = len(auto_isins) - n_fondi
            st.toast(f"✅ Selezionati {len(auto_isins)} strumenti "
                     f"({n_fondi} fondi + {n_etf} ETF)")
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
    v_col1, v_col2, v_col3, v_col4 = st.columns(4)
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
    # Fondi Azimut nella selezione corrente
    _azimut_in_sel = [
        i for i in sel_isins
        if _all_fund_pool.get(i, {}).get("_source") == "azimut"
        or _all_fund_pool.get(i, {}).get("casa","").lower() == "azimut"
    ]
    n_min_azimut = v_col4.number_input(
        f"Min fondi Azimut ({len(_azimut_in_sel)} disponibili)",
        min_value=0,
        max_value=max(len(_azimut_in_sel), 1),
        value=0,
        step=1,
        key="fe_n_min_azimut",
        help="Forza l'inclusione di almeno N fondi Azimut nel portafoglio ottimizzato",
        disabled=len(_azimut_in_sel) == 0,
    )
    # Aggiungi fondi Azimut a forced_include se n_min_azimut > 0
    if n_min_azimut > 0 and _azimut_in_sel:
        _az_forced = _azimut_in_sel[:int(n_min_azimut)]
        forced_include_sel = list(dict.fromkeys(
            (forced_include_sel or []) + _az_forced
        ))
        st.caption(f"🔒 Azimut forzati: {', '.join([_all_fund_pool.get(i,{}).get('nome',i)[:35] for i in _az_forced])}")

    # ── STEP 2b: BLACK-LITTERMAN ───────────────────────────────────────────
    with st.expander("🔮 Black-Litterman — aggiungi le tue view di mercato", expanded=False):
        st.markdown("""
**Cos'è Black-Litterman?**
Combina i rendimenti di equilibrio di mercato (prior) con le **tue aspettative personali**
su uno o più asset. Se pensi che l'azionario emergente renderà il 12% nei prossimi 3 anni,
puoi inserirlo — il modello bilanicia questa view con il mercato in base alla tua confidenza.

> 💡 **Quando usarlo**: hai una tesi su uno specifico asset.
> Se non hai view particolari, usa Max Sharpe o Min Varianza.
""")
        use_bl = st.checkbox("✅ Abilita Black-Litterman", key="use_bl_chk")
        bl_views: dict = {}
        bl_conf: dict = {}
        if use_bl:
            if not sel_isins:
                st.warning("Aggiungi prima gli strumenti nella sezione precedente.")
            else:
                st.markdown("---")
                st.markdown("**Inserisci le tue view** — spunta solo gli asset su cui hai un'opinione:")
                for isin in sel_isins[:15]:
                    info_bl = _all_fund_pool.get(isin, {})
                    nome_bl = str(info_bl.get("nome", isin))[:50]
                    class_bl = str(info_bl.get("classificazione", info_bl.get("categoria","")))[:30]

                    with st.container():
                        c1, c2, c3 = st.columns([3, 2, 2])
                        en = c1.checkbox(
                            f"**{nome_bl}**",
                            help=f"ISIN: {isin} | {class_bl}",
                            key=f"bl_en_{isin}",
                        )
                        if en:
                            bl_views[isin] = c2.number_input(
                                "Rendimento atteso (%)",
                                min_value=-30.0, max_value=60.0, value=8.0, step=0.5,
                                key=f"bl_r_{isin}",
                                help="La tua aspettativa di rendimento annuo per questo asset",
                            )
                            bl_conf[isin] = c3.slider(
                                "Confidenza",
                                min_value=0.1, max_value=1.0, value=0.5, step=0.1,
                                key=f"bl_c_{isin}",
                                help="1.0 = molto sicuro, 0.1 = incerto",
                            )
                if bl_views:
                    st.success(
                        f"✅ {len(bl_views)} view inserit{'a' if len(bl_views)==1 else 'e'}. "
                        "Il portafoglio Black-Litterman apparirà nei risultati dopo il calcolo."
                    )
                else:
                    st.info("Spunta almeno un asset per inserire una view.")

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
                    # Vincoli settoriali se attivati da auto-composizione
                    ac_map   = st.session_state.get("fe_ac_map", {})
                    ac_target= st.session_state.get("fe_ac_target", {})
                    sector_constraints = None
                    if ac_map and ac_target:
                        # Filtra solo asset che hanno dati
                        sc_mapper = {k: v for k, v in ac_map.items() if k in price_dict}
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
                    )
                if "error" in result:
                    st.error(f"Errore ottimizzazione: {result['error']}")
                else:
                    lbl = " (con vincoli asset class)" if sector_constraints else ""
                    st.success(f"Ottimizzazione completata su {len(price_dict)} strumenti{lbl}.")
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

        # ── Controllo qualità dati ───────────────────────────────────────
        _ms_ret = ms.get("ret", 0) if ms and "error" not in ms else 0
        _ms_vol = ms.get("vol", 0) if ms and "error" not in ms else 0
        _ms_sh  = ms.get("sharpe", 0) if ms and "error" not in ms else 0

        if abs(_ms_ret) < 0.02 or abs(_ms_sh) > 8:
            st.warning(
                "⚠️ **I risultati sembrano degeneri** "
                f"(Rendimento atteso: {_ms_ret*100:.2f}%, Sharpe: {_ms_sh:.2f}). "
                "Cause probabili:\n"
                "- Le performance dei fondi nell'Excel sono in formato decimale "
                "(es. 0.27 invece di 27%) e la cache non è stata aggiornata.\n"
                "**Soluzione:** clicca **🗑️ Svuota tutta la cache** nella sidebar, "
                "ricarica i file Excel e ricalcola."
            )

        # ── Grafico Frontiera Efficiente ─────────────────────────────────
        fig_fe = go.Figure()

        # Raccogli tutti i punti validi per calcolare range assi
        _all_vols = []
        _all_rets = []
        mc = result.get("monte_carlo", pd.DataFrame())
        frontier = result.get("frontier_df", pd.DataFrame())
        if not mc.empty:
            _all_vols += (mc["vol"]*100).tolist()
            _all_rets += (mc["ret"]*100).tolist()
        if not frontier.empty:
            _all_vols += (frontier["vol"]*100).tolist()
            _all_rets += (frontier["ret"]*100).tolist()

        # Range assi con padding
        def _axis_range(vals: list, pad_pct=0.15):
            if not vals: return [0, 1]
            lo, hi = min(vals), max(vals)
            span = max(hi - lo, 0.1)
            return [round(lo - span*pad_pct, 2), round(hi + span*pad_pct, 2)]

        x_range = _axis_range(_all_vols)
        y_range = _axis_range(_all_rets)

        # 1. Nuvola Monte Carlo — colorata per Sharpe, trasparente
        if not mc.empty:
            fig_fe.add_trace(go.Scatter(
                x=(mc["vol"]*100).round(2),
                y=(mc["ret"]*100).round(2),
                mode="markers",
                marker=dict(
                    color=mc["sharpe"].round(3),
                    colorscale=[
                        [0.0, "#FEF9C3"], [0.3, "#FED76A"],
                        [0.6, "#F97316"], [1.0, "#DC2626"],
                    ],
                    size=5, opacity=0.45,
                    colorbar=dict(
                        title=dict(text="Sharpe", font=dict(size=11)),
                        thickness=14, len=0.55, x=1.02,
                        tickfont=dict(size=10),
                    ),
                    showscale=True,
                ),
                name="Portafogli simulati (MC)",
                hovertemplate=(
                    "Volatilità: <b>%{x:.2f}%</b><br>"
                    "Rendimento: <b>%{y:.2f}%</b><br>"
                    "Sharpe: <b>%{marker.color:.2f}</b>"
                    "<extra>Monte Carlo</extra>"
                ),
            ))

        # 2. Curva frontiera efficiente
        if not frontier.empty:
            fig_fe.add_trace(go.Scatter(
                x=(frontier["vol"]*100).round(2),
                y=(frontier["ret"]*100).round(2),
                mode="lines",
                line=dict(color=NAVY, width=3, dash="solid"),
                name="Frontiera Efficiente",
                hovertemplate=(
                    "Frontiera | Vol: <b>%{x:.2f}%</b> · Rend: <b>%{y:.2f}%</b>"
                    "<extra></extra>"
                ),
            ))

        # 3. Area ombreggiata sotto la frontiera (zona sub-ottimale)
        if not frontier.empty and len(frontier) > 2:
            fig_fe.add_trace(go.Scatter(
                x=(frontier["vol"]*100).round(2).tolist() +
                  [x_range[0], x_range[0]],
                y=(frontier["ret"]*100).round(2).tolist() +
                  [(frontier["ret"]*100).min(), (frontier["ret"]*100).min()],
                fill="toself",
                fillcolor=f"rgba(26,44,84,0.06)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))

        # 4. Portafogli ottimali con annotazioni
        _pf_configs = [
            (ms,                             "Max Sharpe",      "#DC2626", "star",    24, "top right"),
            (result.get("min_variance", {}), "Min Varianza",    "#1D4ED8", "diamond", 20, "bottom right"),
            (bl_result or {},                "Black-Litterman", "#16A34A", "pentagon",20, "top left"),
        ]
        _annotations = []
        for pdata, pname, pcolor, psym, psz, _pos in _pf_configs:
            if not pdata or "error" in pdata or not pdata.get("vol"):
                continue
            vx = round(pdata["vol"] * 100, 2)
            vy = round(pdata["ret"] * 100, 2)
            sh = round(pdata.get("sharpe", 0), 3)

            fig_fe.add_trace(go.Scatter(
                x=[vx], y=[vy],
                mode="markers",
                marker=dict(
                    color=pcolor, size=psz, symbol=psym,
                    line=dict(color="white", width=2),
                ),
                name=f"{pname}  ·  Sharpe {sh:.2f}",
                hovertemplate=(
                    f"<b>{pname}</b><br>"
                    "Volatilità: <b>%{x:.2f}%</b><br>"
                    "Rendimento: <b>%{y:.2f}%</b><br>"
                    f"Sharpe: <b>{sh}</b>"
                    "<extra></extra>"
                ),
            ))

            # Annotazione con riquadro
            ay_off = 45 if "top" in _pos else -45
            ax_off = 60 if "right" in _pos else -60
            _annotations.append(dict(
                x=vx, y=vy,
                ax=ax_off, ay=ay_off,
                xref="x", yref="y", axref="x", ayref="y",
                # usa offset pixel
                text=(
                    f"<b style='color:{pcolor}'>{pname}</b><br>"
                    f"Rend: <b>{vy:.1f}%</b> | Vol: <b>{vx:.1f}%</b><br>"
                    f"Sharpe: <b>{sh:.2f}</b>"
                ),
                showarrow=True,
                arrowhead=2, arrowsize=1.2, arrowwidth=1.8,
                arrowcolor=pcolor,
                font=dict(size=10.5),
                bgcolor="white",
                bordercolor=pcolor,
                borderwidth=1.5,
                borderpad=5,
                opacity=0.95,
            ))

        # Layout professionale
        fig_fe.update_layout(
            title=dict(
                text=(
                    "<b>Frontiera Efficiente</b>  "
                    "<span style='font-size:13px;color:#666'>— Rischio vs Rendimento atteso</span>"
                ),
                font=dict(size=17, color=NAVY),
                x=0.01,
            ),
            xaxis=dict(
                title=dict(text="Volatilità annua (%)", font=dict(size=12)),
                tickformat=".1f",
                ticksuffix="%",
                gridcolor="#E9EEF4",
                showgrid=True,
                zeroline=False,
                range=x_range,
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title=dict(text="Rendimento atteso annuo (%)", font=dict(size=12)),
                tickformat=".1f",
                ticksuffix="%",
                gridcolor="#E9EEF4",
                showgrid=True,
                zeroline=False,
                range=y_range,
                tickfont=dict(size=11),
            ),
            plot_bgcolor="#FAFBFD",
            paper_bgcolor="white",
            height=580,
            annotations=_annotations,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.20,
                xanchor="center", x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#DDD", borderwidth=1,
                font=dict(size=11),
            ),
            margin=dict(t=70, b=110, l=70, r=80),
            hoverlabel=dict(
                bgcolor="white", bordercolor="#CCC",
                font=dict(size=12),
            ),
        )
        st.plotly_chart(fig_fe, use_container_width=True, config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "toImageButtonOptions": {"format": "png", "width": 1200, "height": 700},
        })

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
            # Converti grafico frontiera in PNG per il PDF
            _fe_png = None
            try:
                from utils.exporter import plotly_to_png
                _fe_png = plotly_to_png(fig_fe)
            except Exception:
                pass
            pdf_bytes = export_portfolio_pdf(
                ms["weights"], metrics,
                title="Portafoglio Max Sharpe",
                chart_bytes=_fe_png,
                fund_df=df_unified if not df_unified.empty else None,
                fund_pool=_all_fund_pool,
            )
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
        # Fondi Azimut disponibili in df_unified
        _n_azimut_avail = int((df_unified.get("_source", pd.Series(dtype=str)) == "azimut").sum()) \
            if not df_unified.empty and "_source" in df_unified.columns else 0
        n_min_az_pq = st.number_input(
            f"Min fondi Azimut ({_n_azimut_avail} disponibili)",
            min_value=0, max_value=max(_n_azimut_avail, 1),
            value=0, step=1, key="pq_n_min_azimut",
            help="Garantisce almeno N fondi Azimut nel portafoglio finale",
            disabled=_n_azimut_avail == 0,
        )

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
| **Lista A — Generalisti** | Fondi terzi + Azimut con classificazione globale/bilanciata/flessibile | File Excel caricato in sidebar |
| **Lista B — Tematici** | Fondi specializzati (emergenti, settoriali, tematici, high yield…) | File Excel caricato in sidebar |
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
**Lista A — Generalisti:**
- Eleggibilità: classificazione FIDA generalista + perf 3Y disponibile + rating ≥ 3★ (o n/d) + perf 3Y ≥ 0%
- **Max 3 fondi** per casa di gestione
- **Max 2 fondi** per sottoclassificazione FIDA
- **Max 1 fondo** per "radice strategia" (prime 3 parole significative del nome, escludendo share class come ACC/MINC)
- Copertura obbligatoria di almeno 5 macro-aree: Azionario globale, Obbligazionario globale, Bilanciato, Ritorno assoluto, Flessibile

**Lista B — Tematici:**
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
