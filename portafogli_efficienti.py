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
    st.markdown("Ottimizzazione quantitativa con PyPortfolioOpt — Max Sharpe, Min Varianza, Black-Litterman.")

    # --- Selezione Asset ---
    with st.expander("1️⃣ Selezione Asset", expanded=True):
        sources = st.multiselect(
            "Sorgenti",
            ["Lista A (Generalisti)", "Lista B (Tematici)", "Lista C (ETF)", "ISIN liberi"],
            default=["Lista A (Generalisti)", "Lista C (ETF)"],
        )

        all_options = {}
        if "Lista A (Generalisti)" in sources and not lista_a.empty:
            for _, r in lista_a.iterrows():
                label = f"[A] {r['isin']} — {r.get('nome','')[:50]}"
                all_options[label] = r.to_dict()
        if "Lista B (Tematici)" in sources and not lista_b.empty:
            for _, r in lista_b.iterrows():
                label = f"[B] {r['isin']} — {r.get('nome','')[:50]}"
                all_options[label] = r.to_dict()
        if "Lista C (ETF)" in sources:
            try:
                df_etf = _load_etf_universe_cached()
                for _, r in df_etf.iterrows():
                    label = f"[C] {r.get('isin','')} — {r.get('nome','')[:50]}"
                    all_options[label] = r.to_dict()
            except Exception:
                pass
        if "ISIN liberi" in sources:
            custom_isins = st.text_area(
                "ISIN aggiuntivi (uno per riga)",
                height=80,
                key="fe_custom_isins",
            )
            if custom_isins:
                for isin in custom_isins.strip().split("\n"):
                    isin = isin.strip()
                    if isin:
                        all_options[f"[Custom] {isin}"] = {"isin": isin}

        selected_labels = st.multiselect(
            "Seleziona asset per il portafoglio (min 3, max 30)",
            options=list(all_options.keys()),
            default=list(all_options.keys())[:5] if len(all_options) >= 5 else list(all_options.keys()),
            max_selections=30,
        )
        selected_assets = [all_options[l] for l in selected_labels]

    # --- Vincoli ---
    with st.expander("2️⃣ Vincoli di peso"):
        col1, col2 = st.columns(2)
        min_w = col1.slider("Peso minimo per asset (%)", 0, 20,
                             st.session_state["min_weight"]) / 100
        max_w = col2.slider("Peso massimo per asset (%)", 10, 100,
                             st.session_state["max_weight"]) / 100
        forced_include_labels = st.multiselect(
            "Forza inclusione (peso min 5%)",
            options=selected_labels,
            key="fe_forced_include",
        )
        forced_include = [
            all_options[l]["isin"] for l in forced_include_labels
            if all_options.get(l, {}).get("isin")
        ]

    # --- Black-Litterman (collassato) ---
    with st.expander("3️⃣ Black-Litterman (opzionale)"):
        use_bl = st.checkbox("Abilita Black-Litterman")
        bl_views = {}
        bl_conf = {}
        if use_bl and selected_assets:
            st.markdown("Inserisci view assolute per gli asset:")
            for asset in selected_assets[:10]:
                isin = asset.get("isin", "")
                nome = asset.get("nome", isin)[:40]
                cols = st.columns([3, 2, 2])
                abilita = cols[0].checkbox(f"{nome}", key=f"bl_en_{isin}")
                if abilita:
                    view_ret = cols[1].number_input(
                        "Rendimento atteso (%)", -20.0, 50.0, 5.0,
                        key=f"bl_ret_{isin}"
                    )
                    conf = cols[2].slider(
                        "Confidenza", 0.1, 1.0, 0.5,
                        key=f"bl_conf_{isin}"
                    )
                    bl_views[isin] = view_ret
                    bl_conf[isin] = conf

    # --- RUN ---
    run_col, _ = st.columns([1, 3])
    run = run_col.button("🚀 Calcola Frontiera", type="primary", use_container_width=True)

    if run:
        if len(selected_assets) < 3:
            st.error("Seleziona almeno 3 asset.")
        else:
            with st.spinner("Recupero dati storici..."):
                period_map = {"1Y": "1y", "3Y": "3y", "5Y": "5y"}
                period = period_map.get(st.session_state["opt_period"], "3y")

                asset_list = []
                for a in selected_assets:
                    isin = a.get("isin", "")
                    asset_list.append({
                        "isin": isin,
                        "ticker": a.get("ticker"),
                        "perf_1y": a.get("perf_1y") or a.get("PERF 1Y"),
                        "perf_3y": a.get("perf_3y") or a.get("PERF 3Y"),
                        "perf_ytd": a.get("perf_ytd"),
                        "perf_2022": a.get("perf_2022") or a.get("perf_2022"),
                        "perf_2023": a.get("perf_2023"),
                        "perf_2024": a.get("perf_2024"),
                    })
                price_dict = get_multiple_nav(asset_list, period=period)

            if len(price_dict) < 3:
                st.error(f"Dati insufficienti: solo {len(price_dict)}/{len(selected_assets)} serie recuperate.")
            else:
                with st.spinner("Ottimizzazione in corso..."):
                    rfr = st.session_state["risk_free_rate"] / 100
                    result = compute_efficient_frontier(
                        price_dict,
                        weight_bounds=(min_w, max_w),
                        risk_free_rate=rfr,
                        forced_include=forced_include or None,
                    )

                if "error" in result:
                    st.error(f"Errore ottimizzazione: {result['error']}")
                else:
                    st.success(f"Ottimizzazione completata su {len(price_dict)} asset.")
                    st.session_state["fe_result"] = result
                    st.session_state["fe_price_dict"] = price_dict
                    st.session_state["fe_assets"] = selected_assets

                    # BL
                    bl_result = None
                    if use_bl and bl_views:
                        with st.spinner("Black-Litterman..."):
                            bl_result = compute_black_litterman(
                                price_dict, bl_views, bl_conf,
                                weight_bounds=(min_w, max_w),
                                risk_free_rate=rfr,
                            )
                        st.session_state["bl_result"] = bl_result

    # --- Visualizzazione risultati ---
    if "fe_result" in st.session_state:
        result = st.session_state["fe_result"]
        price_dict = st.session_state.get("fe_price_dict", {})
        bl_result = st.session_state.get("bl_result")

        st.markdown("---")
        st.subheader("Risultati Ottimizzazione")

        # Metriche principali
        m_col = st.columns(4)
        if "max_sharpe" in result and "error" not in result["max_sharpe"]:
            ms = result["max_sharpe"]
            m_col[0].metric("Max Sharpe — Rendimento", f"{ms['ret']*100:.2f}%")
            m_col[1].metric("Max Sharpe — Volatilità", f"{ms['vol']*100:.2f}%")
            m_col[2].metric("Max Sharpe — Sharpe", f"{ms['sharpe']:.3f}")
            if price_dict:
                mdd = estimate_max_drawdown(ms["weights"], price_dict)
                m_col[3].metric("Max Drawdown stimato", f"{mdd:.2f}%")

        # --- Grafico Frontiera + Monte Carlo ---
        fig = go.Figure()

        mc = result.get("monte_carlo", pd.DataFrame())
        if not mc.empty:
            fig.add_trace(go.Scatter(
                x=mc["vol"] * 100,
                y=mc["ret"] * 100,
                mode="markers",
                marker=dict(
                    color=mc["sharpe"],
                    colorscale="Viridis",
                    size=4,
                    opacity=0.5,
                    colorbar=dict(title="Sharpe"),
                ),
                name="Monte Carlo",
                hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>",
            ))

        frontier = result.get("frontier_df", pd.DataFrame())
        if not frontier.empty:
            fig.add_trace(go.Scatter(
                x=frontier["vol"] * 100,
                y=frontier["ret"] * 100,
                mode="lines",
                line=dict(color=NAVY, width=2),
                name="Frontiera Efficiente",
            ))

        # Punti ottimali
        def _add_star(data: dict, name: str, color: str, symbol: str):
            if data and "error" not in data:
                fig.add_trace(go.Scatter(
                    x=[data["vol"] * 100],
                    y=[data["ret"] * 100],
                    mode="markers+text",
                    marker=dict(color=color, size=16, symbol="star"),
                    text=[name],
                    textposition="top center",
                    name=name,
                ))

        _add_star(result.get("max_sharpe"), "Max Sharpe", "red", "star")
        _add_star(result.get("min_variance"), "Min Varianza", "blue", "star")
        if bl_result and "error" not in bl_result:
            _add_star(bl_result, "Black-Litterman", "green", "star")

        fig.update_layout(
            title="Frontiera Efficiente",
            xaxis_title="Volatilità (%)",
            yaxis_title="Rendimento atteso (%)",
            height=500,
            template="plotly_white",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Pesi portafogli ottimali ---
        tab_ms, tab_mv, tab_bl_t = st.tabs(["Max Sharpe", "Min Varianza", "Black-Litterman"])

        def _render_weights_tab(portfolio: dict, label: str):
            if not portfolio or "error" in portfolio:
                st.info(f"Portafoglio {label} non disponibile.")
                return
            w = portfolio["weights"]
            non_zero = {k: v for k, v in w.items() if v > 0.001}
            col1, col2 = st.columns([1, 1])
            with col1:
                w_df = pd.DataFrame([
                    {"ISIN": k, "Peso (%)": round(v * 100, 2)}
                    for k, v in sorted(non_zero.items(), key=lambda x: -x[1])
                ])
                st.dataframe(w_df, use_container_width=True, hide_index=True)
            with col2:
                pie = px.pie(
                    values=list(non_zero.values()),
                    names=list(non_zero.keys()),
                    hole=0.3,
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                )
                pie.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(pie, use_container_width=True)

            # Metriche
            st.markdown(
                f"**Rendimento atteso:** {portfolio.get('ret', 0)*100:.2f}% &nbsp;|&nbsp; "
                f"**Volatilità:** {portfolio.get('vol', 0)*100:.2f}% &nbsp;|&nbsp; "
                f"**Sharpe:** {portfolio.get('sharpe', 0):.3f}"
            )

        with tab_ms:
            _render_weights_tab(result.get("max_sharpe", {}), "Max Sharpe")
        with tab_mv:
            _render_weights_tab(result.get("min_variance", {}), "Min Varianza")
        with tab_bl_t:
            _render_weights_tab(bl_result or {}, "Black-Litterman")

        # --- Correlazioni ---
        if price_dict:
            with st.expander("📊 Matrice Correlazioni"):
                prices_df = pd.DataFrame(price_dict).pct_change().dropna()
                corr = prices_df.corr()
                fig_corr = px.imshow(
                    corr,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    text_auto=".2f",
                    title="Correlazioni storiche",
                )
                fig_corr.update_layout(height=450)
                st.plotly_chart(fig_corr, use_container_width=True)

        # --- BL dettaglio ---
        if bl_result and "error" not in bl_result and "bl_returns" in bl_result:
            with st.expander("📐 Black-Litterman — Rendimenti Posteriori vs Prior"):
                bl_rets = bl_result["bl_returns"]
                mu_prior = result.get("mu", {})
                compare = pd.DataFrame({
                    "ISIN": list(bl_rets.keys()),
                    "Prior (%)": [mu_prior.get(k, 0) * 100 for k in bl_rets],
                    "Posteriore BL (%)": [v * 100 for v in bl_rets.values()],
                })
                fig_bl = go.Figure()
                fig_bl.add_bar(x=compare["ISIN"], y=compare["Prior (%)"], name="Prior", marker_color="steelblue")
                fig_bl.add_bar(x=compare["ISIN"], y=compare["Posteriore BL (%)"], name="BL", marker_color="coral")
                fig_bl.update_layout(barmode="group", height=350)
                st.plotly_chart(fig_bl, use_container_width=True)

        # --- Export ---
        st.markdown("---")
        exp_col1, exp_col2 = st.columns(2)
        ms_data = result.get("max_sharpe", {})
        if ms_data and "error" not in ms_data:
            metrics = {
                "Rendimento atteso (%)": f"{ms_data.get('ret', 0)*100:.2f}",
                "Volatilità (%)": f"{ms_data.get('vol', 0)*100:.2f}",
                "Sharpe Ratio": f"{ms_data.get('sharpe', 0):.3f}",
                "Generato": datetime.now().strftime("%d/%m/%Y %H:%M"),
            }
            excel_bytes = export_portfolio_excel(
                ms_data["weights"], metrics,
                fund_df=df_unified if not df_unified.empty else None,
                price_dict=price_dict,
                title="Max Sharpe",
            )
            exp_col1.download_button(
                "📥 Esporta Excel (Max Sharpe)",
                data=excel_bytes,
                file_name=f"portafoglio_max_sharpe_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            pdf_bytes = export_portfolio_pdf(ms_data["weights"], metrics, title="Portafoglio Max Sharpe")
            if pdf_bytes:
                exp_col2.download_button(
                    "📄 Esporta PDF (Max Sharpe)",
                    data=pdf_bytes,
                    file_name=f"portafoglio_max_sharpe_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                )


# ===========================================================================
# PORTAFOGLIO QUALITÀ
# ===========================================================================
elif nav == "⭐ Portafoglio Qualità":
    st.title("⭐ Portafoglio Qualità")
    st.markdown("Costruzione per Score Qualità con vincoli di diversificazione.")

    col1, col2 = st.columns([1, 2])
    with col1:
        profilo = st.selectbox(
            "Profilo di rischio",
            list(PROFILI.keys()),
            index=list(PROFILI.keys()).index(st.session_state["profilo"]),
        )
        st.session_state["profilo"] = profilo
        fondi_per_bucket = st.slider("Fondi per bucket", 2, 8,
                                      st.session_state["fondi_per_bucket"])
        st.session_state["fondi_per_bucket"] = fondi_per_bucket

    with col2:
        st.markdown(f"**Allocazioni target — {profilo}**")
        alloc = PROFILI[profilo].copy()
        alloc_adj = {}
        cols_alloc = st.columns(len(alloc))
        for i, (bucket, pct) in enumerate(alloc.items()):
            alloc_adj[bucket] = cols_alloc[i].number_input(
                f"{bucket} (%)", 0, 100, pct, 5, key=f"alloc_{bucket}"
            )
        total = sum(alloc_adj.values())
        if total != 100:
            st.warning(f"Totale allocazioni: {total}% (deve essere 100%)")

    st.markdown("---")

    # Costruisci portafoglio
    with st.spinner("Costruzione portafoglio..."):
        portfolio_buckets = build_portfolio_quality(
            df_unified, profilo=profilo, fondi_per_bucket=fondi_per_bucket
        )

    # Lock / replace in sessione
    locked = st.session_state.get("locked_funds", set())

    # Visualizzazione bucket per bucket
    all_porto_rows = []
    for bucket, df_bucket in portfolio_buckets.items():
        st.subheader(f"📂 {bucket} — {alloc_adj.get(bucket, 0)}%")
        if df_bucket.empty:
            st.info(f"Nessun fondo trovato per {bucket}.")
            continue

        # Applica sostituzioni manuali
        # (semplificato: mostra classifica + bottone lock)
        cols_show = ["isin", "nome", "classificazione", "perf_1y", "perf_3y",
                     "volatilita", "rating_fida", "score_qualita", "_peso_fondo"]
        cols_show = [c for c in cols_show if c in df_bucket.columns]
        display = df_bucket[cols_show].copy()

        col_config = {
            "isin": st.column_config.TextColumn("ISIN"),
            "nome": st.column_config.TextColumn("Fondo"),
            "classificazione": st.column_config.TextColumn("Classificazione"),
            "perf_1y": st.column_config.NumberColumn("Perf 1Y %", format="%.2f"),
            "perf_3y": st.column_config.NumberColumn("Perf 3Y %", format="%.2f"),
            "volatilita": st.column_config.NumberColumn("Vol %", format="%.2f"),
            "rating_fida": st.column_config.NumberColumn("★ FIDA", format="%d"),
            "score_qualita": st.column_config.NumberColumn("Score", format="%.3f"),
            "_peso_fondo": st.column_config.NumberColumn("Peso %", format="%.1f"),
        }
        st.dataframe(display, column_config=col_config, use_container_width=True, hide_index=True)

        # Score chart
        if "score_qualita" in df_bucket.columns and "nome" in df_bucket.columns:
            fig_score = px.bar(
                df_bucket.sort_values("score_qualita"),
                x="score_qualita",
                y=df_bucket["nome"].str[:30],
                orientation="h",
                color="score_qualita",
                color_continuous_scale="Blues",
                title=f"Score Qualità — {bucket}",
            )
            fig_score.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0),
                                     showlegend=False)
            st.plotly_chart(fig_score, use_container_width=True)

        for _, r in df_bucket.iterrows():
            all_porto_rows.append({
                "Bucket": bucket,
                "ISIN": r.get("isin", ""),
                "Fondo": r.get("nome", "")[:50],
                "Classificazione": r.get("classificazione", ""),
                "Perf 1Y %": r.get("perf_1y"),
                "Perf 3Y %": r.get("perf_3y"),
                "Volatilità %": r.get("volatilita"),
                "★ FIDA": r.get("rating_fida"),
                "Score": r.get("score_qualita"),
                "Peso %": r.get("_peso_fondo"),
                "Retrocessione %": r.get("retrocessione"),
            })

    # Summary portafoglio completo
    if all_porto_rows:
        st.markdown("---")
        st.subheader("📋 Portafoglio Completo")
        df_porto = pd.DataFrame(all_porto_rows)

        # Grafico torta macro
        bucket_weights = df_porto.groupby("Bucket")["Peso %"].sum().reset_index()
        fig_pie = px.pie(
            bucket_weights,
            values="Peso %",
            names="Bucket",
            hole=0.35,
            color_discrete_sequence=[NAVY, "#2E5090", "#4472C4", "#7AB0E0", "#A9CCE3"],
            title="Allocazione per Macro Asset Class",
        )
        fig_pie.update_layout(height=350)
        col_pie, col_tbl = st.columns([1, 2])
        col_pie.plotly_chart(fig_pie, use_container_width=True)
        col_tbl.dataframe(df_porto[["Bucket","Fondo","Peso %","Score","★ FIDA"]],
                           use_container_width=True, hide_index=True, height=350)

        # Heatmap correlazioni (se disponibili serie storiche)
        with st.expander("📊 Heatmap Correlazioni (serie sintetiche)"):
            isins = df_porto["ISIN"].tolist()
            asset_list_q = [
                {"isin": r["ISIN"],
                 "perf_1y": r.get("Perf 1Y %"), "perf_3y": r.get("Perf 3Y %"),
                 "perf_2022": None, "perf_2023": None, "perf_2024": None}
                for _, r in df_porto.iterrows()
            ]
            with st.spinner("Recupero serie storiche..."):
                pd_dict = get_multiple_nav(asset_list_q, period="3y")
            if len(pd_dict) >= 2:
                prices_q = pd.DataFrame(pd_dict).pct_change().dropna()
                corr_q = prices_q.corr()
                fig_corr_q = px.imshow(
                    corr_q, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, text_auto=".2f",
                )
                fig_corr_q.update_layout(height=400)
                st.plotly_chart(fig_corr_q, use_container_width=True)
            else:
                st.info("Dati storici insufficienti per la heatmap.")

        # Export
        st.markdown("---")
        exp_c1, exp_c2 = st.columns(2)
        weights_q = {r["ISIN"]: (r["Peso %"] or 0) / 100 for _, r in df_porto.iterrows()}
        metrics_q = {
            "Profilo": profilo,
            "Totale fondi": len(df_porto),
            "Score medio": f"{df_porto['Score'].mean():.3f}",
            "Generato": datetime.now().strftime("%d/%m/%Y %H:%M"),
        }
        excel_q = export_portfolio_excel(weights_q, metrics_q,
                                          fund_df=df_unified,
                                          title=f"Portafoglio Qualità {profilo}")
        exp_c1.download_button(
            "📥 Esporta Excel",
            data=excel_q,
            file_name=f"portafoglio_qualita_{profilo.lower()}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        pdf_q = export_portfolio_pdf(weights_q, metrics_q,
                                      title=f"Portafoglio Qualità — {profilo}")
        if pdf_q:
            exp_c2.download_button(
                "📄 Esporta PDF",
                data=pdf_q,
                file_name=f"portafoglio_qualita_{profilo.lower()}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
            )


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

        cols_etf = [c for c in ["isin", "nome", "categoria", "ter", "aum_mln", "perf_1y", "perf_3y", "perf_5y"]
                    if c in df_display.columns]
        col_config_etf = {
            "isin": st.column_config.TextColumn("ISIN"),
            "nome": st.column_config.TextColumn("Nome", width="large"),
            "categoria": st.column_config.TextColumn("Categoria"),
            "ter": st.column_config.NumberColumn("TER %", format="%.2f"),
            "aum_mln": st.column_config.NumberColumn("AUM (mln €)", format="%.0f"),
            "perf_1y": st.column_config.NumberColumn("Perf 1Y %", format="%.2f"),
            "perf_3y": st.column_config.NumberColumn("Perf 3Y %", format="%.2f"),
            "perf_5y": st.column_config.NumberColumn("Perf 5Y %", format="%.2f"),
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
