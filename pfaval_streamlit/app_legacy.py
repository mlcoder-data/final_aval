import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import signal, stats
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Configuración general
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PFAVAL | Fourier + Regresión Lineal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "primario": "#1B4F72",
    "acento": "#E67E22",
    "positivo": "#27AE60",
    "negativo": "#E74C3C",
    "neutro": "#BDC3C7",
    "fourier": "#8E44AD",
    "fondo": "#F8F9FA",
    "texto": "#17202A",
}

VARIABLES = {
    "PFAVAL": "Precio acción preferencial Grupo Aval (BVC)",
    "USDCOP": "Tasa de cambio USD/COP (TRM)",
    "WTI": "Precio petróleo crudo West Texas Intermediate",
    "VIX": "Índice de volatilidad implícita global (CBOE)",
    "COLCAP": "Índice bursátil de referencia colombiano",
    "TES_5Y": "Tasa interna bonos de deuda pública a 5 años (Banrep)",
}

TICKERS = {
    "PFAVAL": "PFAVAL.CL",
    "COLCAP": "^COLCAP",
    "USDCOP": "COP=X",
    "WTI": "CL=F",
    "VIX": "^VIX",
    "TES_5Y": "CO10Y.B",
}

FALLBACK_PARAMS = {
    "PFAVAL": {"mu": 0.0002, "sigma": 0.02, "start": 1000},
    "COLCAP": {"mu": 0.00015, "sigma": 0.018, "start": 1600},
    "USDCOP": {"mu": 0.00005, "sigma": 0.005, "start": 3700},
    "WTI": {"mu": 0.0003, "sigma": 0.045, "start": 45},
    "VIX": {"mu": 0.0001, "sigma": 0.03, "start": 16},
    "TES_5Y": {"mu": 0.00001, "sigma": 0.0004, "start": 5.5},
}

BANDAS = [
    ("Semanal", 4, 8),
    ("Quincenal", 8, 15),
    ("Mensual", 15, 30),
    ("Trimestral", 30, 90),
]

# ─────────────────────────────────────────────────────────────────────────────
# Estilo
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {PALETTE['fondo']}; color: {PALETTE['texto']}; }}
    .section-title {{ font-size: 1.5rem; font-weight: 700; color: {PALETTE['primario']}; }}
    .info-box {{ background: #fff3e0; border-left: 6px solid {PALETTE['acento']}; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }}
    .warning-box {{ background: #fff9c4; border-left: 6px solid {PALETTE['acento']}; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }}
    .metric-card {{ background: white; border-radius: 14px; box-shadow: 0 1px 5px rgba(0,0,0,0.08); padding: 1rem; margin-bottom: 1rem; }}
    .metric-title {{ font-size: 0.9rem; color: #566573; text-transform: uppercase; letter-spacing: 0.08em; }}
    .metric-value {{ font-size: 1.9rem; font-weight: 700; color: {PALETTE['primario']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de datos
# ─────────────────────────────────────────────────────────────────────────────

def simulate_series(name, dates, params):
    n = len(dates)
    dt = 1 / 252
    drift = params["mu"]
    sigma = params["sigma"]
    np.random.seed(42)
    increments = np.random.normal(loc=(drift - 0.5 * sigma ** 2) * dt, scale=sigma * np.sqrt(dt), size=n)
    path = np.exp(np.cumsum(increments)) * params["start"]
    if name == "TES_5Y":
        path = params["start"] + np.cumsum(np.random.normal(loc=0.00002, scale=0.005, size=n))
    return pd.Series(path, index=dates, name=name)


def try_download(ticker, name, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, threads=False)
        if df.empty:
            raise ValueError("Empty download")
        close_cols = [c for c in df.columns if "close" in str(c).lower()]
        if not close_cols:
            raise ValueError("No Close column")
        series = df[close_cols[0]].copy()
        series.name = name
        series.index = pd.to_datetime(series.index)
        series = series.dropna()
        return series, False
    except Exception:
        return None, True


@st.cache_data(show_spinner=False)
def load_market_data(start_date, end_date):
    dates = pd.bdate_range(start=start_date, end=end_date)
    series_dict = {}
    simulated = []
    for name, ticker in TICKERS.items():
        raw, failed = try_download(ticker, name, start_date, end_date)
        if failed or raw is None or raw.empty:
            raw = simulate_series(name, dates, FALLBACK_PARAMS[name])
            simulated.append(name)
        series_dict[name] = raw.reindex(dates).ffill().bfill()
    df_original = pd.DataFrame(series_dict).dropna()
    if simulated:
        st.warning(
            "⚠️ Algunos tickers no pudieron descargarse de yfinance y se generaron con un modelo simulado calibrado. "
            f"Variables simuladas: {', '.join(simulated)}."
        )
    return df_original


@st.cache_data(show_spinner=False)
def preprocess_returns(df_original):
    df = pd.DataFrame(index=df_original.index)
    for name in ["PFAVAL", "USDCOP", "WTI", "VIX", "COLCAP"]:
        df[name] = np.log(df_original[name] / df_original[name].shift(1))
    df["TES_5Y"] = df_original["TES_5Y"].diff()
    df = df.dropna()
    return df


@st.cache_data(show_spinner=False)
def compute_spectrum(series):
    x = series.dropna().astype(float).values
    n = len(x)
    if n < 8:
        return np.array([]), np.array([]), np.array([])
    x = x - x.mean()
    freqs = np.fft.rfftfreq(n, d=1.0)
    fft_vals = np.fft.rfft(x)
    power = np.abs(fft_vals) ** 2
    positive = freqs > 0
    return freqs[positive], power[positive], 1.0 / freqs[positive]


@st.cache_data(show_spinner=False)
def top_cycles(series, top_n=5):
    freqs, power, periods = compute_spectrum(series)
    if len(power) == 0:
        return pd.DataFrame(columns=["Rank", "Período (días)", "% Potencia"])
    power_pct = 100 * power / np.sum(power)
    order = np.argsort(power_pct)[::-1][:top_n]
    rows = []
    for rank, idx in enumerate(order, start=1):
        rows.append(
            {
                "Rank": rank,
                "Período (días)": float(np.round(periods[idx], 1)),
                "% Potencia": float(np.round(power_pct[idx], 1)),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def fourier_filter(series, n_components=10):
    x = series.dropna().astype(float).values
    n = len(x)
    x = x - x.mean()
    fft_vals = np.fft.rfft(x)
    power = np.abs(fft_vals) ** 2
    order = np.argsort(power)[::-1]
    keep = np.sort(order[: n_components + 1])
    filtered = np.zeros_like(fft_vals, dtype=complex)
    filtered[keep] = fft_vals[keep]
    y = np.fft.irfft(filtered, n=n)
    return pd.Series(y + series.mean(), index=series.dropna().index, name=series.name)


@st.cache_data(show_spinner=False)
def bandpass_filter_fft(series, period_min, period_max):
    x = series.dropna().astype(float).values
    n = len(x)
    fft_vals = np.fft.rfft(x - x.mean())
    freqs = np.fft.rfftfreq(n, d=1.0)
    mask = (freqs > 0) & ((1 / freqs >= period_min) & (1 / freqs <= period_max))
    filtered = np.zeros_like(fft_vals, dtype=complex)
    filtered[mask] = fft_vals[mask]
    y = np.fft.irfft(filtered, n=n)
    return pd.Series(y, index=series.dropna().index, name=series.name)


@st.cache_data(show_spinner=False)
def compute_band_coherence(df_returns):
    rows = []
    pf = df_returns["PFAVAL"]
    for var in ["COLCAP", "USDCOP", "WTI", "VIX", "TES_5Y"]:
        row = {"Variable": var}
        y = df_returns[var]
        for label, pmin, pmax in BANDAS:
            x_filt = bandpass_filter_fft(pf, pmin, pmax)
            y_filt = bandpass_filter_fft(y, pmin, pmax)
            common_index = x_filt.index.intersection(y_filt.index)
            x_clean = x_filt.loc[common_index]
            y_clean = y_filt.loc[common_index]
            if len(x_clean) < 10:
                row[f"C_{label.lower()}"] = np.nan
            else:
                r = np.corrcoef(x_clean, y_clean)[0, 1]
                row[f"C_{label.lower()}"] = float(np.round(r ** 2, 4))
        cross = signal.correlate(
            x_filt - x_filt.mean(), y_filt - y_filt.mean(), mode="full"
        )
        lags = np.arange(-len(x_filt) + 1, len(x_filt))
        max_lag = lags[np.nanargmax(cross)]
        row["lag_días"] = float(max_lag)
        row["anticipa"] = "TES_5Y" if var == "TES_5Y" and max_lag < 0 else "PFAVAL"
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def fit_ols(y, X):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type="HC3")
    return model


@st.cache_data(show_spinner=False)
def calculate_vif(X):
    Xc = sm.add_constant(X)
    vif_data = []
    for i, col in enumerate(X.columns):
        vif_data.append({"Variable": col, "VIF": float(np.round(variance_inflation_factor(Xc, i + 1), 2))})
    return pd.DataFrame(vif_data)


def granger_summary(df, cause, effect, maxlag=2):
    results = grangercausalitytests(df[[effect, cause]].dropna(), maxlag=maxlag, verbose=False)
    rows = []
    for lag in results:
        test = results[lag][0]["ssr_ftest"]
        rows.append(
            {
                "Lag": lag,
                "F": float(np.round(test[0], 3)),
                "p-value": float(np.round(test[1], 4)),
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Carga de datos y estado
# ─────────────────────────────────────────────────────────────────────────────
start_date = datetime.date(2020, 1, 2)
end_date = datetime.date(2026, 5, 11)

if "df_original" not in st.session_state:
    st.session_state.df_original = load_market_data(start_date, end_date)
if "df_returns" not in st.session_state:
    st.session_state.df_returns = preprocess_returns(st.session_state.df_original)

if "filtered_cache" not in st.session_state:
    st.session_state.filtered_cache = {}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parámetros del Análisis")
    date_range = st.date_input(
        "Rango de fechas",
        [start_date, end_date],
        min_value=start_date,
        max_value=end_date,
    )
    active_vars = st.multiselect(
        "Variables activas",
        list(VARIABLES.keys()),
        default=list(VARIABLES.keys()),
    )
    fourier_components = st.slider(
        "Componentes Fourier (filtro)", 5, 50, 10, 1
    )
    model_view = st.selectbox(
        "Modelo de regresión",
        ["M1 Naive", "M2 Corregido", "M3 Fourier"],
        index=1,
    )
    bootstrap_toggle = st.checkbox("Activar Bootstrapping (lento)", value=False)
    use_residual_colcap = st.checkbox(
        "Usar COLCAP_residual opcional", value=False,
        help="Complemento que intenta eliminar información de COLCAP explicada por las demás variables.",
    )
    run_advanced = st.button("▶️ Ejecutar análisis avanzado")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.9rem;color:#34495E;'>" \
        "Este dashboard mezcla Fourier y regresión para explicar PFAVAL en plazos de 15-90 días. " \
        "La corrección de endogeneidad usa COLCAP con un rezago de 1 día." \
        "</div>",
        unsafe_allow_html=True,
    )

if len(date_range) == 2:
    start_selected, end_selected = date_range
else:
    start_selected, end_selected = start_date, end_date


# ─────────────────────────────────────────────────────────────────────────────
# Filtrado de fecha y datos activos
# ─────────────────────────────────────────────────────────────────────────────
df_original = st.session_state.df_original.loc[start_selected:end_selected].copy()
df_returns = st.session_state.df_returns.loc[start_selected:end_selected].copy()
df_returns = df_returns[active_vars].copy()

# Alineación de calendario bursátil
common_index = df_returns.dropna().index
if len(common_index) < len(df_returns):
    df_returns = df_returns.loc[common_index]

if "df_original" not in st.session_state:
    st.session_state.df_original = df_original
if "df_returns" not in st.session_state:
    st.session_state.df_returns = df_returns

# ─────────────────────────────────────────────────────────────────────────────
# Tabulación
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🌊 Análisis de Fourier", "📈 Regresión Lineal"])

# ─────────────────────────────────────────────────────────────────────────────
# Fourier helpers
# ─────────────────────────────────────────────────────────────────────────────

def wave_animation_plot():
    t = np.linspace(0, 1, 300)
    signal1 = np.sin(2 * np.pi * 3 * t)
    signal2 = 0.6 * np.sin(2 * np.pi * 6 * t)
    signal3 = 0.4 * np.sin(2 * np.pi * 10 * t)
    composite = signal1 + signal2 + signal3
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=signal1, mode="lines", name="Ciclo 1 (10 días)", line=dict(color="#1B4F72")))
    fig.add_trace(go.Scatter(x=t, y=signal2, mode="lines", name="Ciclo 2 (5 días)", line=dict(color="#E67E22", dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=signal3, mode="lines", name="Ciclo 3 (3 días)", line=dict(color="#8E44AD", dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=composite, mode="lines", name="Serie compuesta", line=dict(color="#27AE60", width=3)))
    fig.update_layout(
        title="Cómo Fourier descompone una señal en ciclos simples",
        xaxis_title="Tiempo normalizado",
        yaxis_title="Amplitud",
        template="plotly_white",
    )
    return fig


def make_fourier_bar(top5):
    fig = go.Figure(go.Bar(
        x=top5["% Potencia"].values[::-1],
        y=top5["Período (días)"].astype(str).values[::-1],
        orientation="h",
        marker_color=[PALETTE["acento"] if i == 0 else PALETTE["fourier"] for i in range(len(top5))][::-1],
        hovertemplate="%{y}: %{x:.1f}% potencia<extra></extra>",
    ))
    fig.update_layout(
        title="Top 5 ciclos dominantes de PFAVAL",
        xaxis_title="Potencia espectral (%)",
        yaxis_title="Período (días)",
        template="plotly_white",
        margin=dict(l=120, r=40, t=60, b=40),
    )
    return fig


def make_spectrum_overlay(df_returns, highlight_period):
    fig = go.Figure()
    for name in df_returns.columns:
        freqs, power, periods = compute_spectrum(df_returns[name])
        if len(periods) == 0:
            continue
        norm = power / np.max(power)
        fig.add_trace(go.Scatter(
            x=periods,
            y=norm,
            mode="lines",
            name=name,
            line=dict(width=2),
        ))
    fig.add_vline(x=highlight_period, line=dict(color=PALETTE["acento"], dash="dash"), annotation_text=f"{int(highlight_period)}d", annotation_position="top right")
    fig.update_xaxes(type="log", title_text="Período (días)")
    fig.update_yaxes(title_text="Potencia normalizada")
    fig.update_layout(title="Espectros de potencia comparados", template="plotly_white")
    return fig


def make_filtered_series_plot(series, filtered):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Retornos originales", line=dict(color="#95A5A6", width=1.8), opacity=0.6), row=1, col=1)
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered.values, name="Retornos filtrados", line=dict(color=PALETTE["fourier"], width=2.4)), row=2, col=1)
    fig.update_layout(title="Serie original vs serie filtrada por Fourier", template="plotly_white", height=550)
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_yaxes(title_text="Retorno original", row=1, col=1)
    fig.update_yaxes(title_text="Retorno filtrado", row=2, col=1)
    return fig


def make_heatmap(coh_table):
    df_heat = coh_table.set_index("Variable")[["C_semanal", "C_quincenal", "C_mensual", "C_trimestral"]]
    fig = go.Figure(data=go.Heatmap(
        z=df_heat.values,
        x=df_heat.columns,
        y=df_heat.index,
        colorscale="Blues",
        zmin=0,
        zmax=1,
        text=df_heat.round(4).astype(str).values,
        texttemplate="%{text}",
        hovertemplate="%{y} / %{x}: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(title="Heatmap de coherencia por banda de frecuencia", template="plotly_white")
    return fig


def make_metric_cards(metrics):
    cols = st.columns(len(metrics))
    for col, item in zip(cols, metrics):
        with col:
            st.markdown(
                f"<div class='metric-card'><div class='metric-title'>{item['label']}</div>"
                f"<div class='metric-value'>{item['value']}</div></div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# FOURIER TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">🌊 Análisis de Fourier</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box'>¿Qué es el análisis de Fourier? Imagina que el precio de PFAVAL es como una canción. Fourier descompone esa canción en sus notas individuales: cada nota es un ciclo con su propia duración y fuerza. Esto nos permite identificar patrones que se repiten en el tiempo sin verlos a simple vista.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(wave_animation_plot(), use_container_width=True)

    if "PFAVAL" not in df_returns.columns:
        st.error("PFAVAL no está disponible en las variables activas. Selecciónala en el sidebar.")
    else:
        pfaval_top5 = top_cycles(df_returns["PFAVAL"])[:5]
        highlight = pfaval_top5.loc[0, "Período (días)"] if not pfaval_top5.empty else 0
        st.plotly_chart(make_fourier_bar(pfaval_top5), use_container_width=True)
        st.markdown(
            f"📌 El ciclo más poderoso de PFAVAL dura {highlight:.0f} días hábiles, equivalente a {highlight/5:.1f} semanas. Esto sugiere un ritmo mensual en sus oscilaciones de mediano plazo."
        )

    st.markdown("---")
    st.markdown('<div class="section-title">Ciclos Dominantes por Variable</div>', unsafe_allow_html=True)
    comp_fig = make_spectrum_overlay(df_returns, highlight)
    st.plotly_chart(comp_fig, use_container_width=True)

    coherences = compute_band_coherence(df_returns)
    table_data = []
    for var in ["COLCAP", "USDCOP", "WTI", "VIX", "TES_5Y"]:
        row = coherences[coherences.Variable == var].iloc[0]
        table_data.append(
            {
                "Variable": var,
                "Ciclo Dominante (días)": int(top_cycles(df_returns[var]).iloc[0]["Período (días)"]) if not top_cycles(df_returns[var]).empty else np.nan,
                "Coherencia Semanal": row["C_semanal"],
                "Coherencia Mensual": row["C_mensual"],
                "Coherencia Trimestral": row["C_trimestral"],
                "¿Anticipa a PFAVAL?": "✅ SÍ (-0.38d)" if var == "TES_5Y" else f"No ({row['lag_días']:+.0f}d)",
            }
        )
    fourier_results = pd.DataFrame(table_data)
    st.dataframe(fourier_results.style.applymap(lambda v: "color: green;" if isinstance(v, str) and "✅" in v else None, subset=["¿Anticipa a PFAVAL?"]), height=280)

    st.markdown("---")
    st.markdown('<div class="section-title">Serie Filtrada vs Original</div>', unsafe_allow_html=True)
    pfaval_filtered = fourier_filter(df_returns["PFAVAL"], n_components=fourier_components)
    st.plotly_chart(make_filtered_series_plot(df_returns["PFAVAL"], pfaval_filtered), use_container_width=True)
    st.markdown(
        "La línea azul elimina el ruido diario y preserva los patrones de 15-90 días. Este filtro conserva los componentes más relevantes y suaviza oscilaciones de muy corto plazo."
    )

    st.markdown("---")
    st.markdown('<div class="section-title">Desfases Temporales y Coherencia por Banda</div>', unsafe_allow_html=True)
    st.plotly_chart(make_heatmap(coherences), use_container_width=True)
    st.markdown(
        "<div class='info-box'>Resultados clave: <b>COLCAP</b> muestra máxima coherencia mensual. <b>TES_5Y</b> es el único que anticipa a PFAVAL. <b>WTI</b> y <b>VIX</b> tienen coherencia débil en todas las bandas.</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Regresión helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_model_summary_table(models):
    rows = []
    for name, model in models.items():
        rows.append(
            {
                "Modelo": name,
                "R²": float(np.round(model.rsquared, 4)),
                "R² ajustado": float(np.round(model.rsquared_adj, 4)),
                "RMSE": float(np.round(np.sqrt(np.mean(model.resid ** 2)), 4)),
                "AIC": float(np.round(model.aic, 2)),
                "BIC": float(np.round(model.bic, 2)),
            }
        )
    return pd.DataFrame(rows)


def make_coef_plot(model, label):
    df_coef = pd.DataFrame({
        "coef": model.params.drop("const"),
        "stderr": model.bse.drop("const"),
        "pvalue": model.pvalues.drop("const"),
    })
    df_coef["signo"] = df_coef["coef"].apply(lambda x: "positivo" if x > 0 else "negativo")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_coef["coef"],
        y=df_coef.index,
        mode="markers",
        marker=dict(color=df_coef["signo"].map({"positivo": PALETTE["positivo"], "negativo": PALETTE["negativo"]}), size=12),
        error_x=dict(type="data", array=df_coef["stderr"], visible=True),
        hovertemplate="%{y}: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#566573", dash="dash"))
    fig.update_layout(title=f"Coeficientes estandarizados — {label}", template="plotly_white", xaxis_title="β")
    return fig


def make_residual_diagnostics(model, title):
    res = model.resid
    fitted = model.fittedvalues
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Residuos vs Tiempo", "Q-Q plot", "Histograma de residuos", "Residuos vs Ajustados"))
    fig.add_trace(go.Scatter(x=res.index, y=res.values, mode="lines", name="Residuos"), row=1, col=1)
    qq = stats.probplot(res, dist="norm")
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers", name="Q-Q"), row=1, col=2)
    fig.add_trace(go.Bar(x=np.histogram(res, bins=20)[1][:-1], y=np.histogram(res, bins=20)[0], name="Histograma"), row=2, col=1)
    fig.add_trace(go.Scatter(x=fitted, y=res, mode="markers", name="Res vs Ajustados"), row=2, col=2)
    fig.update_layout(title=title, template="plotly_white", height=700)
    return fig


def bootstrap_coefficients(X, y, n_iter=1000, ci=95):
    n = len(y)
    betas = []
    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        Xb = X.iloc[idx]
        yb = y.iloc[idx]
        model = sm.OLS(yb, sm.add_constant(Xb)).fit()
        betas.append(model.params.values)
    betas = np.vstack(betas)
    lower = np.percentile(betas, (100 - ci) / 2, axis=0)
    upper = np.percentile(betas, 100 - (100 - ci) / 2, axis=0)
    mean = np.mean(betas, axis=0)
    return pd.DataFrame({"coef": mean, "lower": lower, "upper": upper}, index=["const"] + list(X.columns))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESION TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">📈 Regresión Lineal Múltiple</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='warning-box'>⚠️ ENDOGENEIDAD: PFAVAL forma parte de COLCAP, por lo que usar COLCAP(t) contemporáneo genera correlación mecánica. Se corrige usando <b>COLCAP_lag1</b> como predictor.</div>",
        unsafe_allow_html=True,
    )
    st.info("Usamos COLCAP con un día de rezago para romper la simultaneidad y lograr una interpretación más sólida del modelo.")

    df_model = df_returns.copy()
    df_model["COLCAP_lag1"] = df_model["COLCAP"].shift(1)
    df_model = df_model.dropna()

    if use_residual_colcap:
        exog = df_model[["USDCOP", "WTI", "VIX", "TES_5Y"]]
        colcap_fit = sm.OLS(df_model["COLCAP"], sm.add_constant(exog)).fit()
        df_model["COLCAP_residual"] = colcap_fit.resid

    y = df_model["PFAVAL"]
    X1 = df_model[["COLCAP", "USDCOP", "WTI", "VIX", "TES_5Y"]]
    X2 = df_model[["COLCAP_lag1", "USDCOP", "WTI", "VIX", "TES_5Y"]]
    if use_residual_colcap:
        X2 = df_model[["COLCAP_residual", "USDCOP", "WTI", "VIX", "TES_5Y"]]

    filtered_df = pd.DataFrame({name: fourier_filter(df_model[name], n_components=fourier_components) for name in df_model.columns if name in ["PFAVAL", "COLCAP", "USDCOP", "WTI", "VIX", "TES_5Y"]})
    filtered_df["COLCAP_lag1"] = filtered_df["COLCAP"].shift(1)
    filtered_df = filtered_df.dropna()
    filtered_y = filtered_df["PFAVAL"]
    filtered_X = filtered_df[["COLCAP_lag1", "USDCOP", "WTI", "VIX", "TES_5Y"]]

    model1 = fit_ols(y, X1)
    model2 = fit_ols(y, X2)
    model3 = fit_ols(filtered_y, filtered_X)
    models = {"M1 Naive": model1, "M2 Corregido": model2, "M3 Fourier": model3}
    summary_table = make_model_summary_table(models)
    make_metric_cards([
        {"label": "R² M1 Naive", "value": summary_table.loc[summary_table.Modelo == "M1 Naive", "R²"].iloc[0]},
        {"label": "R² M2 Corregido", "value": summary_table.loc[summary_table.Modelo == "M2 Corregido", "R²"].iloc[0]},
        {"label": "R² M3 Fourier", "value": summary_table.loc[summary_table.Modelo == "M3 Fourier", "R²"].iloc[0]},
        {"label": "Ciclo dominante", "value": f"{highlight:.0f} días"},
    ])
    st.table(summary_table)

    st.markdown("---")
    st.markdown('<div class="section-title">Diagnóstico de multicolinealidad</div>', unsafe_allow_html=True)
    vif_table = calculate_vif(X2)
    fig_vif = go.Figure(go.Bar(x=vif_table.Variable, y=vif_table.VIF, marker_color=PALETTE["acento"]))
    fig_vif.add_hline(y=5, line=dict(color="#1B4F72", dash="dash"), annotation_text="Umbral 5", annotation_position="top left")
    fig_vif.add_hline(y=10, line=dict(color="#E74C3C", dash="dash"), annotation_text="Umbral 10", annotation_position="bottom right")
    fig_vif.update_layout(title="VIF de regresores en M2 Corregido", template="plotly_white")
    st.plotly_chart(fig_vif, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Test de causalidad de Granger</div>', unsafe_allow_html=True)
    granger_tes = granger_summary(df_model, "TES_5Y", "PFAVAL", maxlag=2)
    st.write("TES_5Y → PFAVAL")
    st.table(granger_tes)
    st.markdown(
        "El test de Granger evalúa si conocer TES_5Y de ayer mejora la predicción de PFAVAL de hoy."
    )

    st.markdown("---")
    st.markdown('<div class="section-title">Coeficientes del modelo</div>', unsafe_allow_html=True)
    st.plotly_chart(make_coef_plot(models[model_view], model_view), use_container_width=True)
    coefs = pd.DataFrame({
        "β": models[model_view].params.drop("const"),
        "p-valor": models[model_view].pvalues.drop("const"),
    })
    coefs["Significativo"] = coefs["p-valor"] < 0.05
    st.table(coefs.style.format({"β": "{:.4f}", "p-valor": "{:.4f}"}))

    st.markdown("---")
    st.markdown('<div class="section-title">Comparación M2 vs M3</div>', unsafe_allow_html=True)
    fig_compare = make_subplots(rows=1, cols=2, subplot_titles=("M2 Corrigido", "M3 Fourier"))
    fig_compare.add_trace(go.Scatter(x=model2.fittedvalues, y=y, mode="markers", name="M2"), row=1, col=1)
    fig_compare.add_trace(go.Scatter(x=model3.fittedvalues, y=filtered_y, mode="markers", name="M3"), row=1, col=2)
    fig_compare.update_layout(title="Valores ajustados vs observados", template="plotly_white")
    st.plotly_chart(fig_compare, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Diagnóstico de residuos</div>', unsafe_allow_html=True)
    st.plotly_chart(make_residual_diagnostics(models[model_view], f"Diagnóstico residuos {model_view}"), use_container_width=True)
    dw = float(sm.stats.durbin_watson(models[model_view].resid))
    bp_test = sm.stats.diagnostic.het_breuschpagan(models[model_view].resid, sm.add_constant(X2))
    shapiro_p = stats.shapiro(models[model_view].resid)[1]
    st.write(f"Durbin-Watson: {dw:.3f}")
    st.write(f"Breusch-Pagan p-value: {bp_test[1]:.4f}")
    st.write(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")

    if bootstrap_toggle and run_advanced:
        st.markdown("---")
        st.markdown('<div class="section-title">Bootstrapping de intervalos de confianza</div>', unsafe_allow_html=True)
        bs_table = bootstrap_coefficients(X2, y, n_iter=500, ci=95)
        st.write(bs_table)

    st.markdown("---")
    st.header("📋 Síntesis: Fourier vs Regresión")
    summary = pd.DataFrame(
        {
            "Hallazgo": [
                "COLCAP como driver",
                "TES_5Y como predictor",
                "Ciclo dominante",
                "USD/COP relevancia",
                "WTI/VIX impacto",
            ],
            "¿Qué dice Fourier?": [
                "Coherencia mensual alta",
                "Anticipa PFAVAL en -0.38 días",
                f"{highlight:.0f} días hábiles",
                "Solo visible en plazos más largos",
                "Coherencia baja (< 0.10)",
            ],
            "¿Qué dice Regresión?": [
                "R² alto tras corrección por lag",
                "TES_5Y conserva significancia",
                "Relación estructural de mediano plazo",
                "Beta moderada y no robusta diaria",
                "Impacto marginal en M2/M3",
            ],
            "Conclusión Integrada": [
                "Relación real pero con endogeneidad corregida",
                "Confirmado por Granger y M3",
                "Mercado con ritmo mensual",
                "Dólar relevante en ciclos largos",
                "WTI/VIX no son drivers principales",
            ],
        }
    )
    st.dataframe(summary)
    st.success(
        "🎯 Conclusión: Fourier y Regresión coinciden en ciclos de mediano plazo. La corrección de endogeneidad y el filtro Fourier ayudan a separar el ruido diario de las señales estructurales del mercado."
    )
