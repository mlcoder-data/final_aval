import numpy as np
import pandas as pd
import scipy.signal as signal
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT = Path(r"d:\ITM-------WORKSPACE\IA_code\Fourier_Analisis\pfaval_study")
PATH_UNIFIED = DATA_ROOT / "data" / "processed" / "df_unified.parquet"
PATH_RETURNS = DATA_ROOT / "data" / "processed" / "df_returns.parquet"
PATH_FILTERED = DATA_ROOT / "data" / "processed" / "df_returns_filtered.parquet"
PATH_COHERENCE = DATA_ROOT / "outputs" / "tables" / "coherencia_bandas.csv"

PALETTE = {
    "bg": "#080d1a",
    "surface": "#0d1b2e",
    "border": "#1e3a5f",
    "accent1": "#00c8ff",
    "accent2": "#f97316",
    "accent3": "#a78bfa",
    "accent4": "#22c55e",
    "accent5": "#f43f5e",
    "text": "#e2f0ff",
    "muted": "#6a8faa",
}

VAR_COLORS = {
    "PFAVAL": "steelblue",
    "USDCOP": "crimson",
    "WTI": "darkorange",
    "VIX": "purple",
    "COLCAP": "teal",
    "TES_5Y": "goldenrod",
}

VAR_LABELS = {
    "PFAVAL": "PFAVAL",
    "USDCOP": "USDCOP",
    "WTI": "WTI",
    "VIX": "VIX",
    "COLCAP": "COLCAP",
    "TES_5Y": "TES_5Y",
}

BANDAS = [
    ("Semanal", 4, 8),
    ("Quincenal", 8, 15),
    ("Mensual", 15, 30),
    ("Trimestral", 30, 90),
]

st.set_page_config(
    page_title="Análisis Espectral PFAVAL",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS de diseño tomado de la referencia
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
* {{ font-family: 'DM Sans', sans-serif; box-sizing: border-box; }}
.stApp {{ background-color: {PALETTE['bg']}; color: {PALETTE['text']}; }}
div[data-testid="stSidebarContent"] {{
    background: linear-gradient(180deg, #04091a 0%, #080d1a 100%);
    border-right: 1px solid {PALETTE['border']};
}}

.hero {{ margin-bottom: 0.5rem; }}
.hero-badge {{
    display: inline-block;
    background: rgba(0,200,255,0.1);
    border: 1px solid rgba(0,200,255,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.9rem;
    color: {PALETTE['accent1']};
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}}
.hero-title {{
    font-size: 2.6rem;
    font-weight: 700;
    color: {PALETTE['text']};
    line-height: 1.15;
    margin: 0;
}}
.hero-title span {{ color: {PALETTE['accent1']}; }}
.hero-sub {{ color: {PALETTE['muted']}; font-size: 0.95rem; margin-top: 0.3rem; }}

.kpi {{
    background: linear-gradient(135deg, {PALETTE['surface']}, #112240);
    border: 1px solid {PALETTE['border']};
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}}
.kpi-val {{
    font-family: 'DM Mono', monospace;
    font-size: 1.55rem;
    font-weight: 500;
    color: {PALETTE['accent1']};
    line-height: 1.1;
}}
.kpi-lbl {{ font-size: 0.72rem; color: {PALETTE['muted']}; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.2rem; }}
.kpi-delta-p {{ color: {PALETTE['accent4']}; font-size: 0.82rem; font-weight: 600; }}
.kpi-delta-n {{ color: {PALETTE['accent5']}; font-size: 0.82rem; font-weight: 600; }}

.sec-hdr {{
    font-size: 1.25rem;
    font-weight: 600;
    color: {PALETTE['text']};
    border-left: 3px solid {PALETTE['accent1']};
    padding-left: 0.7rem;
    margin: 1.6rem 0 0.7rem 0;
}}
.sec-sub {{ color: {PALETTE['muted']}; font-size: 0.88rem; margin: -0.4rem 0 1rem 1rem; }}

.insight {{
    background: #0a1628;
    border: 1px solid {PALETTE['border']};
    border-top: 3px solid {PALETTE['accent1']};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #c5ddf0;
    font-size: 0.88rem;
    line-height: 1.65;
}}
.insight b {{ color: {PALETTE['text']}; }}

.cb-pos {{ display:inline-block; background:rgba(34,197,94,0.12); border:1px solid #22c55e;
           border-radius:20px; padding:0.2rem 0.7rem; color:#22c55e; font-weight:600; font-size:0.95rem; }}
.cb-neg {{ display:inline-block; background:rgba(244,63,94,0.12); border:1px solid #f43f5e;
           border-radius:20px; padding:0.2rem 0.7rem; color:#f43f5e; font-weight:600; font-size:0.95rem; }}
.cb-mid {{ display:inline-block; background:rgba(249,115,22,0.12); border:1px solid #f97316;
           border-radius:20px; padding:0.2rem 0.7rem; color:#f97316; font-weight:600; font-size:0.95rem; }}

eq-box {{
    background: #050c1a;
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 1.3rem 2rem;
    text-align: center;
    font-family: 'DM Mono', monospace;
    color: {PALETTE['accent1']};
    font-size: 1.05rem;
    line-height: 2;
    margin: 0.5rem 0;
}}
.eq-box small {{ color: {PALETTE['muted']}; font-family: 'DM Sans', sans-serif; font-size: 0.8rem; }}

.data-note {{
    background: rgba(167,139,250,0.08);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    color: #b8a8e0;
    font-size: 0.82rem;
    margin-bottom: 0.5rem;
}}
hr {{ border-color: {PALETTE['border']}; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df_unified = pd.read_parquet(PATH_UNIFIED)
    df_returns = pd.read_parquet(PATH_RETURNS)
    df_filtered = pd.read_parquet(PATH_FILTERED)
    df_coherence = pd.read_csv(PATH_COHERENCE)
    return df_unified, df_returns, df_filtered, df_coherence

@st.cache_data(show_spinner=False)
def compute_spectrum(series):
    series = np.asarray(series.dropna(), dtype=float)
    n = len(series)
    signal_centered = series - np.mean(series)
    freqs = np.fft.rfftfreq(n, d=1.0)
    fft_vals = np.fft.rfft(signal_centered)
    power = np.abs(fft_vals) ** 2
    return freqs, power

@st.cache_data(show_spinner=False)
def top_cycles(series, top_n=5):
    freqs, power = compute_spectrum(series)
    if len(freqs) < 2:
        return pd.DataFrame(columns=["Ranking", "Período (días)", "% Poder Espectral"])
    periods = 1.0 / freqs[1:]
    power = power[1:]
    power_pct = 100 * power / np.sum(power)
    rank = np.argsort(power)[::-1]
    selected = rank[:top_n]
    rows = []
    for idx, k in enumerate(selected, start=1):
        rows.append({
            "Ranking": idx,
            "Período (días)": float(np.round(periods[k], 2)),
            "% Poder Espectral": float(np.round(power_pct[k], 2)),
        })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def compute_coherence_matrix(reference, target):
    reference = np.asarray(reference.dropna(), dtype=float)
    target = np.asarray(target.dropna(), dtype=float)
    if len(reference) != len(target):
        n = min(len(reference), len(target))
        reference = reference[:n]
        target = target[:n]
    n = len(reference)
    nperseg = min(512, n)
    noverlap = max(0, nperseg // 2)
    freqs, coh = signal.coherence(reference, target, fs=1.0, nperseg=nperseg, noverlap=noverlap)
    _, Pxy = signal.csd(reference, target, fs=1.0, nperseg=nperseg, noverlap=noverlap)
    periods = np.zeros_like(freqs)
    lags = np.zeros_like(freqs)
    valid = freqs > 0
    periods[valid] = 1.0 / freqs[valid]
    lags[valid] = -np.angle(Pxy[valid]) / (2 * np.pi * freqs[valid])
    return pd.DataFrame({
        "Periodo": periods,
        "Coherencia": coh,
        "Lag_dias": lags,
    })

@st.cache_data(show_spinner=False)
def create_descriptive_stats(df):
    stats = df.agg(["mean", "std", "min", "max"]).T.reset_index()
    stats.columns = ["Variable", "Media", "Std", "Mínimo", "Máximo"]
    stats["Media"] = stats["Media"].round(2)
    stats["Std"] = stats["Std"].round(2)
    stats["Mínimo"] = stats["Mínimo"].round(2)
    stats["Máximo"] = stats["Máximo"].round(2)
    return stats

@st.cache_data(show_spinner=False)
def format_plotly_layout(fig):
    fig.update_layout(
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor=PALETTE["bg"],
        font_color=PALETTE["text"],
        legend=dict(bgcolor=PALETTE["surface"], bordercolor=PALETTE["border"], borderwidth=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(gridcolor="#112240", zerolinecolor="#112240", showline=False)
    fig.update_yaxes(gridcolor="#112240", zerolinecolor="#112240", showline=False)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Datos cargados
# ─────────────────────────────────────────────────────────────────────────────
df_unified, df_returns, df_filtered, df_coherence = load_data()

observaciones_totales = len(df_unified)
observaciones_fourier = len(df_returns)
fecha_inicio = df_unified.index.min().strftime("%d/%m/%Y")
fecha_final = df_unified.index.max().strftime("%d/%m/%Y")

# ─────────────────────────────────────────────────────────────────────────────
# Barra lateral y navegación
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='color:{PALETTE['accent1']};font-weight:700;font-size:1.1rem;'>📌 Navegación</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Selecciona una página",
        [
            "Datos del Estudio",
            "Transformación de Series",
            "Análisis Fourier Individual",
            "Coherencia Espectral Cruzada",
            "Síntesis y Conclusiones",
        ],
    )
    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.88rem;color:{PALETTE['muted']};'>Ruta de datos:<br>{PATH_UNIFIED}</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Helpers de visualización
# ─────────────────────────────────────────────────────────────────────────────
def plot_original_series(variable):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_unified.index,
        y=df_unified[variable],
        name=variable,
        line=dict(color=VAR_COLORS[variable], width=2.2),
    ))
    fig.update_layout(
        title=f"Serie original — {variable}",
        xaxis_title="Fecha",
        yaxis_title="Valor",
    )
    return format_plotly_layout(fig)


def plot_dual_axis(variable):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_unified.index,
        y=df_unified[variable],
        name=f"Original {variable}",
        line=dict(color="#9ca3af", width=1.8),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=df_returns.index,
        y=df_returns[variable],
        name=f"Retorno {variable}",
        line=dict(color=VAR_COLORS[variable], width=1.8),
        yaxis="y2",
    ))
    fig.update_layout(
        title=f"Serie original y retorno diario — {variable}",
        xaxis_title="Fecha",
        yaxis=dict(title="Original", titlefont=dict(color="#9ca3af"), tickfont=dict(color="#9ca3af"), anchor="x"),
        yaxis2=dict(title="Retorno diario", titlefont=dict(color=VAR_COLORS[variable]), tickfont=dict(color=VAR_COLORS[variable]), overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.2, x=0.03),
        margin=dict(l=40, r=60, t=60, b=40),
    )
    return format_plotly_layout(fig)


def plot_return_comparison(variable):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_returns.index,
        y=df_returns[variable],
        name="Retorno crudo",
        line=dict(color="rgba(180,180,180,0.4)", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=df_filtered.index,
        y=df_filtered[variable],
        name="Retorno filtrado",
        line=dict(color=VAR_COLORS[variable], width=2.0),
    ))
    fig.update_layout(
        title=f"Retorno crudo vs filtrado — {variable}",
        xaxis_title="Fecha",
        yaxis_title="Retorno",
        legend=dict(orientation="h", y=-0.2, x=0.03),
    )
    return format_plotly_layout(fig)


def plot_power_spectrum(variable):
    freqs, power = compute_spectrum(df_returns[variable])
    positive = freqs > 0
    freqs = freqs[positive]
    power = power[positive]
    periods = 1.0 / freqs
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periods,
        y=power,
        mode="lines",
        line=dict(color=VAR_COLORS[variable], width=2.2),
        name="Espectro",
    ))
    fig.update_xaxes(type="log", title_text="Período (días)")
    fig.update_yaxes(type="log", title_text="Potencia")
    fig.update_layout(title=f"Espectro de potencia — {variable}")
    return format_plotly_layout(fig)


def plot_coherence_heatmap(df):
    df_plot = df.rename(columns={
        "C_semanal": "Semanal",
        "C_quincenal": "Quincenal",
        "C_mensual": "Mensual",
        "C_trimestral": "Trimestral",
    })
    z = df_plot[["Semanal", "Quincenal", "Mensual", "Trimestral"]].values
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Semanal", "Quincenal", "Mensual", "Trimestral"],
        y=df_plot["Variable"].tolist(),
        colorscale="Blues",
        zmin=0,
        zmax=0.75,
        text=np.round(z, 4),
        texttemplate="%{text}",
        hovertemplate="%{y} / %{x}: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(title="Coherencia R² por Variable y Banda de Frecuencia")
    return format_plotly_layout(fig)


def plot_coherence_bars(coh_df, variable):
    result = compute_coherence_matrix(df_returns["PFAVAL"], df_returns[variable] if variable != "PFAVAL" else df_returns["PFAVAL"])
    plot_df = result[result["Periodo"] > 0].copy()
    max_idx = plot_df["Coherencia"].idxmax()
    max_row = plot_df.loc[max_idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["Periodo"],
        y=plot_df["Coherencia"],
        mode="lines",
        line=dict(color=VAR_COLORS[variable], width=2.2),
        name="Coherencia",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#9ca3af", annotation_text="C=0.5", annotation_position="top left")
    for label, start, end in BANDAS:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(0,200,255,0.08)" if label in ["Semanal", "Mensual"] else "rgba(34,197,94,0.08)",
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font_color=PALETTE["muted"],
        )
    fig.add_trace(go.Scatter(
        x=[max_row["Periodo"]],
        y=[max_row["Coherencia"]],
        mode="markers+text",
        marker=dict(color="red", size=10),
        text=[f"{max_row['Periodo']:.1f}d · {max_row['Coherencia']:.3f}"],
        textposition="top center",
        showlegend=False,
    ))
    fig.update_xaxes(type="log", title_text="Período (días)")
    fig.update_yaxes(title_text="Coherencia C(ν)")
    fig.update_layout(title=f"Coherencia PFAVAL vs {variable}")
    return format_plotly_layout(fig)


def plot_phase_lag(coh_df, variable):
    result = compute_coherence_matrix(df_returns["PFAVAL"], df_returns[variable] if variable != "PFAVAL" else df_returns["PFAVAL"])
    plot_df = result[result["Periodo"] > 0].copy()
    max_idx = plot_df["Coherencia"].idxmax()
    max_row = plot_df.loc[max_idx]
    colors = ["rgba(34,197,94,0.18)" if val > 0 else "rgba(244,63,94,0.18)" for val in plot_df["Lag_dias"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["Periodo"],
        y=plot_df["Lag_dias"],
        mode="lines",
        line=dict(color=VAR_COLORS[variable], width=2.2),
        name="Desfase (días)",
    ))
    fig.add_hline(y=0, line_color="#9ca3af", line_dash="dash")
    fig.add_trace(go.Scatter(
        x=[max_row["Periodo"]],
        y=[max_row["Lag_dias"]],
        mode="markers+text",
        marker=dict(color="red", size=10),
        text=[f"{max_row['Lag_dias']:.2f}d"],
        textposition="bottom center",
        showlegend=False,
    ))
    fig.update_xaxes(type="log", title_text="Período (días)")
    fig.update_yaxes(title_text="Lag (días)")
    fig.update_layout(title=f"Desfase temporal PFAVAL vs {variable}")
    return format_plotly_layout(fig)


def plot_pfaval_milestones():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_unified.index,
        y=df_unified["PFAVAL"],
        name="PFAVAL",
        line=dict(color=VAR_COLORS["PFAVAL"], width=2.4),
    ))
    fig.add_trace(go.Scatter(
        x=df_filtered.index,
        y=df_filtered["PFAVAL"],
        name="Retorno filtrado PFAVAL",
        line=dict(color="skyblue", width=2.2),
        yaxis="y2",
    ))
    for date, label in [
        ("2020-03-15", "Pandemia"),
        ("2021-05-01", "Paro Nacional"),
        ("2022-10-28", "Pico Tasas Banrep"),
    ]:
        fig.add_vline(x=pd.to_datetime(date), line=dict(color="red", dash="dot"), opacity=0.7)
        fig.add_annotation(
            x=pd.to_datetime(date),
            y=df_unified["PFAVAL"].max() * 0.95,
            text=label,
            showarrow=False,
            font=dict(color="red", size=11),
            xanchor="left",
        )
    fig.update_layout(
        title="PFAVAL y hitos macroeconómicos",
        xaxis_title="Fecha",
        yaxis=dict(title="PFAVAL"),
        yaxis2=dict(title="Retorno filtrado", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.2, x=0.02),
        margin=dict(l=40, r=60, t=60, b=40),
    )
    return format_plotly_layout(fig)


def plot_lead_lag_bars(df_coh):
    df_plot = df_coh.set_index("Variable").loc[["COLCAP", "USDCOP", "WTI", "VIX", "TES_5Y"]].copy()
    df_plot["Lag"] = df_plot["lag_dias_max_coherencia"]
    df_plot["Color"] = df_plot["Lag"].apply(lambda v: PALETTE["accent4"] if v > 0 else PALETTE["accent5"])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot["Lag"].tolist()[::-1],
        y=df_plot.index.tolist()[::-1],
        orientation="h",
        marker_color=df_plot["Color"].tolist()[::-1],
        text=df_plot["Lag"].round(2).astype(str).tolist()[::-1],
        textposition="outside",
    ))
    fig.add_vline(x=0, line=dict(color="#c5ddf0", dash="dash"))
    fig.update_layout(
        title="Desfase temporal de cada variable respecto a PFAVAL",
        xaxis_title="Lag (días)",
        yaxis_title="Variable",
        margin=dict(l=120, r=80, t=60, b=50),
    )
    return format_plotly_layout(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Páginas
# ─────────────────────────────────────────────────────────────────────────────
if page == "Datos del Estudio":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">Análisis Espectral PFAVAL — Grupo Aval Preferencial</div>
            <div class="hero-title">Análisis Espectral PFAVAL</div>
            <div class="hero-sub">Fourier · BVC · 2020–2026 · 6 variables</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    metrics = [
        (f"{observaciones_totales:,}", "Observaciones"),
        (f"{fecha_inicio} → {fecha_final}", "Período"),
        ("6", "Variables analizadas"),
        ("yfinance + Banrep", "Fuentes de datos"),
    ]
    for col, (value, label) in zip(cols, metrics):
        with col:
            st.markdown(
                f"<div class='kpi'><div class='kpi-val'>{value}</div><div class='kpi-lbl'>{label}</div></div>",
                unsafe_allow_html=True,
            )
    st.markdown("---")

    stats = create_descriptive_stats(df_unified)
    table_colors = []
    for idx, row in stats.iterrows():
        variable = row["Variable"]
        color_row = []
        for col_name in stats.columns:
            if col_name == "Variable":
                color_row.append(PALETTE["surface"])
            elif variable == "WTI" and col_name == "Mínimo":
                color_row.append("rgba(249,115,22,0.2)")
            elif variable == "VIX" and col_name == "Máximo":
                color_row.append("rgba(244,63,94,0.2)")
            elif variable == "PFAVAL" and col_name in ["Mínimo", "Máximo"]:
                color_row.append("rgba(34,197,94,0.15)")
            else:
                color_row.append(PALETTE["surface"])
        table_colors.append(color_row)
    fig_table = go.Figure(data=go.Table(
        header=dict(
            values=[f"<b>{col}</b>" for col in stats.columns],
            fill_color=PALETTE["surface"],
            font=dict(color=PALETTE["text"], size=12),
            align="left",
            line_color=PALETTE["border"],
        ),
        cells=dict(
            values=[stats[col] for col in stats.columns],
            fill_color=list(map(list, zip(*table_colors))),
            font=dict(color=PALETTE["text"], size=12),
            align="left",
            line_color=PALETTE["border"],
        ),
    ))
    fig_table.update_layout(paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"], margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_table, use_container_width=True)

    st.markdown("---")
    variable = st.selectbox("Selecciona una variable", list(VAR_COLORS.keys()), index=0)
    st.plotly_chart(plot_original_series(variable), use_container_width=True)

elif page == "Transformación de Series":
    st.markdown(
        """
        <div class='insight'>
        <b>Fourier requiere series estacionarias.</b> Los precios tienen tendencia creciente — no sirven directamente. Se convierten a log-retornos diarios: log(Pt / Pt-1). Para TES_5Y, al ser una tasa y no un precio, se usa primera diferencia: Δt = Tt - Tt-1. El resultado oscila alrededor de cero sin tendencia.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            f"<div class='kpi'><div class='kpi-val'>{observaciones_fourier:,}</div><div class='kpi-lbl'>Observaciones disponibles para Fourier</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("<div class='kpi' style='background: rgba(34,197,94,0.08); border-color: rgba(34,197,94,0.35);'><div class='kpi-val'>1.504</div><div class='kpi-lbl'>Observaciones para Fourier</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    variable = st.selectbox("Selecciona una variable", list(VAR_COLORS.keys()), index=0)
    st.plotly_chart(plot_dual_axis(variable), use_container_width=True)

elif page == "Análisis Fourier Individual":
    variable = st.selectbox("Selecciona una variable", list(VAR_COLORS.keys()), index=0)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_return_comparison(variable), use_container_width=True)
    with col2:
        st.plotly_chart(plot_power_spectrum(variable), use_container_width=True)
    st.markdown("---")
    st.markdown("#### Top-5 ciclos dominantes")
    top5 = top_cycles(df_returns[variable])
    st.table(top5)

elif page == "Coherencia Espectral Cruzada":
    st.markdown("<div class='sec-hdr'>Coherencia Espectral Cruzada</div>", unsafe_allow_html=True)
    st.markdown("---")
    k1, k2, k3 = st.columns(3)
    kpis = [
        ("Variable más explicativa: COLCAP", "C mensual = 0.6705"),
        ("Único driver que anticipa a PFAVAL: TES_5Y", "Lidera 0.38 días"),
        ("Banda de mayor coherencia: Mensual (15-30 días)", ""),
    ]
    for col, (title, subtitle) in zip([k1, k2, k3], kpis):
        with col:
            st.markdown(
                f"<div class='kpi'><div class='kpi-val'>{title}</div><div class='kpi-lbl'>{subtitle}</div></div>",
                unsafe_allow_html=True,
            )
    st.markdown("---")
    st.plotly_chart(plot_coherence_heatmap(df_coherence), use_container_width=True)
    st.markdown("---")

    table_df = df_coherence.copy()
    table_df["Semanal"] = table_df["C_semanal"].map("{:.4f}".format)
    table_df["Quincenal"] = table_df["C_quincenal"].map("{:.4f}".format)
    table_df["Mensual"] = table_df["C_mensual"].map("{:.4f}".format)
    table_df["Trimestral"] = table_df["C_trimestral"].map("{:.4f}".format)
    leaders = []
    for v in table_df["Variable"]:
        if v == "COLCAP":
            leaders.append("PFAVAL +1.97d")
        elif v == "USDCOP":
            leaders.append("PFAVAL +1.72d")
        elif v == "TES_5Y":
            leaders.append("TES_5Y −0.38d")
        elif v == "VIX":
            leaders.append("PFAVAL +0.57d")
        else:
            leaders.append("PFAVAL +0.37d")
    table_df["Líder"] = leaders

    fill_colors = []
    for _, row in table_df.iterrows():
        base_color = [PALETTE["surface"]] * 6
        if row["Variable"] == "COLCAP":
            base_color = ["rgba(34,197,94,0.14)"] * 6
        if row["Variable"] == "TES_5Y":
            base_color[-1] = "rgba(249,115,22,0.22)"
        fill_colors.append(base_color)
    fill_colors = list(map(list, zip(*fill_colors)))

    fig_table = go.Figure(data=go.Table(
        header=dict(
            values=["<b>Variable</b>", "<b>Semanal</b>", "<b>Quincenal</b>", "<b>Mensual</b>", "<b>Trimestral</b>", "<b>Líder</b>"],
            fill_color=PALETTE["surface"],
            font=dict(color=PALETTE["text"], size=12),
            align="left",
            line_color=PALETTE["border"],
        ),
        cells=dict(
            values=[table_df[c] for c in ["Variable", "Semanal", "Quincenal", "Mensual", "Trimestral", "Líder"]],
            fill_color=fill_colors,
            font=dict(color=PALETTE["text"], size=12),
            align="left",
            line_color=PALETTE["border"],
        ),
    ))
    fig_table.update_layout(paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"], margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_table, use_container_width=True)

    st.markdown("---")
    variable = st.selectbox("Selecciona una variable para coherencia individual", [v for v in VAR_COLORS.keys() if v != "PFAVAL"], index=0)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_coherence_bars(variable), use_container_width=True)
    with col2:
        st.plotly_chart(plot_phase_lag(variable), use_container_width=True)

elif page == "Síntesis y Conclusiones":
    st.markdown("<div class='sec-hdr'>Síntesis y Conclusiones</div>", unsafe_allow_html=True)
    st.plotly_chart(plot_pfaval_milestones(), use_container_width=True)
    st.plotly_chart(plot_lead_lag_bars(df_coherence), use_container_width=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    conclusions = [
        (
            PALETTE["accent4"],
            "COLCAP — Driver Principal",
            "Explica el 67% del movimiento mensual de PFAVAL. Coherencia quincenal: 0.42 | Trimestral: 0.62 PFAVAL anticipa al COLCAP por 1.97 días.",
        ),
        (
            PALETTE["accent5"],
            "USD/COP — Relación de Largo Plazo",
            "R² global = 0.20, pero coherencia trimestral = 0.26. Fourier revela que el dólar solo importa en ciclos de 30 a 90 días. Es invisible en el ruido semanal.",
        ),
        (
            PALETTE["accent2"],
            "TES 5Y — El Único Que Anticipa",
            "Único driver que lidera a PFAVAL: −0.38 días. Cuando las tasas de deuda se mueven, PFAVAL reacciona antes que el mercado accionario.",
        ),
    ]
    for col, (color, title, text) in zip([c1, c2, c3], conclusions):
        with col:
            st.markdown(
                f"<div class='kpi' style='border-color: {color}; background: rgba(13,27,46,0.95);'><div class='kpi-val' style='color: {color};'>{title}</div><div class='kpi-lbl' style='color:{PALETTE['text']};'>{text}</div></div>",
                unsafe_allow_html=True,
            )
    st.markdown("---")
    st.markdown(
        f"""
        <div class='insight' style='background: #020714; border-top-color: {PALETTE['accent1']};'>
        <b>Resumen final:</b><br>
        Período: {fecha_inicio} → {fecha_final}<br>
        Observaciones: {observaciones_fourier:,} días hábiles<br>
        Ciclo dominante de PFAVAL: 32.70 días<br>
        Variable más explicativa: COLCAP (C_mensual = 0.6705)<br>
        Única variable que anticipa: TES_5Y (−0.38 días)
        </div>
        """,
        unsafe_allow_html=True,
    )
