from typing import Iterable
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import coherence as sp_coherence

BAND_DEFINITIONS = [
    ("Corto plazo", 4, 10),
    ("Mediano plazo", 10, 30),
    ("Largo plazo", 30, 90),
]


def _safe_series(series: pd.Series) -> pd.Series:
    return series.dropna().astype(float)


def _stationary_series_for_fft(series: pd.Series) -> pd.Series:
    price_vars = {"PFAVAL", "USDCOP", "WTI", "VIX", "CDS Colombia"}
    rate_vars = {"TES_5Y", "TPM"}
    if series.name in price_vars:
        return np.log(series / series.shift(1)).dropna()
    if series.name in rate_vars:
        return series.diff().dropna()
    return series.dropna()


@st.cache_data(show_spinner=False)
def zscore_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0)


@st.cache_data(show_spinner=False)
def compute_spectrum(series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    processed = _stationary_series_for_fft(series)
    signal = _safe_series(processed)
    n = len(signal)
    if n < 4:
        return np.array([]), np.array([]), np.array([])
    centered = signal - signal.mean()
    fft_vals = np.fft.rfft(centered)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    positive = freqs > 0
    return freqs[positive], power[positive], 1.0 / freqs[positive]


@st.cache_data(show_spinner=False)
def top_n_cycles(series: pd.Series, n: int = 5, min_period_days: float = 5.0, max_period_days: float = float("inf")) -> pd.DataFrame:
    freqs, power, periods = compute_spectrum(series)
    if len(power) == 0:
        return pd.DataFrame(columns=["Periodo_dias", "Potencia_norm"])
    valid = (periods >= min_period_days) & (periods <= max_period_days)
    if np.any(valid):
        freqs = freqs[valid]
        power = power[valid]
        periods = periods[valid]
    weights = power / np.sum(power)
    top = np.argsort(weights)[::-1][:n]
    df = pd.DataFrame(
        {
            "Periodo_dias": np.round(periods[top], 0).astype(int),
            "Potencia_norm": np.round(100 * weights[top], 1),
        }
    ).reset_index(drop=True)
    return df


def fourier_filter(series: pd.Series, n_componentes: int = 10) -> pd.Series:
    signal = _safe_series(series)
    n = len(signal)
    if n < 2:
        return pd.Series([], index=signal.index)
    centered = signal - signal.mean()
    fft_vals = np.fft.rfft(centered.values)
    keep = min(n_componentes, len(fft_vals))
    filtered_fft = np.zeros_like(fft_vals)
    filtered_fft[:keep] = fft_vals[:keep]
    filtered_values = np.fft.irfft(filtered_fft, n=n)
    return pd.Series(filtered_values + signal.mean(), index=signal.index, name=series.name)


@st.cache_data(show_spinner=False)
def dominant_period(series: pd.Series, min_period_days: float = 5.0, max_period_days: float = 90.0) -> float:
    specs = top_n_cycles(series, n=5, min_period_days=min_period_days, max_period_days=max_period_days)
    if specs.empty:
        specs = top_n_cycles(series, n=1, min_period_days=min_period_days)
    if specs.empty:
        return float("nan")
    return float(specs.loc[0, "Periodo_dias"])


def average_days_between_changes(series: pd.Series, epsilon: float = 1e-8) -> float:
    changes = series.diff().abs()
    changed = changes[changes > epsilon]
    if len(changed) < 2:
        return float("nan")
    return float(changed.index.to_series().diff().dt.days.median())


def _bandpass_fft(series: pd.Series, min_days: float, max_days: float) -> pd.Series:
    signal = _safe_series(series)
    n = len(signal)
    x = signal.values - signal.mean()
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    mask = (freqs > 0) & ((1.0 / freqs >= min_days) & (1.0 / freqs <= max_days))
    filtered = np.zeros_like(X, dtype=complex)
    filtered[mask] = X[mask]
    values = np.fft.irfft(filtered, n=n)
    return pd.Series(values, index=signal.index, name=series.name)


def calcular_coherencia_bandas(serie_x: pd.Series, serie_y: pd.Series) -> dict:
    """
    Calcula coherencia espectral entre dos series en tres bandas de tiempo.
    Retorna diccionario con valores entre 0 y 1.
    """
    df_temp = pd.DataFrame({"x": serie_x, "y": serie_y}).dropna()
    x = df_temp["x"].values
    y = df_temp["y"].values

    if len(x) < 120:
        return {"corto": 0.0, "mediano": 0.0, "largo": 0.0}

    nperseg = min(120, len(x) // 4)
    freqs, Cxy = sp_coherence(x, y, fs=1.0, nperseg=nperseg)

    periodos = np.where(freqs > 0, 1.0 / freqs, np.inf)

    mask_corto = (periodos >= 4) & (periodos <= 10)
    mask_mediano = (periodos > 10) & (periodos <= 30)
    mask_largo = (periodos > 30) & (periodos <= 90)

    coh_corto = float(np.mean(Cxy[mask_corto])) if mask_corto.any() else 0.0
    coh_mediano = float(np.mean(Cxy[mask_mediano])) if mask_mediano.any() else 0.0
    coh_largo = float(np.mean(Cxy[mask_largo])) if mask_largo.any() else 0.0

    return {
        "corto": round(coh_corto, 3),
        "mediano": round(coh_mediano, 3),
        "largo": round(coh_largo, 3),
    }


def coherence_category(value: float) -> str:
    if value > 0.7:
        return "Alta"
    if value > 0.3:
        return "Media"
    if value > 0.15:
        return "Baja"
    return "Muy baja"


@st.cache_data(show_spinner=False)
def compute_coherence_table(df_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pf = df_returns["PFAVAL"]
    variables = [c for c in df_returns.columns if c != "PFAVAL"]
    for var in variables:
        row = {"Variable": var}
        scores = calcular_coherencia_bandas(pf, df_returns[var])
        row["Corto plazo"] = scores["corto"]
        row["Mediano plazo"] = scores["mediano"]
        row["Largo plazo"] = scores["largo"]
        row["Anticipa"] = "Sí" if (scores["mediano"] > 0.20) or (scores["largo"] > 0.20) else "No"
        row["Lag_dias"] = int(_estimate_lead_lag(pf, df_returns[var]))
        rows.append(row)
    return pd.DataFrame(rows)


def _estimate_lead_lag(x: pd.Series, y: pd.Series) -> int:
    x_clean = _safe_series(x)
    y_clean = _safe_series(y)
    common = x_clean.index.intersection(y_clean.index)
    if len(common) < 5:
        return 0
    x_vals = x_clean.loc[common].values
    y_vals = y_clean.loc[common].values
    corr = np.correlate(x_vals - x_vals.mean(), y_vals - y_vals.mean(), mode="full")
    lags = np.arange(-len(x_vals) + 1, len(x_vals))
    return int(lags[np.argmax(corr)])


def synthetic_fourier_example() -> pd.DataFrame:
    t = np.linspace(0, 2 * np.pi, 200)
    cycle1 = np.sin(2 * t)
    cycle2 = 0.6 * np.sin(4 * t)
    cycle3 = 0.35 * np.sin(8 * t)
    total = cycle1 + cycle2 + cycle3
    return pd.DataFrame(
        {
            "Tiempo": t,
            "Señal total": total,
            "Ciclo 30 días": cycle1,
            "Ciclo 15 días": cycle2,
            "Ciclo 7 días": cycle3,
        }
    )
