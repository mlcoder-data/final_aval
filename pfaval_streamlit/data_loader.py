from pathlib import Path
import re
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Base del repositorio (dos niveles arriba del archivo: proyecto_root)
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "pfaval_study" / "data" / "raw"

RUTAS = {
    "TES_5Y": RAW_DIR / "TSE.xlsx",
    "TPM": RAW_DIR / "TPM.xlsx",
    "CDS Colombia": RAW_DIR / "CDS_COL.csv",
}

TICKERS = {
    "PFAVAL": "PFAVAL.CL",
    "USDCOP": "COP=X",
    "WTI": "CL=F",
    "VIX": "^VIX",
}

PERIODO = ("2020-01-02", "2026-05-11")

FALLBACK_PARAMS = {
    "PFAVAL": {"start": 980.0, "mu": 0.00025, "sigma": 0.018},
    "USDCOP": {"start": 3300.0, "mu": 0.00008, "sigma": 0.006},
    "WTI": {"start": 45.0, "mu": 0.0004, "sigma": 0.03},
    "VIX": {"start": 18.0, "mu": 0.0001, "sigma": 0.02},
}


def _clean_numeric_column(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({"-": np.nan, "": np.nan})
    cleaned = cleaned.str.replace(r"\.", "", regex=True)
    cleaned = cleaned.str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _find_date_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "fecha" in str(col).lower() or "date" in str(col).lower():
            return col
    return df.columns[0]


def _choose_value_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "fecha" not in str(c).lower() and "date" not in str(c).lower()]
    daily = [c for c in candidates if "diario" in str(c).lower() or "daily" in str(c).lower()]
    if daily:
        return daily[0]
    counts = [(c, df[c].notna().sum()) for c in candidates]
    counts.sort(key=lambda item: item[1], reverse=True)
    return counts[0][0]


def _load_excel_data(path: Path) -> pd.Series:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Archivo Excel no encontrado: {path}. Coloque los archivos en '{RAW_DIR}' o actualice `RUTAS` en pfaval_streamlit/data_loader.py"
        )
    df = pd.read_excel(path, dtype=str)
    if df.shape[0] > 0:
        first_row = df.iloc[0].astype(str).str.lower().str.strip()
        if first_row.str.contains("dd/mm").any() or first_row.str.contains("%|").any():
            df = df.iloc[1:].copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = _find_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col].astype(str).str.strip(), dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col])
    value_col = _choose_value_column(df)
    df[value_col] = _clean_numeric_column(df[value_col])
    return df.set_index(date_col)[value_col].sort_index()


def _load_cds_data(path: Path) -> pd.Series:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Archivo CDS no encontrado: {path}. Coloque los archivos en '{RAW_DIR}' o actualice `RUTAS` en pfaval_streamlit/data_loader.py"
        )
    for sep in [",", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, dtype=str)
        except Exception:
            continue
        if "Date" in df.columns and "Price" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), dayfirst=False, errors="coerce")
            df = df.dropna(subset=["Date"])  # keep only valid dates
            df["Price"] = _clean_numeric_column(df["Price"])
            return df.set_index("Date")["Price"].sort_index()
    raise ValueError(f"No se pudo leer el archivo CDS desde {path}")


def _load_yfinance_close(symbol: str, start: str, end: str) -> pd.Series:
    data = yf.download(symbol, start=start, end=end, progress=False, threads=False)
    if data is None or data.empty:
        return pd.Series(dtype=float)
    close_cols = [c for c in data.columns if "close" in str(c).lower()]
    if not close_cols:
        return pd.Series(dtype=float)
    series = data[close_cols[0]].copy()
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def _simulate_if_missing(name: str, index: pd.DatetimeIndex) -> pd.Series:
    params = FALLBACK_PARAMS.get(name, {"start": 100.0, "mu": 0.0, "sigma": 0.01})
    np.random.seed(2025 + hash(name) % 1000)
    dt = 1 / 252
    steps = len(index)
    increments = np.random.normal(loc=(params["mu"] - 0.5 * params["sigma"] ** 2) * dt, scale=params["sigma"] * np.sqrt(dt), size=steps)
    values = np.exp(np.cumsum(increments)) * params["start"]
    return pd.Series(values, index=index, name=name)


@st.cache_data(show_spinner=False)
def load_original_data() -> tuple[pd.DataFrame, list[str]]:
    warnings = []
    pfaval = _load_yfinance_close(TICKERS["PFAVAL"], PERIODO[0], PERIODO[1])
    if pfaval.empty:
        business_days = pd.bdate_range(start=PERIODO[0], end=PERIODO[1])
        pfaval = _simulate_if_missing("PFAVAL", business_days)
        warnings.append("PFAVAL no pudo descargarse de yfinance y se generó una serie simulada para mantener la alineación.")
    base_index = pfaval.index

    series = {"PFAVAL": pfaval}
    for name, ticker in TICKERS.items():
        if name == "PFAVAL":
            continue
        candidate = _load_yfinance_close(ticker, PERIODO[0], PERIODO[1])
        if candidate.empty:
            candidate = _simulate_if_missing(name, base_index)
            warnings.append(f"{name} no pudo descargarse de yfinance y se generó una serie simulada.")
        series[name] = candidate.reindex(base_index).ffill().bfill()

    local_tes = _load_excel_data(RUTAS["TES_5Y"])
    local_tpm = _load_excel_data(RUTAS["TPM"])
    local_cds = _load_cds_data(RUTAS["CDS Colombia"])

    series["TES_5Y"] = local_tes.reindex(base_index).ffill().bfill()
    series["TPM"] = local_tpm.reindex(base_index).ffill().bfill()
    series["CDS Colombia"] = local_cds.reindex(base_index).ffill().bfill()

    df_original = pd.DataFrame(series).dropna()
    if df_original.empty:
        raise ValueError("No hay datos originales alineados después de la carga.")
    return df_original, warnings


@st.cache_data(show_spinner=False)
def prepare_returns(df_original: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=df_original.index)
    for name in ["PFAVAL", "USDCOP", "WTI", "VIX", "CDS Colombia"]:
        df[name] = np.log(df_original[name] / df_original[name].shift(1))
    for name in ["TES_5Y", "TPM"]:
        df[name] = df_original[name].diff()
    return df.dropna()


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df_original, warnings = load_original_data()
    df_returns = prepare_returns(df_original)
    return df_original, df_returns, warnings
