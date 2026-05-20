from typing import Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import streamlit as st

VARIABLE_LABELS = {
    "USDCOP": "USD/COP",
    "WTI": "WTI",
    "VIX": "VIX",
    "TES_5Y": "TES 5Y",
    "TPM": "TPM",
    "CDS Colombia": "CDS Colombia",
}


def fit_ols(x: pd.Series, y: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = sm.add_constant(x)
    return sm.OLS(y, X).fit()


def simple_regression(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    model = fit_ols(x, y)
    slope = float(model.params.iloc[1]) if len(model.params) > 1 else 0.0
    intercept = float(model.params.iloc[0])
    r2 = float(model.rsquared)
    pvalue = float(model.pvalues.iloc[1]) if len(model.pvalues) > 1 else 1.0
    corr = float(x.corr(y)) if x.size and y.size else 0.0
    rmse = float(np.sqrt(mean_squared_error(y, model.fittedvalues)))
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "pvalue": pvalue,
        "corr": corr,
        "rmse": rmse,
        "model": model,
    }


def format_relation(r2: float) -> str:
    if r2 > 0.3:
        return "✅"
    if r2 > 0.1:
        return "⚠️"
    return "❌"


def build_variable_summary(df_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, label in VARIABLE_LABELS.items():
        if key not in df_returns.columns:
            continue
        metrics = simple_regression(df_returns[key], df_returns["PFAVAL"])
        rows.append(
            {
                "Variable": label,
                "R2": round(metrics["r2"], 3),
                "Pendiente": round(metrics["slope"], 4),
                "p-valor": round(metrics["pvalue"], 4),
                "Correlación": round(metrics["corr"], 3),
                "Relación": format_relation(metrics["r2"]),
            }
        )
    return pd.DataFrame(rows)


def build_filter_comparison(df_returns: pd.DataFrame, df_filtered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, label in VARIABLE_LABELS.items():
        if key not in df_returns.columns or key not in df_filtered.columns:
            continue
        orig = simple_regression(df_returns[key], df_returns["PFAVAL"])
        filt = simple_regression(df_filtered[key], df_filtered["PFAVAL"])
        delta_r2 = filt["r2"] - orig["r2"]
        improvement_points = 100 * delta_r2
        rows.append(
            {
                "Variable": label,
                "R2 Original": round(orig["r2"], 3),
                "R2 Filtrado": round(filt["r2"], 3),
                "Mejora_pct": round(improvement_points, 1),
                "Interpretación": _interpret_improvement(improvement_points),
            }
        )
    return pd.DataFrame(rows)


def _interpret_improvement(value: float) -> str:
    if value > 5:
        return "Relación real de mediano plazo"
    if value > 1:
        return "Relación moderada mejora"
    return "Relación débil persiste"


def multi_regression(df: pd.DataFrame, features: list[str]) -> Dict[str, object]:
    X = df[features]
    y = df["PFAVAL"]
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    coef_table = pd.DataFrame(
        {
            "Variable": model.params.index,
            "Coeficiente": model.params.values,
            "p-valor": model.pvalues.values,
        }
    )
    coef_table["Significativa"] = coef_table["p-valor"] < 0.05
    coef_table["Interpretación"] = coef_table.apply(lambda row: _variable_interpretation(row["Variable"], row["Coeficiente"]), axis=1)
    metrics = {
        "r2": float(model.rsquared),
        "r2_adj": float(model.rsquared_adj),
        "rmse": float(np.sqrt(mean_squared_error(y, model.fittedvalues))),
        "significativas": int(coef_table.loc[coef_table["Significativa"], "Variable"].size - 1),
        "coef_table": coef_table,
        "model": model,
    }
    return metrics


def _variable_interpretation(variable: str, coef: float) -> str:
    if variable == "const":
        return "Nivel promedio de PFAVAL cuando las variables son cero."
    signo = "sube" if coef > 0 else "baja"
    return f"Por cada unidad de la variable, PFAVAL {signo} en {abs(coef):.4f}."
