# RESUMEN METODOLÓGICO — Análisis Espectral PFAVAL

## Objetivo
Descomponer el comportamiento de PFAVAL (acciones preferenciales de Grupo Aval, BVC) mediante análisis de Fourier para identificar ciclos dominantes, relaciones espectrales con variables macroeconómicas y liderazgos temporales.

## Datos
**Período:** 02/01/2020 → 11/05/2026 | **Observaciones:** 1.507 días hábiles

**Variables (6):**
- PFAVAL: precio acción preferencial
- USDCOP: tasa de cambio dólar/peso
- WTI: precio petróleo crudo (USD/barril)
- VIX: índice volatilidad implícita CBOE
- COLCAP: índice bursátil colombiano
- TES_5Y: tasa bonos deuda 5 años (Banrep)

**Fuentes:**
- yfinance: PFAVAL, USDCOP, WTI, VIX
- Banco de la República: COLCAP, TES_5Y

---

## PIPELINE DE ANÁLISIS

### Sección 1-2: Carga y Limpieza de Datos
- Lectura de archivos .xlsx (Banrep) y descarga desde yfinance
- Conversión de formatos numéricos colombianos (1.658,77 → 1658.77)
- Tratamiento de fechas y valores faltantes
- Guardado de series en parquet por variable
- **Output:** `df_unified.parquet` (6 columnas, 1.507 filas)

### Sección 3: Transformación a Series Estacionarias
- **Log-retornos:** `Rt = log(Pt / Pt-1)` para PFAVAL, USDCOP, WTI, VIX, COLCAP
- **Primera diferencia:** `ΔTt = Tt - Tt-1` para TES_5Y (es tasa, no precio)
- Eliminación de NaN: mayo 2020 (WTI negativo abril) → **1.504 observaciones estacionarias**
- **Output:** `df_returns.parquet`, `df_returns_filtered.parquet`

### Sección 4: Análisis de Fourier Individual
**Metodología:**
1. **FFT (Transformada Rápida de Fourier):** `X[k] = Σ x[n]·exp(-2πikn/N)` sobre cada serie
2. **Espectro de Potencia:** `|X[k]|²` para extraer energía por frecuencia
3. **Conversión a Períodos:** `T = 1/f` (días)
4. **Top-5 Ciclos Dominantes:** ranking por potencia espectral
5. **Filtro Pasa-Bajas:** preservación de DC (k=0) + top-10 componentes de mayor potencia
   - Simetría hermitiana: `X[N-k] = X*[k]` para garantizar IFFT real
6. **Visualización:** 3 paneles por variable (series filtradas, espectro log-log, tabla ciclos)

**Resultados Clave:**
- Ciclo dominante PFAVAL: **32.70 días** (frecuencia 0.0306 d⁻¹)
- COLCAP: 32.80 días (sincronizado con PFAVAL)
- Persistencia de ciclos mensuales en todas las variables

### Sección 5: Análisis Espectral Cruzado (Coherencia)
**Metodología:**
1. **Coherencia Espectral:** `C(ν) = |Sxy(ν)|² / (Sxx(ν)·Syy(ν))`
   - Rango: [0,1] | Interpretación: proporción de varianza compartida a frecuencia ν
2. **Densidad Espectral Cruzada:** `Sxy = FFT(x)·conj(FFT(y))`
3. **Desfase Temporal:** `φ(ν) = arg(Sxy)` → `lag = φ/(2π·f)` (en días)
4. **Bandas de Frecuencia:**
   - Semanal (4–8 días)
   - Quincenal (8–15 días)
   - Mensual (15–30 días)
   - Trimestral (30–90 días)
5. **R² por Banda:** correlación pearsoniana de series filtradas en cada banda

**Resultados Principales:**
| Variable | C_semanal | C_quincenal | C_mensual | C_trimestral | Líder |
|----------|-----------|------------|-----------|--------------|-------|
| COLCAP | 0.1608 | 0.4239 | **0.6705** | 0.6154 | PFAVAL +1.97d |
| USDCOP | 0.0108 | 0.0259 | 0.1242 | 0.2610 | PFAVAL +1.72d |
| WTI | 0.0044 | 0.0011 | 0.0419 | 0.0834 | PFAVAL +0.37d |
| VIX | 0.0281 | 0.0015 | 0.0439 | 0.0679 | PFAVAL +0.57d |
| **TES_5Y** | 0.0141 | 0.0224 | 0.1063 | 0.1427 | **TES_5Y −0.38d** |

**Interpretación:**
- COLCAP: 67% de varianza compartida mensual; anticipa PFAVAL por 1.97 días
- TES_5Y: ÚNICO driver que anticipa PFAVAL (lidera 0.38 días)
- USDCOP: relación invisible en ruido semanal, visible en plazos 30–90 días
- Todas las coherencias concentradas en ciclos 15–90 días (no hay ruido semanal)

### Sección 6: Síntesis Final
- Figura consolidada 2×3 paneles con:
  - R² por variable (regresión filtrada)
  - Heatmap de coherencias por banda
  - Ciclos dominantes por variable
  - Desfases temporales
  - Serie PFAVAL con hitos macroeconómicos (Pandemia, Paro, Pico Tasas)
  - Cuadro de conclusiones ejecutivas

---

## HALLAZGOS CLAVE

1. **Ciclo de 32.70 días es dominante** en PFAVAL y sus covariables (resonancia de mercado)
2. **COLCAP es el driver principal:** explica 67% de movimiento mensual de PFAVAL
3. **TES_5Y es el único predictor:** anticipa PFAVAL, revelando eficiencia del mercado de deuda
4. **USD/COP importa solo a largo plazo:** invisible en ciclos cortos (<15 días)
5. **VIX tiene bajo acoplamiento:** volatilidad global débilmente correlacionada con PFAVAL local

---

## ARQUIVOS GENERADOS

**Datos Procesados:**
- `data/processed/df_unified.parquet` (series originales)
- `data/processed/df_returns.parquet` (retornos estacionarios)
- `data/processed/df_returns_filtered.parquet` (retornos filtrados por Fourier)

**Tablas:**
- `outputs/tables/coherencia_bandas.csv` (métrica principal)
- `outputs/tables/r2_comparativo.csv` (comparativa regresiones)

**Figuras:**
- `01_series_retornos.png` – 6 paneles originales vs retornos
- `02_fourier_[var].png` – FFT individual por variable (6 figuras)
- `03_espectros_comparados.png` – overlay de todos los espectros
- `06_coherencia_[var].png` – coherencia + desfase por variable (5 figuras)
- `07_heatmap_coherencia.png` – matriz coherencias R² por banda
- `08_sintesis_final.png` – cuadro resumen 2×3 paneles

---

## TECNOLOGÍAS

- **Python 3.11** | NumPy, Pandas, Scipy
- **FFT:** numpy.fft (algoritmo Cooley-Tukey)
- **Coherencia:** scipy.signal.coherence, scipy.signal.csd
- **Visualización:** Matplotlib, Seaborn
- **Persistencia:** Parquet (Apache), CSV
