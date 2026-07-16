# Clasificación de COVID-19 mediante señales acústicas de tos

Clasificación binaria de audios de tos para detectar casos positivos de COVID-19, usando extracción de características acústicas clásicas, reducción de dimensionalidad, optimización bayesiana de hiperparámetros y ensambles de modelos.

## Dataset

| Clase | Descripción | Muestras |
|---|---|---|
| 0 | Paciente Negativo | 1207 |
| 1 | Paciente Positivo | 150 |

Desbalance ≈ **8:1**. `random_state = 42` en todo el pipeline para reproducibilidad.

## Estructura del notebook (`ML_covid.ipynb`)

1. **Carga de datos y EDA profundo** — Extracción de 136 características acústicas (MFCC, delta/delta-delta, espectrales, armónicas) con `librosa`, espectrogramas Mel y análisis de separabilidad no lineal con t-SNE y UMAP.
2. **Reducción de dimensionalidad** — Comparación empírica entre PCA (varianza) y LDA (separabilidad supervisada).
3. **Control de complejidad (Bishop)** — Regularización L1/L2 para mitigar el sobreajuste de máxima verosimilitud.
4. **Optimización bayesiana (Optuna, TPE)** — Búsqueda de hiperparámetros para SVM, LightGBM y XGBoost con validación cruzada estratificada (K=5) y sin fuga de datos.
5. **Evaluación de modelos base** — Comparación por K-fold CV y prueba final en test held-out (XGBoost como mejor modelo individual).
6. **Arquitectura avanzada** — Embeddings profundos con YAMNet, stacking topológico (UMAP + DBSCAN), modelo generativo bayesiano (GMM + EM + BIC), Focal Loss en LightGBM, y teoría de decisión con matriz de costos (FN cuesta 100× más que FP).
7. **Maximización del Macro F1** — Blending de las mejores combinaciones de características/modelo + seed bagging (5 semillas × 3 combos).

## Resultado final

**Macro F1 = 0.807** en test (blend de 3 arquitecturas × 5 semillas = 15 modelos promediados), frente a 0.765 del mejor modelo individual (XGBoost/LightGBM con Optuna).

## Requisitos

```bash
uv pip install librosa soundfile scikit-learn imbalanced-learn xgboost lightgbm optuna umap-learn pandas numpy matplotlib seaborn
# Opcional (Módulo 0 — YAMNet):
uv pip install tensorflow tensorflow-hub
```

## Estructura de datos esperada

El notebook define `DATA_DIR = "cleaned_data"`. La carpeta ya está incluida en este repositorio:

```
cleaned_data/
├── Negative/   # audios .wav / .mp3 de pacientes negativos (1207 muestras)
├── Positive/   # audios .wav / .mp3 de pacientes positivos (150 muestras)
└── Unknown/    # audios sin etiqueta confirmada (no usados en el pipeline)
```

## Salidas generadas

- `features_cache.npz` — caché de las 136 características extraídas.
- `yamnet_cache.npz` — caché de embeddings YAMNet (opcional).
- `resultados_cv_experto.csv` — tabla comparativa de modelos.
- `figuras/` — espectrogramas, proyecciones t-SNE/UMAP, PCA/LDA, curvas de regularización, historia de Optuna y matrices de confusión.

## Autores

- Christopher Mendoza
- Alex Valdez
- Piero Freundt
- Diego Trujillo
