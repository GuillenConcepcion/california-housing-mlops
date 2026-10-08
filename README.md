# 🧠 Odysseus AI — Ajuste de Hiperparámetros California Housing

**Subproyecto MLOps:** California Housing Predictive Modeling & Hyperparameter Optimization  
**Lead Architect:** [Guillén Concepción](https://www.linkedin.com/in/guillen-concepcion-25266b127) *(Senior Data Scientist & MLOps Engineer)*  
**Contacto:** [LinkedIn](https://www.linkedin.com/in/guillen-concepcion-25266b127) | [GitHub](https://github.com/GuillenConcepcion) | `guillenconcepcion@gmail.com`  
**Ecosistema:** Odysseus AI Platform & Predictive MLOps Framework  
**Frameworks:** `Scikit-Learn`, `XGBoost`, `Optuna`, `Scikit-Optimize`, `FastAPI`, `Plotly`, `Marimo`  
**Contenedores:** `Podman` / `Docker Multi-stage`  

---

## 🎯 1. Resumen Ejecutivo y Pregunta de Negocio

El objetivo estratégico de este subproyecto es predecir el valor mediano de las viviendas en distritos censales de California (`median_house_value`), abordando de forma integral los desafíos analíticos y de ingeniería de datos:
1. **No-linealidades espaciales y económicas severas:** Relación asimétrica entre ingreso familiar (`median_income`), ubicación geográfica (`latitude`, `longitude`) y proximidad costera (`ocean_proximity`).
2. **Tratamiento robusto de datos faltantes y calidad:** Diagnóstico estadístico formal de ausencia mediante el **Test de Little MCAR (1988)** sobre variables como `total_bedrooms`.
3. **Prevención de Data Leakage:** Pipeline de preprocesamiento desacoplado con ajuste estricto *Out-of-Fold* y compatibilidad nativa con Pandas 2.x/3.x (*Copy-on-Write*).
4. **Optimización Bayesiana de Alta Eficiencia:**
   - **XGBoost con Optuna (TPE Sampler):** Maximizando $R^2$ con algoritmo de histogramas y parada temprana (*early stopping*).
   - **DecisionTree con BayesSearchCV (`skopt`):** Espacio matemático regularizado con poda por costo-complejidad ($ccp\_\alpha$ logarítmico) para prevenir el sobreajuste y el colapso del árbol.
5. **Cuantificación de Incertidumbre sin Supuestos de Distribución (Conformal Prediction):** Intervalos de predicción mediante **Conformalized Quantile Regression (CQR, Romano et al. 2019)** garantizando matemáticamente cobertura marginal del **95%**.
6. **Despliegue Productivo de Baja Latencia:** Microservicio REST contenerizado con **FastAPI**, validación de tipos Pydantic e inspección de *Data Drift* continuo.

---

## 📊 2. Benchmarking de Algoritmos Evaluados

Evaluación comparativa sobre partición independiente de Test (30% holdout, $N=6.192$ muestras):

| Algoritmo | Familia / Enfoque | $R^2$ Train | $R^2$ Test | MAE Test ($100k) | RMSE Test ($100k) | Observaciones / Diagnóstico |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression** | Paramétrico / OLS | 0.6741 | 0.6066 | 0.4981 | 0.6974 | Subajuste en extremos; no captura clusters espaciales. |
| **Ridge ($L_2$)** | Regularizado ($L_2$) | 0.6741 | 0.6066 | 0.4980 | 0.6974 | Coeficientes contraídos; colinealidad controlada. |
| **Lasso ($L_1$)** | Selección de variables ($L_1$) | 0.6741 | 0.6066 | 0.4982 | 0.6975 | Esparcidad moderada; rendimiento equivalente a Ridge. |
| **ElasticNet** | Híbrido ($L_1 + L_2$) | 0.5848 | 0.5826 | 0.5210 | 0.7183 | Sobrerregularización de coeficientes continuos. |
| **Decision Tree (Default)** | Árbol no regularizado | 1.0000 | 0.6124 | 0.4520 | 0.7102 | Sobreajuste severo (profundidad libre $\approx 35$). |
| **Decision Tree (Bayes)** | Árbol regularizado (`skopt`) | 0.7720 | 0.7145 | 0.3980 | 0.5960 | Poda efectiva vía $ccp\_\alpha$; balance sesgo-varianza. |
| **Random Forest** | Bagging Ensemble | 0.9780 | 0.8152 | 0.3280 | 0.4780 | Excelente reducción de varianza; alta capacidad explicativa. |
| **Gradient Boosting** | Boosting Clásico | 0.8172 | 0.7957 | 0.3470 | 0.5030 | Modelo de base robusto y balanceado. |
| **XGBoost (Optuna)** 🏆 | **Gradient Boosting Optimizado** | **0.9530** | **0.8397** | **0.3085** | **0.4462** | **Modelo Campeón:** Convergencia óptima, menor MAE/RMSE. |

---

## 🔬 3. Fundamentos Teóricos y Mejoras de Ingeniería

### 3.1. Diagnóstico de Ausencia Multivariada: Test de Little MCAR (1988)
Para contrastar la hipótesis nula de ausencia aleatoria:
$$\begin{cases} H_0: \text{Los datos faltantes son Missing Completely at Random (MCAR)} \\ H_1: \text{El mecanismo de ausencia es MAR o MNAR} \end{cases}$$

El estadístico Chi-Cuadrado de Little se evalúa mediante:
$$d^2 = \sum_{s=1}^S N_s \left( \mathbf{\bar{y}}_{obs, s} - \mathbf{\hat{\mu}}_s \right)^T \mathbf{\hat{\Sigma}}_s^{-1} \left( \mathbf{\bar{y}}_{obs, s} - \mathbf{\hat{\mu}}_s \right)$$
Donde $\mathbf{\hat{\mu}}_s$ y $\mathbf{\hat{\Sigma}}_s$ se estiman vía Expectation-Maximization (EM). Con $p\text{-valor} > 0.05$ en bloques homogéneos, la imputación por la media condicional en entrenamiento preserva la insesgadez sin inducir distorsión en la covarianza.

### 3.2. Preprocesamiento Zero-Leakage & Compatibilidad Pandas Copy-on-Write
* **Prevención de fugas:** Estadísticos de escala y rangos IQR calculados exclusivamente sobre $X_{\text{train}}$ (`preprocessing.fit(X_train)`).
* **Compatibilidad Pandas 2+/3+:** Sustitución de llamadas deprecadas `inplace=True` por asignaciones explícitas `X[col] = X[col].fillna(...)` y `.clip()`.
* **Codificación Determinista:** Generación garantizada de dummies para categorías de muy baja frecuencia (p. ej., `ISLAND`, con sólo 5 ocurrencias en 20.640 registros) mediante `.reindex()`.

### 3.3. Optimización Bayesiana: Condicionamiento Matemático del Espacio
1. **XGBoost con Optuna:**
   * Muestreo log-uniforme de regularizadores $L_1$ (`reg_alpha`) y $L_2$ (`reg_lambda`) en $[1.0, 20.0]$.
   * Algoritmo de histogramas `tree_method='hist'` para aceleración vectorizada.
   * *Early stopping* de 15 rondas sobre partición de validación interna independiente.
2. **DecisionTreeRegressor con BayesSearchCV:**
   * **Corrección de criterio:** Enfoque en `squared_error` y `poisson` para evitar la complejidad $\mathcal{O}(N^2)$ de `absolute_error`.
   * **Poda por Costo-Complejidad:** Exploración en escala logarítmica $\text{Real}(10^{-5}, 0.02, \text{prior='log-uniform'})$ en lugar de uniforme $[0, 1]$, evitando el colapso del árbol a un nodo nulo.

### 3.4. Conformal Prediction (CQR Romano et al. 2019)
Para proporcionar garantías estadísticas rigurosas sin asumir normalidad gaussiana de residuos:
$$C(X) = \left[ \hat{q}_{\alpha/2}(X) - Q_{\text{conf}}, \; \hat{q}_{1 - \alpha/2}(X) + Q_{\text{conf}} \right]$$
* **Garantía teórica marginal:** $\mathbb{P}\left( Y \in C(X) \right) \ge 1 - \alpha = 95.0\%$
* **Cobertura empírica medida en Test:** **$95.45\%$**
* **Corrección conforme ($Q_{\text{conf}}$):** $0.0854$ (en unidades monetarias de $\$100\text{k}$)

---

## 🏗️ 4. Estructura del Repositorio

```
ajuste_hiperparametros_vivienda_california/
├── data/
│   ├── raw/housing.csv          # Dataset canónico completo (20.640 x 10)
│   ├── processed/               # Particiones limpias para modelado
│   └── golden/                  # Baseline calibrado
├── notebooks/
│   ├── 01_eda_and_statistical_inference.ipynb           # EDA, Test Little, Optuna y Modelos
│   ├── 01_visualising_decision_trees_california_housing.ipynb # Visualización de grafos James Gibbins
│   ├── 02_jamesdeluk_decision_trees_marimo_export.ipynb # Export interactivo reactivo Marimo
│   ├── 03_hyperparameter_tuning_visualised.ipynb        # Espacios bayesianos visualizados
│   └── 04_conformal_prediction_cqr.ipynb                # Calibración CQR al 95%
├── src/
│   ├── config.py                # Rutas y constantes de negocio
│   ├── data/cleaner.py          # Little's MCAR Test y validación de esquema
│   ├── features/                # Pipeline de ingeniería sin fuga (DataPreprocessing)
│   ├── models/                  # Entrenamiento, calibración isotónica y conformal
│   ├── api/main.py              # Microservicio REST FastAPI (/health, /predict)
│   └── monitoring/              # Auditoría continua de Drift (PSI / KS)
├── reports/
│   └── presentacion_ajuste_hiperparametros_vivienda_california.pptx # Presentación ejecutiva
├── docker/
│   ├── Dockerfile               # Multi-stage build optimizado
│   └── docker-compose.yml       # Stack productivo con MLflow
├── scripts/
│   ├── train.py                 # Pipeline de entrenamiento reproducible
│   ├── evaluate.py              # Validación out-of-sample
│   └── serve.py                 # Lanzador de servidor Uvicorn
├── run_api.bat                  # Lanzador rápido de la API REST
├── run_marimo_trees.bat         # Lanzador del visualizador de árboles Marimo
├── pyproject.toml               # Especificación PEP 621 con uv
└── README.md
```

---

## 🚀 5. Guía de Ejecución Rápida

### 5.1. Ejecución Local con Entorno Virtual
```bash
# Iniciar la API REST de inferencia
.\run_api.bat
# Documentación interactiva Swagger: http://127.0.0.1:8000/docs

# Iniciar el visualizador reactivo de árboles en Marimo
.\run_marimo_trees.bat
```

### 5.2. Inferencia Vía API REST (FastAPI)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "MedInc": 8.3252,
       "HouseAge": 41.0,
       "AveRooms": 6.984,
       "AveBedrms": 1.023,
       "Population": 322.0,
       "AveOccup": 2.555,
       "Latitude": 37.88,
       "Longitude": -122.23
     }'
```
**Respuesta:**
```json
{
  "prediction": 4.125,
  "confidence_interval_95": [3.682, 4.568],
  "model_version": "v1.0.0_champion",
  "conformal_coverage": 0.9545
}
```

---

## 🛡️ 6. Monitoreo MLOps y Gobernanza

* **Population Stability Index (PSI):**
  $$\text{PSI} = \sum_{b=1}^B (Actual_b - Expected_b) \times \ln\left(\frac{Actual_b + \epsilon}{Expected_b + \epsilon}\right)$$
  * $\text{PSI} < 0.10$: Operación normal sin modificaciones.
  * $0.10 \le \text{PSI} < 0.25$: Alerta preventiva de deriva de características.
  * $\text{PSI} \ge 0.25$: Disparo automático de reentrenamiento (*Continuous Training*).
* **Kolmogorov-Smirnov Test (KS 2-Sample):** Detección continua de drift marginal en variables continuas críticas (`MedInc`, `AveOccup`).
* **Licencia:** MIT  
* **Lead Architect:** [Guillén Concepción](https://www.linkedin.com/in/guillen-concepcion-25266b127)
