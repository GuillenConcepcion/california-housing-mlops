# 🧠 Odysseus AI — California Housing Predictive Modeling & MLOps Framework

**Subproyecto MLOps:** California Housing Predictive Modeling & Hyperparameter Optimization  
**Lead Architect:** [Guillén Concepción](https://www.linkedin.com/in/guillen-concepcion-25266b127) *(Senior Data Scientist & MLOps Engineer)*  
**Contacto:** [LinkedIn](https://www.linkedin.com/in/guillen-concepcion-25266b127) | [GitHub](https://github.com/GuillenConcepcion) | `guillenconcepcion@gmail.com`  
**Ecosistema:** Odysseus AI Platform & Predictive MLOps Framework  
**Frameworks:** `Scikit-Learn`, `XGBoost`, `Optuna`, `Scikit-Optimize`, `FastAPI`, `Plotly`, `Marimo`  
**Contenedores:** `Podman` / `Docker Multi-stage`  
**Repositorio GitHub:** [https://github.com/GuillenConcepcion/california-housing-mlops](https://github.com/GuillenConcepcion/california-housing-mlops)  

---

## 🎯 1. Descripción del Proyecto, Problema de Negocio y Alcance MLOps

### 1.1. Contexto de Negocio & Automated Valuation Models (AVM)
En las industrias de **PropTech, Banca Hipotecaria y Gestión de Riesgos Financieros**, la tasación automatizada de bienes raíces (*Automated Valuation Models - AVM*) representa un componente crítico para la originación crediticia, la provisión de capital por ratio *Loan-to-Value* (LTV) y la valoración de carteras de inversión. 

Los modelos hedónicos lineales tradicionales fracasan sistemáticamente al estimar el mercado inmobiliario debido a:
* **Interacciones territoriales no lineales:** La prima de precio atribuible a la cercanía costera (*Pacific Coast*) y centros urbanos de alta densidad (San Francisco Bay Area, Los Ángeles) genera discontinuidades espaciales abruptas.
* **Heterocedasticidad y asimetría de precios:** La varianza de los precios aumenta fuertemente en distritos de altos ingresos, invalidando los intervalos de confianza simétricos gaussianos habituales ($\hat{y} \pm 1.96\sigma$).
* **Riesgo asimétrico de tasación:** Una sobrevaloración incrementa el riesgo de impago y pérdida severa ante ejecuciones hipotecarias, mientras que una subvaloración rechaza operaciones crediticias solventes de alto retorno.

### 1.2. Pregunta Central de Negocio & Métrica North Star
> **Pregunta Estratégica:** *¿Cómo estimar el valor mediano de la vivienda (`median_house_value`) en distritos censales de California con máxima precisión predictiva, asegurando al mismo tiempo una cuantificación de incertidumbre rigurosa (garantía formal de cobertura del 95%) y una latencia de inferencia en producción apta para microservicios financieros?*

* **Métrica North Star Predictiva:** Maximizar el coeficiente de determinación fuera de muestra ($R^2_{\text{test}} > 0.83$) y minimizar el error absoluto medio ($\text{MAE}_{\text{test}} < \$31,000\text{ USD}$).
* **Métrica North Star de Seguridad & Riesgo:** Garantizar cobertura estadística marginal exacta $\mathbb{P}(Y \in C(X)) \ge 95.0\%$ en muestras finitas sin supuestos de distribución paramétrica vía **Conformalized Quantile Regression (CQR)**.
* **Métrica Operativa MLOps:** Latencia P95 de inferencia REST en producción $< 15\text{ ms}$ y detección temprana de deriva distributiva ($\text{PSI} \ge 0.10$).

---

### 1.3. Alcance del Proyecto (In-Scope vs. Out-of-Scope)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   ALCANCE DEL PROYECTO (SCOPE)                              │
├──────────────────────────────────────────────────────────┬──────────────────────────────────┤
│                       DENTRO DEL ALCANCE (IN-SCOPE)      │   FUERA DEL ALCANCE (NON-GOALS)  │
├──────────────────────────────────────────────────────────┼──────────────────────────────────┤
│ ✔ Ingesta y validación censal completa (20.640 registros)│ ✘ Procesamiento de imágenes no   │
│ ✔ Diagnóstico Little's MCAR Test (JASA 1988)             │   estructuradas (visión satelital│
│ ✔ Preprocesamiento Zero-Leakage (Pandas Copy-on-Write)   │   o fachadas de inmuebles).      │
│ ✔ Benchmarking de 9 modelos en 4 familias algorítmicas   │ ✘ Modelado de texto libre o      │
│ ✔ Optimización Bayesiana dual (Optuna TPE & skopt)       │   análisis de contratos en NLP.  │
│ ✔ Conformal Prediction CQR (Romano et al. NeurIPS 2019)  │ ✘ Tasación a nivel de vivienda   │
│ ✔ Grafos de árboles interactivos (James Gibbins)         │   individual puntual (el censo   │
│ ✔ Cartografía interactiva MapLibre con Plotly 7          │   opera a nivel de block group). │
│ ✔ Microservicio REST FastAPI con validación Pydantic V2  │ ✘ Reentrenamiento distribuido    │
│ ✔ Monitoreo de Data Drift con PSI y KS-Test 2-Sample     │   multi-nodo en clúster Spark/Ray│
│ ✔ Despliegue en contenedor multi-stage (Podman / Docker) │   (dataset de volumen tabular).  │
└──────────────────────────────────────────────────────────┴──────────────────────────────────┘
```

### 1.4. Casos de Uso Empresariales
1. **Pre-aprobación hipotecaria instantánea:** Generación de valoraciones de garantía colateral con bandas de incertidumbre certificadas al 95% para comités de crédito.
2. **Plataformas PropTech & AVM:** Estimación en tiempo real de precios justos de mercado e identificación de activos subvalorados.
3. **Planificación Urbana y Catastro Municipal:** Evaluación de brechas de asequibilidad de vivienda y tarificación fiscal predial transparente.

---

## 💻 2. Stack Tecnológico & Arquitectura de Software

El ecosistema está construido siguiendo principios **Cloud-Native**, alta eficiencia de ejecución y máxima reproducibilidad:

| Dimensión | Tecnologías / Herramientas | Rol en la Arquitectura |
| :--- | :--- | :--- |
| **Core & Lenguaje** | `Python 3.13.5 (64-bit)`, `uv 0.10.2` | Runtime moderno, gestión ultrarrápida de dependencias en Rust sin conflictos ABI. |
| **Data Engine** | `Pandas 2.x/3.x (Copy-on-Write)`, `NumPy 2.x` | Ingesta vectorial, prevención de mutaciones silenciosas y preprocesamiento de alta velocidad. |
| **Machine Learning** | `Scikit-Learn 1.6+`, `XGBoost 3.2.0` | Estimadores tabulares, árboles de decisión y gradient boosting con aceleración por histogramas. |
| **Optimización** | `Optuna 4.x`, `Scikit-Optimize (skopt 0.10.2)` | Muestreo Bayesiano Tree-structured Parzen Estimator (TPE), poda asíncrona y búsqueda condicionada. |
| **Incertidumbre** | `Conformal Prediction (CQR)` | Cuantificación de incertidumbre libre de distribución con garantía de cobertura finita. |
| **Serving & API** | `FastAPI 0.115+`, `Pydantic V2`, `Uvicorn` | Microservicio REST asíncrono, validación estricta de esquemas y latencia $< 8\text{ ms}$. |
| **Visualización** | `Plotly 7.1.0`, `Marimo 0.11+`, `NetworkX` | Mapas interactivos MapLibre, notebooks reactivos reproducibles y grafos de decisión interactivos. |
| **MLOps & DevOps** | `Podman`, `Docker Multi-stage`, `MLflow` | Contenedores ultraligeros ($\sim 180\text{ MB}$), tracking de parámetros y auditoría de Data Drift. |

---

## 📊 3. Benchmarking Multialgorítmico de Modelos

Evaluación comparativa rigurosa sobre partición independiente de Test (30% holdout, $N=6.192$ muestras):

| Algoritmo Evaluado | Familia / Paradigma | $R^2$ Train | $R^2$ Test | MAE Test ($100k) | RMSE Test ($100k) | Diagnóstico Sesgo-Varianza |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression (OLS)** | Paramétrico Clásico | 0.6741 | 0.6066 | 0.4981 | 0.6974 | Alto sesgo (Underfitting); no captura curvatura espacial costera. |
| **Ridge ($L_2$)** | Regularización Ridge | 0.6741 | 0.6066 | 0.4980 | 0.6974 | Estabiliza colinealidad pero mantiene techo lineal en $R^2 \approx 0.607$. |
| **Lasso ($L_1$)** | Regularización Sparsity | 0.6741 | 0.6066 | 0.4982 | 0.6975 | Selección de variables sin ganancia fuera de muestra. |
| **ElasticNet** | Híbrido ($L_1 + L_2$) | 0.5848 | 0.5826 | 0.5210 | 0.7183 | Sobrerregularización de coeficientes continuos. |
| **Decision Tree (Default)** | Árbol No Regularizado | 1.0000 | 0.6124 | 0.4520 | 0.7102 | Sobreajuste severo (profundidad libre $\approx 35$, memoriza ruido censal). |
| **Decision Tree (skopt Tuned)**| Árbol Podado Bayesiano | 0.7720 | 0.7145 | 0.3980 | 0.5960 | **+16.8% mejora:** Poda por costo-complejidad ($ccp\_\alpha$) restaura generalización. |
| **Random Forest** | Bagging Ensemble | 0.9780 | 0.8152 | 0.3280 | 0.4780 | Excelente reducción de varianza; alta capacidad explicativa. |
| **Gradient Boosting** | Boosting Clásico | 0.8172 | 0.7957 | 0.3470 | 0.5030 | Modelo robusto con aprendizaje secuencial de residuos. |
| **XGBoost (Optuna Tuned) 🏆**| **Gradient Boosting Optimizado** | **0.9530** | **0.8397** | **0.3085** | **0.4462** | **Modelo Campeón:** Máximo $R^2$ y menor error monetario ($\$30.85\text{k}$). |

> **Impacto del Benchmarking:** La optimización de XGBoost con Optuna logró un incremento de **+38.4% en varianza explicada** frente al modelo lineal de referencia y una reducción de **$18,960 USD** en el error monetario medio por vivienda tasada.

---

## 🔬 4. Fundamentos Teóricos y Mejoras de Ingeniería

### 4.1. Diagnóstico de Ausencia Multivariada: Test de Little MCAR (1988)
Para contrastar formalmente la hipótesis nula sobre el mecanismo generador de valores faltantes:
$$\begin{cases} H_0: \text{Los datos faltantes son Missing Completely at Random (MCAR)} \\ H_1: \text{El mecanismo de ausencia es MAR o MNAR} \end{cases}$$

El estadístico Chi-Cuadrado de Little se evalúa mediante:
$$d^2 = \sum_{s=1}^S N_s \left( \mathbf{\bar{y}}_{obs, s} - \mathbf{\hat{\mu}}_s \right)^T \mathbf{\hat{\Sigma}}_s^{-1} \left( \mathbf{\bar{y}}_{obs, s} - \mathbf{\hat{\mu}}_s \right)$$
Donde $\mathbf{\hat{\mu}}_s$ y $\mathbf{\hat{\Sigma}}_s$ se estiman vía Expectation-Maximization (EM). Al obtener $p\text{-valor} = 1.0 > 0.05$, se concluye que los 207 valores nulos de `total_bedrooms` obedecen a un patrón MCAR puro, justificando que la imputación condicional en entrenamiento preserva la insesgadez estadística de los estimadores.

### 4.2. Preprocesamiento Zero-Leakage & Compatibilidad Pandas Copy-on-Write
* **Prevención de Fugas:** Los estadísticos (medias, percentiles, rangos IQR) se aprenden estrictamente sobre $X_{\text{train}}$ (`preprocessing.fit(X_train)`), evitando la contaminación out-of-fold.
* **Compatibilidad Pandas 2+/3+:** Sustitución de mutaciones deprecadas `inplace=True` por asignación directa explícita (`X[col] = X[col].fillna(...)`) y `.clip()`, evitando fallos de actualización silenciosos por *Copy-on-Write*.
* **Codificación Determinista:** Blindaje de One-Hot Encoding mediante `.reindex()` para garantizar la presencia de las 5 categorías de proximidad oceánica (`<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`), incluso en particiones donde la clase rara `ISLAND` ($n=5$) no esté presente.

### 4.3. Optimización Bayesiana Dual: Acondicionamiento Matemático
1. **XGBoost con Optuna:**
   * Espacio de 10 dimensiones con muestreo log-uniforme de regularizadores $L_1$ (`reg_alpha`) y $L_2$ (`reg_lambda`) en $[1.0, 20.0]$.
   * Algoritmo de histogramas `tree_method='hist'` para procesamiento paralelo vectorizado.
   * *Early stopping* de 15 rondas sobre partición interna $X_{\text{valid}}$ (previniendo sobreentrenamiento).
2. **DecisionTreeRegressor con BayesSearchCV (`skopt`):**
   * **Corrección de complejidad:** Se eliminó `absolute_error` ($\mathcal{O}(N^2)$) en favor de `squared_error` y `poisson` ($\mathcal{O}(N \log N)$).
   * **Poda por Costo-Complejidad Condicionada:** Exploración de $ccp\_\alpha$ en escala logarítmica $\text{Real}(10^{-5}, 0.02, \text{prior='log-uniform'})$, evitando el colapso del árbol a un único nodo raíz.
   * Tiempo de convergencia reducido de horas a menos de 30 segundos.

### 4.4. Cuantificación de Incertidumbre con Conformal Prediction (CQR Romano et al. 2019)
A diferencia de los intervalos simétricos gaussianos que asumen normalidad irreal de los residuos, **Conformalized Quantile Regression (CQR)** genera bandas dinámicas adaptadas a la heterocedasticidad local:
$$C(X) = \left[ \hat{q}_{\alpha/2}(X) - Q_{\text{conf}}, \; \hat{q}_{1 - \alpha/2}(X) + Q_{\text{conf}} \right]$$
* **Garantía Teórica Marginal:** $\mathbb{P}\left( Y \in C(X) \right) \ge 1 - \alpha = 95.0\%$
* **Cobertura Empírica Verificada en Test:** **$95.45\%$**
* **Score Conforme de Corrección ($Q_{\text{conf}}$):** $0.0854$ (equivalente a $\$8,540\text{ USD}$)

---

## 🏗️ 5. Estructura del Repositorio

```
california-housing-mlops/
├── data/
│   ├── raw/housing.csv          # Dataset canónico completo (20.640 x 10)
│   ├── processed/               # Particiones limpias para modelado
│   └── golden/                  # Baseline calibrado
├── notebooks/
│   ├── 01_eda_and_statistical_inference.ipynb           # EDA, Test Little, Optuna y Modelos
│   ├── 01_visualising_decision_trees_california_housing.ipynb # Visualización de grafos James Gibbins
│   ├── 02_jamesdeluk_decision_trees_marimo_export.ipynb # Export reactivo interactivo Marimo
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
│   ├── presentacion_ajuste_hiperparametros_vivienda_california.pptx # Presentación ejecutiva (11 slides)
│   └── figures/                 # Gráficos ejecutivos en alta resolución (220 DPI)
├── docker/
│   ├── Dockerfile               # Multi-stage build optimizado (~180 MB)
│   └── docker-compose.yml       # Stack productivo con servidor MLflow
├── scripts/
│   ├── train.py                 # Pipeline de entrenamiento reproducible
│   ├── evaluate.py              # Validación out-of-sample
│   └── serve.py                 # Servidor de producción Uvicorn
├── run_api.bat                  # Lanzador rápido de la API REST
├── run_marimo_trees.bat         # Lanzador del visualizador de árboles Marimo
├── pyproject.toml               # Especificación PEP 621 con uv
└── README.md
```

---

## 🚀 6. Guía de Ejecución Rápida

### 6.1. Ejecución Local con Entorno Virtual
```bash
# Iniciar la API REST de inferencia
.\run_api.bat
# Documentación interactiva Swagger: http://127.0.0.1:8000/docs

# Iniciar el visualizador reactivo de árboles en Marimo
.\run_marimo_trees.bat
```

### 6.2. Inferencia Vía API REST (FastAPI)
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

## 🛡️ 7. Monitoreo MLOps y Gobernanza

* **Population Stability Index (PSI):**
  $$\text{PSI} = \sum_{b=1}^B (Actual_b - Expected_b) \times \ln\left(\frac{Actual_b + \epsilon}{Expected_b + \epsilon}\right)$$
  * $\text{PSI} < 0.10$: Operación normal sin modificaciones.
  * $0.10 \le \text{PSI} < 0.25$: Alerta preventiva de deriva de características.
  * $\text{PSI} \ge 0.25$: Disparo automático de reentrenamiento continuo (*Continuous Training*).
* **Kolmogorov-Smirnov Test (KS 2-Sample):** Detección continua de drift marginal en variables continuas críticas (`MedInc`, `AveOccup`).
* **Licencia:** MIT  
* **Lead Architect:** [Guillén Concepción](https://www.linkedin.com/in/guillen-concepcion-25266b127) *(Senior Data Scientist & MLOps Engineer)*
