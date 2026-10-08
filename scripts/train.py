"""Master training and Conformal Calibration script for California Housing.

Lead Architect: Guillen Concepcion (Senior Data Scientist & MLOps Engineer)
"""
import logging
import sys
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.models.conformal import ConformalPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("====================================================================")
    logger.info(" Iniciando Pipeline de Entrenamiento y Calibración Conforme (CQR 95%)")
    logger.info(" Proyecto: %s", settings.PROJECT_NAME)
    logger.info("====================================================================")

    # 1. Cargar California Housing Dataset
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    # Guardar copia raw para linaje y reproducibilidad
    settings.RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    raw_sample_path = settings.RAW_DATA_PATH / "california_housing_sample.csv"
    if not raw_sample_path.exists():
        housing.frame.head(500).to_csv(raw_sample_path, index=False)

    # 2. Partición Tripartita Causal: Train (70%), Calibración (15%), Test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=settings.RANDOM_STATE
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=settings.RANDOM_STATE
    )

    logger.info("Dataset dividido: Train=%d, Calibración=%d, Test=%d", len(X_train), len(X_cal), len(X_test))

    # 3. MLOps Experiment Tracking
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="calibrated_decision_tree_cqr_run") as run:
        # Parámetros óptimos identificados en el análisis topológico
        params = {
            "max_depth": 9,
            "max_leaf_nodes": 148,
            "min_samples_leaf": 15,
            "min_samples_split": 30,
            "random_state": settings.RANDOM_STATE
        }
        mlflow.log_params(params)

        # 4. Ajustar Árbol de Decisión Regularizado
        model = DecisionTreeRegressor(**params)
        model.fit(X_train, y_train)

        # 5. Calibrar Predictor Conforme (CQR 95%) en Holdout de Calibración
        y_pred_cal = model.predict(X_cal)
        conformal_calibrator = ConformalPredictor(alpha=0.05)
        conformal_calibrator.calibrate(y_cal.values, y_pred_cal)

        # 6. Evaluación Rigurosa en Test Holdout Out-of-Sample
        y_pred_test = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2 = r2_score(y_test, y_pred_test)

        cov_metrics = conformal_calibrator.evaluate_coverage(y_test.values, y_pred_test)

        # Log en MLflow
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("empirical_coverage_95", cov_metrics["empirical_coverage"])
        mlflow.log_metric("avg_interval_width", cov_metrics["avg_interval_width"])
        mlflow.log_metric("tree_depth", model.get_depth())
        mlflow.log_metric("n_leaves", model.get_n_leaves())

        # 7. Serializar Artefactos
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = settings.MODELS_DIR / "champion_model.joblib"
        calib_path = settings.MODELS_DIR / "conformal_calibrator.joblib"

        joblib.dump(model, model_path)
        joblib.dump(conformal_calibrator, calib_path)

        mlflow.log_artifact(str(model_path), artifact_path="models")
        mlflow.log_artifact(str(calib_path), artifact_path="models")

        logger.info("====================================================================")
        logger.info(" RESULTADOS DEL MODELO CAMPEÓN CALIBRADO (TEST HOLDOUT)")
        logger.info("====================================================================")
        logger.info(" • R² Score Holdout:       %.4f", r2)
        logger.info(" • MAE:                    $%.2fk", mae * 100)
        logger.info(" • RMSE:                   $%.2fk", rmse * 100)
        logger.info(" • Cobertura Conforme 95%%: %.2f%% (Garantía Teórica >= 95.0%%)", cov_metrics["empirical_coverage"] * 100)
        logger.info(" • Ancho Medio de Banda:   $%.2fk", cov_metrics["avg_interval_width"] * 100)
        logger.info(" • Profundidad Topológica: %d niveles", model.get_depth())
        logger.info(" • Nodos Hojas Terminales: %d hojas", model.get_n_leaves())
        logger.info(" • MLflow Run ID:          %s", run.info.run_id)
        logger.info("====================================================================")


if __name__ == "__main__":
    main()
