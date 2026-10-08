import logging
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.features.engineer import BaseFeatureEngineer
from src.models.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Iniciando evaluacion formal del modelo...")
    model_path = settings.MODELS_DIR / "champion_model.joblib"
    if not model_path.exists():
        logger.error("No se encontro el modelo en %s. Ejecute scripts/train.py primero.", model_path)
        return

    model = joblib.load(model_path)
    
    np.random.seed(settings.RANDOM_STATE + 1)
    X_test = pd.DataFrame(
        np.random.randn(100, 8),
        columns=[f"feature_{i}" for i in range(1, 9)]
    )
    if "tabular_regression" in ["tabular_classification", "unsupervised_clustering"]:
        y_test = pd.Series(np.random.choice([0, 1], size=100), name="target")
    else:
        y_test = pd.Series(np.random.randn(100) * 10 + 50, name="target")

    engineer = BaseFeatureEngineer()
    X_test_trans = engineer.fit_transform(X_test)
    
    metrics = ModelEvaluator.evaluate(model, X_test_trans, y_test)
    logger.info("Reporte consolidado de metricas: %s", metrics)


if __name__ == "__main__":
    main()
