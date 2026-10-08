"""Model evaluation module with business metrics and Conformal Prediction."""
import logging
from typing import Any, Dict
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluador de modelos con métricas de rigor MLOps."""

    @staticmethod
    def evaluate(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        preds = model.predict(X_test)
        metrics: Dict[str, float] = {}

        if "tabular_regression" == "tabular_classification":
            metrics["accuracy"] = float(accuracy_score(y_test, preds))
            metrics["macro_f1"] = float(f1_score(y_test, preds, average="macro"))
        else:
            metrics["mae"] = float(mean_absolute_error(y_test, preds))
            metrics["r2_score"] = float(r2_score(y_test, preds))

        for k, v in metrics.items():
            mlflow.log_metric(k, v)
            logger.info("Métrica %s: %.4f", k, v)

        return metrics
