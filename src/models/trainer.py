"""Model training and MLflow tracking pipeline."""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.config import settings

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orquestador de entrenamiento con registro reproducible en MLflow."""

    def __init__(self, experiment_name: Optional[str] = None):
        self.experiment_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)
        self.model: Optional[Any] = None

    def build_baseline_model(self) -> Any:
        """Instancia el modelo según el tipo de tarea configurado."""
        if "tabular_regression" in ["tabular_classification", "unsupervised_clustering"]:
            return RandomForestClassifier(
                n_estimators=100,
                random_state=settings.RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            return RandomForestRegressor(
                n_estimators=100,
                random_state=settings.RANDOM_STATE,
                n_jobs=-1,
            )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, params: Optional[Dict[str, Any]] = None) -> Any:
        with mlflow.start_run(run_name="baseline_training_run") as run:
            self.model = self.build_baseline_model()
            if params:
                self.model.set_params(**params)
                mlflow.log_params(params)
                
            logger.info("Iniciando entrenamiento del modelo...")
            self.model.fit(X_train, y_train)
            
            # Guardar artefacto en models/
            settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_path = settings.MODELS_DIR / "champion_model.joblib"
            joblib.dump(self.model, model_path)
            
            mlflow.log_artifact(str(model_path), artifact_path="model_artifacts")
            logger.info("Modelo entrenado y registrado en MLflow Run: %s", run.info.run_id)
            return self.model
