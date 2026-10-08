"""Centralized configuration module for Ajuste de Hiperparametros Vivienda California.

Lead Architect: Guillén Concepción
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class ProjectSettings(BaseSettings):
    """Configuracion global del proyecto y rutas canónicas."""

    PROJECT_NAME: str = "Ajuste de Hiperparametros Vivienda California"
    PROJECT_SLUG: str = "ajuste_hiperparametros_vivienda_california"
    TASK_TYPE: str = "tabular_regression"
    
    # Rutas del Sistema de Archivos
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_PATH: Path = DATA_DIR / "raw"
    PROCESSED_DATA_PATH: Path = DATA_DIR / "processed"
    GOLDEN_DATA_PATH: Path = DATA_DIR / "golden"
    
    MODELS_DIR: Path = BASE_DIR / "models"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    FIGURES_DIR: Path = BASE_DIR / "reports" / "figures"
    
    # MLOps & Experiment Tracking
    MLFLOW_TRACKING_URI: str = Field(default="sqlite:///mlflow.db", alias="MLFLOW_TRACKING_URI")
    MLFLOW_EXPERIMENT_NAME: str = "ajuste_hiperparametros_vivienda_california_experiment"
    
    # Inferencia y Servidor API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Semilla y Reproducibilidad
    RANDOM_STATE: int = 42

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = ProjectSettings()
