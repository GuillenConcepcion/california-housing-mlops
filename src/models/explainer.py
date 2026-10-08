"""Explainable AI (XAI) module with SHAP (Shapley Additive exPlanations).

Referencia: Lundberg, S. M., & Lee, S.-I. (NeurIPS 2017).
"""
import logging
from typing import Any
import numpy as np
import pandas as pd
import shap

from src.config import settings

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Generador de explicabilidad local y global con SHAP."""

    def __init__(self, model: Any):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X_sample: pd.DataFrame) -> np.ndarray:
        logger.info("Calculando valores SHAP para %d observaciones...", len(X_sample))
        shap_values = self.explainer.shap_values(X_sample)
        return shap_values
