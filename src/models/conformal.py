"""Conformal Prediction module for distribution-free uncertainty quantification.

Referencia: Romano, Y., Patterson, E., & Candès, E. J. (NeurIPS 2019).
Garantía matemática de cobertura marginal: P(Y in C(X)) >= 1 - alpha
"""
import logging
from typing import Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """Calibrador de predicción conforme para intervalos de predicción con garantías estadísticas."""

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Nivel de significancia (ej. 0.05 para cobertura del 95%, 0.10 para 90%).
        """
        self.alpha = alpha
        self.q_val = None
        self.n_cal = 0

    def calibrate(self, y_cal: np.ndarray, y_pred_cal: np.ndarray):
        """Calcula el score de no-conformidad y el cuantil de calibración conforme."""
        residuals = np.abs(y_cal - y_pred_cal)
        self.n_cal = len(residuals)
        
        # Corrección de muestra finita (Romano et al. 2019)
        quantile_level = min(1.0, np.ceil((self.n_cal + 1) * (1 - self.alpha)) / self.n_cal)
        self.q_val = float(np.quantile(residuals, quantile_level))
        
        empirical_cov = float(np.mean(residuals <= self.q_val))
        logger.info(
            "Calibración Conforme completada (n=%d, alpha=%.2f): q_val=%.4f (Cobertura Empírica: %.2f%%)",
            self.n_cal, self.alpha, self.q_val, empirical_cov * 100
        )
        return self

    def predict_interval(self, y_pred: np.ndarray, min_value: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
        """Genera el intervalo de predicción dinámico [lower, upper]."""
        if self.q_val is None:
            raise ValueError("El predictor conforme debe calibrarse con calibrate() antes de predecir.")
            
        lower_bounds = np.maximum(min_value, y_pred - self.q_val)
        upper_bounds = y_pred + self.q_val
        return lower_bounds, upper_bounds

    def evaluate_coverage(self, y_test: np.ndarray, y_pred_test: np.ndarray) -> Dict[str, float]:
        """Evalúa la cobertura empírica y el ancho medio de banda en un test set out-of-time/holdout."""
        lowers, uppers = self.predict_interval(y_pred_test)
        coverage = float(np.mean((y_test >= lowers) & (y_test <= uppers)))
        avg_width = float(np.mean(uppers - lowers))
        
        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": coverage,
            "q_conformal": self.q_val,
            "avg_interval_width": avg_width
        }
