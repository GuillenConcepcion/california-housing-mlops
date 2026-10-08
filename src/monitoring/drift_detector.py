"""Data Drift Detection via Population Stability Index (PSI) and Kolmogorov-Smirnov.

Canónico de Odysseus MLOps Framework.
"""
import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calcula el Population Stability Index (PSI) entre distribucion esperada y actual.
    
    PSI < 0.10: Sin drift significativo
    0.10 <= PSI < 0.25: Drift moderado
    PSI >= 0.25: Drift critico (Reentrenamiento requerido)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(expected, percentiles)
    bucket_bounds[0] = -np.inf
    bucket_bounds[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bucket_bounds)
    actual_counts, _ = np.histogram(actual, bins=bucket_bounds)

    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct = (actual_counts / len(actual)) + epsilon

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


def check_feature_drift(expected_df: pd.DataFrame, actual_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Evalúa drift en todas las variables numéricas mediante PSI y KS 2-Sample."""
    common_cols = [c for c in expected_df.select_dtypes(include=[np.number]).columns if c in actual_df.columns]
    results = {}
    
    for col in common_cols:
        exp_arr = expected_df[col].dropna().values
        act_arr = actual_df[col].dropna().values
        
        psi = calculate_psi(exp_arr, act_arr)
        ks_stat, p_val = ks_2samp(exp_arr, act_arr)
        
        is_drift = (psi >= 0.25) or (p_val < 0.05 and psi >= 0.10)
        
        results[col] = {
            "psi": psi,
            "ks_stat": float(ks_stat),
            "p_value": float(p_val),
            "is_drift": bool(is_drift),
        }
        
    return results
