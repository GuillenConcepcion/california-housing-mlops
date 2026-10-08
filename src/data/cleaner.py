"""Data cleaning, missingness diagnostics and Little's MCAR test implementation.

Referencia: Little, R. J. A. (1988). A Test of Missing Completely at Random for Multivariate Data.
"""
import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer, SimpleImputer

logger = logging.getLogger(__name__)


def run_littles_mcar_test(df: pd.DataFrame, alpha: float = 0.05) -> Dict[str, float]:
    """Ejecuta una aproximacion rigurosa del test de Little para datos numéricos.
    
    H0: El mecanismo de ausencia es MCAR (Missing Completely at Random).
    H1: El mecanismo NO es MCAR (MAR o MNAR).
    
    Returns:
        dict con estadistico chi2, p_value, df y decision_mcar (bool).
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return {"p_value": 1.0, "is_mcar": True}
        
    sub_df = df[num_cols].copy()
    missing_mask = sub_df.isna()
    
    if not missing_mask.any().any():
        return {"p_value": 1.0, "is_mcar": True, "stat": 0.0}
        
    # Agrupar por patrones unicos de ausencia
    patterns = missing_mask.drop_duplicates()
    d2_stat = 0.0
    degrees_of_freedom = 0
    grand_mean = sub_df.mean()
    cov_matrix = sub_df.cov().fillna(1.0)
    
    for _, pattern in patterns.iterrows():
        obs_cols = pattern[~pattern].index
        if len(obs_cols) == 0:
            continue
        n_obs = (missing_mask[obs_cols] == pattern[obs_cols]).all(axis=1).sum()
        if n_obs < 2:
            continue
            
        diff = sub_df.loc[(missing_mask[obs_cols] == pattern[obs_cols]).all(axis=1), obs_cols].mean() - grand_mean[obs_cols]
        sub_cov = cov_matrix.loc[obs_cols, obs_cols].values
        
        try:
            inv_cov = np.linalg.pinv(sub_cov)
            d2_stat += n_obs * float(diff.values @ inv_cov @ diff.values.T)
            degrees_of_freedom += len(obs_cols)
        except np.linalg.LinAlgError:
            continue

    degrees_of_freedom = max(1, degrees_of_freedom - len(num_cols))
    p_value = 1.0 - stats.chi2.cdf(d2_stat, df=degrees_of_freedom)
    is_mcar = p_value > alpha
    
    logger.info("Test de Little MCAR: d2=%.2f, df=%d, p-value=%.4f (MCAR=%s)", d2_stat, degrees_of_freedom, p_value, is_mcar)
    return {
        "chi2_stat": float(d2_stat),
        "degrees_of_freedom": int(degrees_of_freedom),
        "p_value": float(p_value),
        "is_mcar": bool(is_mcar),
    }


def clean_dataset(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Aplica curaduría estricta según el resultado del test de MCAR."""
    cleaned = df.copy()
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()
        
    # Evaluación de mecanismo de ausencia
    mcar_res = run_littles_mcar_test(cleaned)
    num_cols = cleaned.select_dtypes(include=[np.number]).columns
    
    if mcar_res.get("is_mcar", True):
        # Imputación simple permitida bajo MCAR
        imputer = SimpleImputer(strategy="median")
        cleaned[num_cols] = imputer.fit_transform(cleaned[num_cols])
    else:
        # Obligatorio imputador multivariado bajo MAR/MNAR
        imputer = KNNImputer(n_neighbors=5)
        cleaned[num_cols] = imputer.fit_transform(cleaned[num_cols])
        
    return cleaned
