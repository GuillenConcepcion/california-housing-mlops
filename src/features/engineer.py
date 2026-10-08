"""Zero-Leakage Feature Engineering Pipeline.

Implementa transformaciones estables ajustadas exclusivamente en split de entrenamiento.
"""
from typing import List, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class BaseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformador base para ingenieria de variables sin fuga de informacion."""

    def __init__(self, numeric_features: Optional[List[str]] = None):
        self.numeric_features = numeric_features or []
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not self.numeric_features:
            self.numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
            
        if self.numeric_features:
            self.scaler.fit(X[self.numeric_features])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.numeric_features:
            scaled_vals = self.scaler.transform(X_out[self.numeric_features])
            for idx, col in enumerate(self.numeric_features):
                X_out[f"{col}_scaled"] = scaled_vals[:, idx]
        return X_out
