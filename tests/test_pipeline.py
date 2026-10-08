"""Unit and integration tests for Ajuste de Hiperparametros Vivienda California pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import run_littles_mcar_test
from src.features.engineer import BaseFeatureEngineer
from src.monitoring.drift_detector import calculate_psi


def test_littles_mcar_clean_data():
    """Valida que un dataset sin nulos devuelva is_mcar True."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    res = run_littles_mcar_test(df)
    assert res["is_mcar"] is True


def test_feature_engineering_transformation():
    """Valida que el transformador escale y no introduzca nulos."""
    df = pd.DataFrame({"x1": [10.0, 20.0, 30.0], "x2": [1.0, 2.0, 3.0]})
    engineer = BaseFeatureEngineer()
    out = engineer.fit_transform(df)
    assert "x1_scaled" in out.columns
    assert "x2_scaled" in out.columns
    assert not out.isna().any().any()


def test_psi_identical_distributions():
    """Valida que dos distribuciones idénticas tengan PSI cercano a cero."""
    data = np.random.normal(0, 1, 1000)
    psi = calculate_psi(data, data)
    assert psi < 0.05
