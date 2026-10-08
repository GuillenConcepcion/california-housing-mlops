"""Data loader module for Ajuste de Hiperparametros Vivienda California."""
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)


def load_raw_dataset(file_name: str, subfolder: Optional[str] = None) -> pd.DataFrame:
    """Carga un dataset desde el almacenamiento raw garantizando inmutabilidad.
    
    Args:
        file_name: Nombre del archivo a cargar (csv o parquet).
        subfolder: Subcarpeta opcional dentro de raw.
        
    Returns:
        pd.DataFrame: DataFrame con los datos brutos.
    """
    base_path = settings.RAW_DATA_PATH
    if subfolder:
        base_path = base_path / subfolder
        
    target_path = base_path / file_name
    if not target_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado en ruta raw: {target_path}")
        
    if target_path.suffix == ".parquet":
        df = pd.read_parquet(target_path)
    else:
        df = pd.read_csv(target_path)
        
    logger.info("Dataset raw cargado: %s con shape %s", file_name, df.shape)
    return df
