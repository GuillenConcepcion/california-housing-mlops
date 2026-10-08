"""Pydantic schemas for California Housing inference and conformal intervals."""
from typing import Dict, Optional
from pydantic import BaseModel, Field


class CaliforniaHousingFeatures(BaseModel):
    """Características de un distrito censal de California."""
    MedInc: float = Field(..., description="Ingreso mediano en decenas de miles de dólares", example=8.3252)
    HouseAge: float = Field(..., description="Antigüedad mediana de las viviendas en años", example=41.0)
    AveRooms: float = Field(..., description="Promedio de habitaciones por hogar", example=6.9841)
    AveBedrms: float = Field(..., description="Promedio de dormitorios por hogar", example=1.0238)
    Population: float = Field(..., description="Población del distrito censal", example=322.0)
    AveOccup: float = Field(..., description="Promedio de ocupantes por hogar", example=2.5555)
    Latitude: float = Field(..., description="Coordenada de latitud geográfica", example=37.88)
    Longitude: float = Field(..., description="Coordenada de longitud geográfica", example=-122.23)


class ConformalIntervalDTO(BaseModel):
    """Intervalo de predicción conforme con garantía marginal de muestra finita."""
    lower_usd: float = Field(..., description="Límite inferior calibrado en dólares")
    upper_usd: float = Field(..., description="Límite superior calibrado en dólares")
    coverage_guarantee: str = Field(default="95.0%", description="Garantía estadística marginal")
    interval_width_usd: float = Field(..., description="Ancho de banda del intervalo de incertidumbre")


class HousingPredictionResponse(BaseModel):
    """Respuesta estructurada del microservicio de inferencia."""
    predicted_value_usd: float = Field(..., description="Valor predicho de la vivienda en dólares")
    predicted_value_raw: float = Field(..., description="Valor en unidades estándar del dataset")
    conformal_interval_95: Optional[ConformalIntervalDTO] = None
    model_version: str = "v1.0.0-champion"
    latency_ms: float
    status: str = "success"
