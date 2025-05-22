from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, date

class DatoVentaDiaria(BaseModel):
    ds: date = Field(..., description="Fecha de la venta")
    y: int = Field(..., description="Cantidad de artículos vendidos en esta fecha")
    store_id: str = Field(..., description="Identificador de la sucursal")
    art_codigo: str = Field(..., description="Código del artículo")

class ParametrosEspeciales(BaseModel):
    art_codigo: str = Field(..., description="Código del artículo")
    store_id: str = Field(..., description="Identificador de la sucursal")
    changepoint_prior_scale: float = Field(..., description="Parámetro de ajuste de Prophet para la flexibilidad de la tendencia")
    seasonality_prior_scale: float = Field(..., description="Parámetro de ajuste de Prophet para la fuerza de la estacionalidad")
    changepoint_range: float = Field(..., description="Rango de puntos de cambio en Prophet")

class ResultadoProyeccion(BaseModel):
    id_sucursal: str
    art_codigo: str
    demanda_pronosticada_7d: int
    demanda_pronosticada_30d: int
    stock_seguridad_7d: int
    stock_seguridad_30d: int
    stock_recomendado_7d: int
    stock_recomendado_30d: int
    intervalo_confianza_inferior: int
    intervalo_confianza_superior: int
    tendencia: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "id_sucursal": "S1",
                "art_codigo": "P1",
                "demanda_pronosticada_7d": 100,
                "demanda_pronosticada_30d": 200,
                "stock_seguridad_7d": 50,
                "stock_seguridad_30d": 100,
                "stock_recomendado_7d": 75,
                "stock_recomendado_30d": 150,
                "intervalo_confianza_inferior": 70,
                "intervalo_confianza_superior": 130,
                "tendencia": "Creciendo"
            }
        }
    }

class ProyeccionInput(BaseModel):
    by_store: bool = Field(..., description="Indica si el pronóstico debe realizarse por tienda")
    parametros_especiales: Optional[List[ParametrosEspeciales]] = Field(None, description="Lista de parámetros especiales para artículos específicos")
    datos_ventas: List[DatoVentaDiaria] = Field(..., description="Lista de datos de ventas diarias")

    model_config = {
        "json_schema_extra": {
            "example": {
                "by_store": True,
                "parametros_especiales": [
                    {
                        "art_codigo": "P1",
                        "store_id": "S1",
                        "changepoint_prior_scale": 0.05,
                        "seasonality_prior_scale": 0.1,
                        "changepoint_range": 0.8
                    }
                ],
                "datos_ventas": [
                    {
                        "art_codigo": "P1",
                        "ds": "2023-01-01",
                        "store_id": "S1",
                        "y": 15
                    }
                ]
            }
        }
    }

class ProyeccionOutput(BaseModel):
    resultados: List[ResultadoProyeccion]
    fecha_calculo: datetime = Field(default_factory=datetime.now)
    mensaje: str = "Proyección calculada exitosamente"

class ProyeccionURLInput(BaseModel):
    url: HttpUrl = Field(..., description="URL del archivo JSON con los datos de proyección")

    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://ejemplo.com/datos_proyeccion.json"
            }
        }
    } 