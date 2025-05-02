from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, date

class DatoVentaDiaria(BaseModel):
    ds: date = Field(..., description="Fecha de la venta")
    y: int = Field(..., description="Cantidad de artículos vendidos en esta fecha")
    store_id: str = Field(..., description="Identificador de la sucursal")
    art_codigo: str = Field(..., description="Código del artículo")

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
    datos_ventas: List[DatoVentaDiaria] = Field(..., description="Lista de datos de ventas diarias")
    by_store: bool = Field(default=True, description="Indica si el pronóstico debe realizarse por tienda")

    model_config = {
        "json_schema_extra": {
            "example": {
                "datos_ventas": [
                    {
                        "ds": "2023-01-01",
                        "y": 100,
                        "store_id": "S1",
                        "art_codigo": "P1"
                    }
                ],
                "by_store": True
            }
        }
    }

class ProyeccionOutput(BaseModel):
    resultados: List[ResultadoProyeccion]
    fecha_calculo: datetime = Field(default_factory=datetime.now)
    mensaje: str = "Proyección calculada exitosamente" 