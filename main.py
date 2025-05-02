from fastapi import FastAPI, HTTPException, Response
from app.dto.proyeccion_dto import ProyeccionInput, ProyeccionOutput, DatoVentaDiaria
from app.services.proyeccion_service import calcular_proyeccion
import os
from datetime import datetime, timedelta, date
from typing import List, Union
import pandas as pd
import numpy as np
import random

app = FastAPI(
    title="Novacaja API de Proyección",
    description="API para calcular proyecciones de ventas y recomendaciones de stock",
    version="1.0.0"
)

# Valores predeterminados para parámetros opcionales
DEFAULTS = {
    "sucursal_agregada": False,
    "nivel_servicio": 0.95,
    "manejar_atipicos": True,
    "umbral_atipicos": 3.0,
    "lead_time": 1
}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/proyeccion", response_model=ProyeccionOutput, responses={404: {"description": "Ocurrió un problema del lado del servidor del modelo, intente nuevamente"}})
async def proyeccion(input_data: ProyeccionInput):
    try:
        # Procesar la proyección directamente con los datos de ventas proporcionados
        return await calcular_proyeccion(input_data.datos_ventas, input_data.by_store)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Ocurrió un problema: {str(e)}")

@app.get("/readme")
async def get_readme():
    try:
        readme_path = os.path.join(os.path.dirname(__file__), "/app/README.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        return Response(content=content, media_type="text/plain")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="README.md no encontrado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer README.md: {str(e)}")

@app.get("/ejemplo", response_model=ProyeccionInput)
async def get_ejemplo():
    """
    Devuelve un ejemplo de datos para usar en el endpoint de proyección
    """
    # Fechas de ejemplo (un año atrás hasta hoy)
    fecha_inicio = date.today() - timedelta(days=365)
    fecha_fin = date.today()
    
    # Crear objeto de ejemplo
    ejemplo = ProyeccionInput(
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        id_sucursal=["SUC001", "SUC002"],
        art_codigo=["P001", "P002"]
    )
    
    return ejemplo
