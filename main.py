from fastapi import FastAPI, HTTPException, Response
from app.dto.proyeccion_dto import ProyeccionInput, ProyeccionOutput, DatoVentaDiaria, ProyeccionURLInput
from app.services.proyeccion_service import calcular_proyeccion
import os
from datetime import datetime, timedelta, date
from typing import List, Union
import pandas as pd
import numpy as np
import random
import requests
import json
import tempfile
import shutil
import asyncio
from fastapi.responses import JSONResponse
from fastapi.requests import Request

app = FastAPI(
    title="Novacaja API de Proyección",
    description="API para calcular proyecciones de ventas y recomendaciones de stock",
    version="1.0.0"
)

# Crear un lock global para el endpoint de proyección
proyeccion_lock = asyncio.Lock()

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

@app.post("/proyeccion", response_model=ProyeccionOutput, responses={
    404: {"description": "Ocurrió un problema del lado del servidor del modelo, intente nuevamente"},
    400: {"description": "Error al procesar el archivo JSON o URL inválida"},
    503: {"description": "El servicio está ocupado procesando otra solicitud"}
})
async def proyeccion(input_data: ProyeccionURLInput):
    # Intentar adquirir el lock
    if not await proyeccion_lock.acquire():
        raise HTTPException(
            status_code=503,
            detail="El servicio está ocupado procesando otra solicitud. Por favor, intente nuevamente en unos momentos."
        )
    
    try:
        # Descargar el archivo JSON
        try:
            response = requests.get(str(input_data.url))
            response.raise_for_status()  # Lanzar excepción si hay error HTTP
            input_json = response.json()
        except requests.RequestException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error al descargar el archivo JSON: {str(e)}"
            )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="El archivo descargado no es un JSON válido"
            )
        
        # Validar la estructura del JSON
        try:
            input_data = ProyeccionInput(**input_json)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"El JSON no tiene la estructura correcta: {str(e)}"
            )
        
        # Procesar la proyección
        return await calcular_proyeccion(
            datos_ventas=input_data.datos_ventas,
            by_store=input_data.by_store,
            parametros_especiales=input_data.parametros_especiales
        )
            
    except HTTPException as he:
        # Preservar el error HTTP original
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")
    finally:
        # Liberar el lock
        proyeccion_lock.release()

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

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Error interno del servidor: {str(exc)}"}
    )
