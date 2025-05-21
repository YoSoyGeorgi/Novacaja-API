from fastapi import FastAPI, HTTPException, Response
from app.dto.proyeccion_dto import ProyeccionInput, ProyeccionOutput, DatoVentaDiaria
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
async def proyeccion(input_data: ProyeccionInput):
    # Intentar adquirir el lock
    if not await proyeccion_lock.acquire():
        raise HTTPException(
            status_code=503,
            detail="El servicio está ocupado procesando otra solicitud. Por favor, intente nuevamente en unos momentos."
        )
    
    try:
        # Crear un directorio temporal
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "datos_ventas.json")
        
        try:
            # Descargar el archivo
            response = requests.get(input_data.url_archivo)
            response.raise_for_status()  # Lanza una excepción si hay error HTTP
            
            # Guardar el archivo temporalmente
            with open(temp_file_path, 'wb') as f:
                f.write(response.content)
            
            # Leer y validar el JSON
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Validar la estructura del JSON
            if not isinstance(json_data, dict):
                raise HTTPException(status_code=400, detail="El archivo JSON debe ser un objeto")
            
            if 'datos_ventas' not in json_data:
                raise HTTPException(status_code=400, detail="El archivo JSON debe contener una clave 'datos_ventas'")
            
            if 'by_store' not in json_data:
                raise HTTPException(status_code=400, detail="El archivo JSON debe contener una clave 'by_store'")
            
            if not isinstance(json_data['by_store'], bool):
                raise HTTPException(status_code=400, detail="El valor de 'by_store' debe ser un booleano")
            
            # Convertir los datos a DatoVentaDiaria
            datos_ventas = [DatoVentaDiaria(**venta) for venta in json_data['datos_ventas']]
            
            # Procesar la proyección usando by_store del JSON
            return await calcular_proyeccion(datos_ventas, json_data['by_store'])
            
        finally:
            # Limpiar: eliminar el directorio temporal y su contenido
            shutil.rmtree(temp_dir)
            
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error al descargar el archivo: {str(e)}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="El archivo no contiene un JSON válido")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Ocurrió un problema: {str(e)}")
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
