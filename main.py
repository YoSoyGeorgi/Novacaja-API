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
        return await calcular_proyeccion(input_data.datos_ventas)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Ocurrió un problema: {str(e)}")

@app.get("/readme")
async def get_readme():
    try:
        readme_path = os.path.join(os.path.dirname(__file__), "README.md")
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

'''
def generar_datos_ventas(fecha_inicio: date, fecha_fin: date, id_sucursal: str, art_codigo: str) -> List[DatoVentaDiaria]:
    """
    Genera datos de ventas sintéticos para el período, sucursal y artículo especificados
    """
    # Crear fecha_actual para iterar por todas las fechas entre inicio y fin
    fecha_actual = fecha_inicio
    datos_ventas = []
    
    # Definir patrones base (similar al ejemplo en modelo.py)
    # Variar patrones según la sucursal y artículo para hacer los datos más realistas
    base_venta = 100 + hash(id_sucursal) % 50  # Venta base diaria variada por sucursal
    tendencia = 0.5 + (hash(art_codigo) % 10) / 10  # Tendencia variada por artículo
    estacionalidad = 1.2  # Factor de estacionalidad
    efecto_semanal = 1.5  # Mayor efecto los fines de semana
    
    # Generar datos para cada día en el rango
    dias_totales = (fecha_fin - fecha_inicio).days + 1
    for dia in range(dias_totales):
        # Calcular la fecha actual
        fecha_actual = fecha_inicio + timedelta(days=dia)
        
        # Calcular ventas con variaciones (similar al código en modelo.py)
        venta_base = base_venta + dia * tendencia
        variacion_mensual = 20 * np.sin(2 * np.pi * fecha_actual.month / 12)
        variacion_semanal = 30 * efecto_semanal * (fecha_actual.weekday() == 5)  # Mayor venta los sábados
        ruido = random.normalvariate(0, 5)  # Ruido aleatorio
        
        # Combinar todos los efectos
        ventas = (venta_base + variacion_mensual + variacion_semanal + ruido) * estacionalidad
        
        # Asegurar que las ventas sean positivas y enteras
        ventas = max(0, round(ventas))
        
        # Agregar a la lista
        datos_ventas.append(
            DatoVentaDiaria(
                fecha=fecha_actual,
                cantidad_articulos=ventas,
                id_sucursal=id_sucursal,
                art_codigo=art_codigo
            )
        )
    
    return datos_ventas

'''