from app.dao.proyeccion_dao import ProyeccionDAO
from app.dto.proyeccion_dto import ProyeccionInput, ProyeccionOutput, ResultadoProyeccion, DatoVentaDiaria
from datetime import datetime, timedelta
from typing import Dict, Any, List, Union, Optional
from fastapi import HTTPException, status
import logging
import asyncio
import psutil
import gc
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from itertools import groupby
from operator import itemgetter
from collections import defaultdict
import time
from app.dao.modelo import run_forecast
import json

# Configurar logger
logger = logging.getLogger(__name__)

# Constantes para optimización
MAX_MEMORY_PERCENT = 90  # Porcentaje máximo de memoria a utilizar
MIN_MEMORY_REQUIRED_MB = 1024  # Memoria mínima requerida en MB
NUM_WORKERS = multiprocessing.cpu_count()  # Número de workers (uno por núcleo físico)
MIN_SERIES_LENGTH = 2

def calcular_tamano_bloque(num_tiendas: int, num_articulos: int) -> tuple:
    """
    Calcula el tamaño óptimo de los bloques basado en la relación tiendas/artículos
    """
    # Calcular la relación tiendas/artículos
    ratio = num_tiendas / num_articulos if num_articulos > 0 else float('inf')
    
    # Ajustar el tamaño del bloque según la relación
    if ratio > 10:  # Muchas tiendas, pocos artículos
        # Ajustar el tamaño del bloque para optimizar el uso de núcleos
        tiendas_por_bloque = max(1, num_tiendas // (NUM_WORKERS * 2))
        return (tiendas_por_bloque, 1)  # (tiendas_por_bloque, articulos_por_bloque)
    elif ratio < 0.1:  # Pocas tiendas, muchos artículos
        # Ajustar el tamaño del bloque para optimizar el uso de núcleos
        articulos_por_bloque = max(1, num_articulos // (NUM_WORKERS * 2))
        return (1, articulos_por_bloque)  # (tiendas_por_bloque, articulos_por_bloque)
    else:  # Relación balanceada
        return (50, 50)  # (tiendas_por_bloque, articulos_por_bloque)

async def calcular_proyeccion(datos_ventas: List[DatoVentaDiaria], by_store: bool = True) -> ProyeccionOutput:
    """
    Servicio optimizado para calcular la proyección de ventas y stock recomendado
    con procesamiento por bloques dinámicos
    """
    try:
        # Verificar memoria disponible
        memoria_disponible_mb = psutil.virtual_memory().available / (1024 * 1024)
        if memoria_disponible_mb < MIN_MEMORY_REQUIRED_MB:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Memoria insuficiente para procesar la solicitud"
            )

        if not datos_ventas:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se proporcionaron datos de ventas"
            )

        # Convertir datos a DataFrame
        df = pd.DataFrame({
            'store_id': np.array([d.store_id for d in datos_ventas]),
            'art_codigo': np.array([d.art_codigo for d in datos_ventas]),
            'ds': pd.to_datetime([d.ds for d in datos_ventas]),
            'y': np.array([d.y for d in datos_ventas], dtype=np.float64)
        })

        # Obtener número único de tiendas y artículos
        num_tiendas = df['store_id'].nunique()
        num_articulos = df['art_codigo'].nunique()

        # Calcular tamaño óptimo de bloques
        tiendas_por_bloque, articulos_por_bloque = calcular_tamano_bloque(num_tiendas, num_articulos)

        # Crear bloques de datos
        bloques = []
        if by_store:
            # Agrupar por tienda
            tiendas_grupos = list(df.groupby('store_id'))
            # Dividir tiendas en bloques más pequeños
            for i in range(0, len(tiendas_grupos), tiendas_por_bloque):
                bloque_tiendas = tiendas_grupos[i:i + tiendas_por_bloque]
                articulos_bloque = []
                for store_id, grupo_tienda in bloque_tiendas:
                    for art_codigo, grupo_producto in grupo_tienda.groupby('art_codigo'):
                        if len(grupo_producto) >= MIN_SERIES_LENGTH:
                            articulos_bloque.append({
                                "store_id": store_id,
                                "art_codigo": art_codigo,
                                "ds": grupo_producto['ds'].tolist(),
                                "y": grupo_producto['y'].tolist()
                            })
                if articulos_bloque:
                    bloques.append(articulos_bloque)
        else:
            # Agrupar por artículo
            articulos_grupos = list(df.groupby('art_codigo'))
            # Dividir artículos en bloques más pequeños
            for i in range(0, len(articulos_grupos), articulos_por_bloque):
                bloque_articulos = articulos_grupos[i:i + articulos_por_bloque]
                tiendas_bloque = []
                for art_codigo, grupo_articulo in bloque_articulos:
                    for store_id, grupo_tienda in grupo_articulo.groupby('store_id'):
                        if len(grupo_tienda) >= MIN_SERIES_LENGTH:
                            tiendas_bloque.append({
                                "store_id": store_id,
                                "art_codigo": art_codigo,
                                "ds": grupo_tienda['ds'].tolist(),
                                "y": grupo_tienda['y'].tolist()
                            })
                if tiendas_bloque:
                    bloques.append(tiendas_bloque)

        # Procesar bloques en paralelo
        resultados_totales = []
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for bloque in bloques:
                future = executor.submit(
                    ProyeccionDAO.obtener_proyeccion_sync,
                    bloque,
                    by_store
                )
                futures.append(future)

            # Recolectar resultados
            for future in as_completed(futures):
                try:
                    resultados_bloque = future.result()
                    if resultados_bloque:
                        resultados_totales.extend(resultados_bloque)
                except Exception as e:
                    logger.error(f"Error procesando bloque: {str(e)}")
                    continue
                finally:
                    gc.collect()

        if not resultados_totales:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No se pudo procesar ninguna serie temporal"
            )

        # Formatear resultados
        resultados_formateados = [
            ResultadoProyeccion(
                id_sucursal=r["id_sucursal"],
                art_codigo=r["art_codigo"],
                demanda_pronosticada_7d=r["demanda_pronosticada_7d"],
                demanda_pronosticada_30d=r["demanda_pronosticada_30d"],
                stock_seguridad_7d=r["stock_seguridad_7d"],
                stock_seguridad_30d=r["stock_seguridad_30d"],
                stock_recomendado_7d=r["stock_recomendado_7d"],
                stock_recomendado_30d=r["stock_recomendado_30d"],
                intervalo_confianza_inferior=r["intervalo_confianza_inferior"],
                intervalo_confianza_superior=r["intervalo_confianza_superior"],
                tendencia=r["tendencia"]
            )
            for r in resultados_totales
        ]

        return ProyeccionOutput(
            resultados=resultados_formateados,
            fecha_calculo=datetime.now(),
            mensaje=f"Proyección calculada exitosamente para {len(resultados_formateados)} series temporales"
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error al calcular proyección: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al calcular proyección: {str(e)}"
        )
    finally:
        gc.collect()

def generar_reporte_final(resultados: Dict, errores: List[str], tiempo_total: float) -> str:
    """Genera un reporte final con estadísticas de rendimiento"""
    report_lines = [
        "REPORTE DE RENDIMIENTO - PROYECCIÓN DE VENTAS",
        "===========================================\n",
        f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tiempo total de procesamiento: {tiempo_total:.2f} segundos",
        f"Número de artículos procesados: {len(resultados)}",
        f"Número de errores: {len(errores)}",
        "\nEstadísticas de rendimiento:",
        f"Tiempo promedio por artículo: {tiempo_total/len(resultados):.2f} segundos",
        f"Artículos por segundo: {len(resultados)/tiempo_total:.2f}",
        "\nDistribución de modelos usados:"
    ]
    
    # Contar modelos usados
    modelos_usados = defaultdict(int)
    for result in resultados.values():
        modelos_usados[result['insights']['modelo_usado']] += 1
    
    for modelo, count in modelos_usados.items():
        report_lines.append(f"- {modelo}: {count} artículos")
    
    if errores:
        report_lines.extend([
            "\nErrores encontrados:",
            *[f"- {error}" for error in errores]
        ])
    
    return "\n".join(report_lines) 
