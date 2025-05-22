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
import json

# Configurar logger
logger = logging.getLogger(__name__)

# Constantes para optimización
MAX_MEMORY_PERCENT = 90  # Porcentaje máximo de memoria a utilizar
MIN_MEMORY_REQUIRED_MB = 1024  # Memoria mínima requerida en MB
NUM_WORKERS = psutil.cpu_count(logical=False)  # Número de workers (uno por núcleo físico)
print(f"Número de workers (núcleos físicos): {NUM_WORKERS}")
MIN_SERIES_LENGTH = 2

async def calcular_proyeccion(datos_ventas: List[DatoVentaDiaria], by_store: bool = True, parametros_especiales: Optional[List[Dict]] = None) -> ProyeccionOutput:
    """
    Servicio optimizado para calcular la proyección de ventas y stock recomendado
    El batching óptimo se maneja en la capa DAO para mejor distribución de CPU
    """
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"INICIO DEL SERVICIO: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
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

        # Convertir parámetros especiales a diccionario si es necesario
        if parametros_especiales:
            parametros_especiales = [param.dict() if hasattr(param, 'dict') else param for param in parametros_especiales]

        # Convertir datos a DataFrame
        df = pd.DataFrame({
            'store_id': np.array([d.store_id for d in datos_ventas]),
            'art_codigo': np.array([d.art_codigo for d in datos_ventas]),
            'ds': pd.to_datetime([d.ds for d in datos_ventas]),
            'y': np.array([d.y for d in datos_ventas], dtype=np.float64)
        })

        # Si by_store es False, agrupar por artículo y fecha, sumando las ventas
        if not by_store:
            df = df.groupby(['art_codigo', 'ds'])['y'].sum().reset_index()
            df['store_id'] = 'global'  # Asignar store_id global

        # Obtener número único de tiendas y artículos
        num_tiendas = df['store_id'].nunique()
        num_articulos = df['art_codigo'].nunique()

        # Validar límite de artículos
        if num_articulos > 2500:
            logger.warning(f"Se excedió el límite de artículos. Artículos recibidos: {num_articulos}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Límite de artículos excedido. Se recibieron {num_articulos} artículos, el máximo permitido es 2500"
            )

        logger.info(f"Procesando {num_tiendas} tiendas y {num_articulos} artículos")

        # Preparar las series temporales para el DAO
        series_temporales = []
        
        if by_store:
            # Agrupar por tienda y artículo
            for (store_id, art_codigo), grupo in df.groupby(['store_id', 'art_codigo']):
                if len(grupo) >= MIN_SERIES_LENGTH:
                    # Buscar parámetros especiales para este artículo/tienda
                    parametros = None
                    if parametros_especiales:
                        for param in parametros_especiales:
                            if param.get('art_codigo') == art_codigo and param.get('store_id') == store_id:
                                parametros = param
                                break
                    
                    serie = {
                        "store_id": store_id,
                        "art_codigo": art_codigo,
                        "ds": grupo['ds'].tolist(),
                        "y": grupo['y'].tolist()
                    }
                    if parametros:
                        serie["parametros_especiales"] = parametros
                    
                    series_temporales.append(serie)
        else:
            # Agrupar solo por artículo
            for art_codigo, grupo in df.groupby('art_codigo'):
                if len(grupo) >= MIN_SERIES_LENGTH:
                    # Buscar parámetros especiales para este artículo
                    parametros = None
                    if parametros_especiales:
                        for param in parametros_especiales:
                            if param.get('art_codigo') == art_codigo:
                                parametros = param
                                break
                    
                    serie = {
                        "store_id": "global",
                        "art_codigo": art_codigo,
                        "ds": grupo['ds'].tolist(),
                        "y": grupo['y'].tolist()
                    }
                    if parametros:
                        serie["parametros_especiales"] = parametros
                    
                    series_temporales.append(serie)

        logger.info(f"Preparadas {len(series_temporales)} series temporales para procesar")

        if not series_temporales:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No hay series temporales con suficientes datos para procesar"
            )

        # Delegar el procesamiento al DAO con su estrategia de batching optimizada
        resultados_totales = await ProyeccionDAO.obtener_proyeccion(
            datos_ventas=series_temporales,
            by_store=by_store
        )

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

        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"FIN DEL SERVICIO: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tiempo total del servicio: {total_time:.2f} segundos")
        print(f"Series procesadas: {len(resultados_formateados)}")
        print(f"Series por segundo: {len(resultados_formateados)/total_time:.2f}")
        print(f"{'='*60}\n")

        return ProyeccionOutput(
            resultados=resultados_formateados,
            fecha_calculo=datetime.now(),
            mensaje=f"Proyección calculada exitosamente para {len(resultados_formateados)} series temporales"
        )

    except HTTPException as he:
        end_time = time.time()
        total_time = end_time - start_time
        logger.error(f"Error HTTP en proyección: {str(he.detail)}")
        print(f"\n{'='*60}")
        print(f"ERROR EN SERVICIO: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tiempo hasta el error: {total_time:.2f} segundos")
        print(f"{'='*60}\n")
        raise he
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        logger.error(f"Error al calcular proyección: {str(e)}")
        print(f"\n{'='*60}")
        print(f"ERROR EN SERVICIO: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tiempo hasta el error: {total_time:.2f} segundos")
        print(f"{'='*60}\n")
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
