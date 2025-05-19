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
NUM_WORKERS = multiprocessing.cpu_count() # Número de workers (uno por núcleo físico)
BATCH_SIZE = 50  # Tamaño del lote para procesamiento
CHUNK_SIZE = 10  # Tamaño del chunk para el ProcessPoolExecutor
MIN_SERIES_LENGTH = 2  # Mínimo de puntos de datos para considerar una serie válida

def calcular_proyeccion(datos: List[DatoVentaDiaria], by_store: bool = True, 
                       nivel_servicio: float = 0.95, manejar_atipicos: bool = True,
                       umbral_atipicos: float = 3.0, lead_time: int = 1) -> ResultadoProyeccion:
    """
    Función optimizada para calcular proyecciones de ventas
    """
    tiempo_inicio = time.time()
    
    try:
        # Verificar memoria disponible
        memoria_disponible = psutil.virtual_memory().available / (1024 * 1024)  # MB
        if memoria_disponible < MIN_MEMORY_REQUIRED_MB:
            raise MemoryError(f"Memoria insuficiente. Disponible: {memoria_disponible:.2f}MB, Requerida: {MIN_MEMORY_REQUIRED_MB}MB")
        
        # Convertir datos a DataFrame de manera eficiente
        df = pd.DataFrame([{
            'store_id': d.store_id,
            'art_codigo': d.art_codigo,
            'ds': pd.to_datetime(d.ds),
            'y': float(d.y)
        } for d in datos])
        
        # Agrupar datos por artículo
        grupos_articulos = df.groupby('art_codigo')
        articulos_procesar = []
        
        # Filtrar artículos con suficientes datos
        for art_codigo, grupo in grupos_articulos:
            if len(grupo) >= 7:  # Mínimo 7 días de datos
                articulos_procesar.append(art_codigo)
        
        if not articulos_procesar:
            raise ValueError("No hay artículos con suficientes datos para procesar")
        
        # Preparar lotes de artículos
        lotes = [articulos_procesar[i:i + BATCH_SIZE] 
                for i in range(0, len(articulos_procesar), BATCH_SIZE)]
        
        resultados = {}
        errores = []
        
        # Procesar lotes en paralelo
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            
            for lote in lotes:
                # Filtrar datos para el lote actual
                df_lote = df[df['art_codigo'].isin(lote)]
                
                # Crear tarea para el lote
                future = executor.submit(
                    run_forecast,
                    df_lote,
                    by_store,
                    nivel_servicio,
                    manejar_atipicos,
                    umbral_atipicos,
                    lead_time
                )
                futures.append(future)
            
            # Recolectar resultados
            for future in as_completed(futures):
                try:
                    result_json, report = future.result()
                    result_dict = json.loads(result_json)
                    resultados.update(result_dict)
                except Exception as e:
                    logger.error(f"Error procesando lote: {str(e)}")
                    errores.append(str(e))
        
        # Calcular tiempo total
        tiempo_total = time.time() - tiempo_inicio
        
        # Crear resultado final
        return ResultadoProyeccion(
            resultados=resultados,
            reporte=generar_reporte_final(resultados, errores, tiempo_total),
            tiempo_procesamiento=tiempo_total,
            num_articulos_procesados=len(articulos_procesar),
            num_errores=len(errores)
        )
        
    except Exception as e:
        logger.error(f"Error en cálculo de proyección: {str(e)}")
        raise

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

async def calcular_proyeccion(datos_ventas: List[DatoVentaDiaria], by_store: bool = True) -> ProyeccionOutput:
    """
    Servicio optimizado para calcular la proyección de ventas y stock recomendado
    Ahora los workers procesan por tienda (store_id), no por producto.
    """
    try:
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
        try:
            df = pd.DataFrame({
                'store_id': np.array([d.store_id for d in datos_ventas]),
                'art_codigo': np.array([d.art_codigo for d in datos_ventas]),
                'ds': pd.to_datetime([d.ds for d in datos_ventas]),
                'y': np.array([d.y for d in datos_ventas], dtype=np.float64)
            })
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error al procesar datos de entrada: {str(e)}"
            )

        # Agrupar por tienda (store_id)
        tiendas_por_lote = defaultdict(list)
        for store_id, grupo_tienda in df.groupby('store_id'):
            # Para cada tienda, agrupar por producto
            productos = []
            for art_codigo, grupo_producto in grupo_tienda.groupby('art_codigo'):
                if len(grupo_producto) >= MIN_SERIES_LENGTH:
                    productos.append({
                        "store_id": store_id,
                        "art_codigo": art_codigo,
                        "ds": grupo_producto['ds'].tolist(),
                        "y": grupo_producto['y'].tolist()
                    })
            if productos:
                tiendas_por_lote[store_id] = productos

        # Procesar lotes de tiendas en paralelo
        resultados_totales = []
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for store_id, productos in tiendas_por_lote.items():
                # Cada worker recibe todos los productos de una tienda
                future = executor.submit(
                    ProyeccionDAO.obtener_proyeccion_sync,  # Debe ser función síncrona
                    productos,
                    by_store
                )
                futures.append((store_id, future))
            for store_id, future in futures:
                try:
                    resultados_tienda = future.result()
                    if resultados_tienda:
                        resultados_totales.extend(resultados_tienda)
                except Exception as e:
                    logger.error(f"Error procesando tienda {store_id}: {str(e)}")
                    continue
                gc.collect()

        if not resultados_totales:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No se pudo procesar ninguna serie temporal"
            )
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
