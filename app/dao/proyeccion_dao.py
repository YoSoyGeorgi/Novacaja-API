import pandas as pd
import json
from app.dao.modelo import run_forecast
from datetime import date, timedelta
from typing import List, Dict, Any
import numpy as np
import asyncio
from concurrent.futures import ProcessPoolExecutor
import logging
from functools import partial
import multiprocessing

# Configurar logger
logger = logging.getLogger(__name__)

# Constantes para optimización
MAX_WORKERS = multiprocessing.cpu_count()  # Usar todos los núcleos disponibles
BATCH_SIZE = 1000  # Tamaño de lote para procesamiento en paralelo

class ProyeccionDAO:
    @staticmethod
    async def obtener_proyeccion(datos_ventas: list, by_store: bool = True):
        """
        Obtiene la proyección utilizando el modelo de pronóstico con procesamiento optimizado
        
        Parámetros:
        -----------
        datos_ventas : list
            Lista de datos de ventas diarias, donde cada elemento es un diccionario con:
            - store_id: str
            - art_codigo: str
            - ds: List[datetime] - Lista completa de fechas para la serie
            - y: List[float] - Lista completa de valores para la serie
        by_store : bool
            Indica si el pronóstico debe realizarse por tienda
            
        Retorna:
        --------
        List[Dict[str, Any]] - Lista de resultados de proyección
        """
        try:
            if not datos_ventas:
                raise ValueError("Datos de ventas no proporcionados")
            
            # Dividir los datos en lotes más grandes para procesamiento en paralelo
            batches = [datos_ventas[i:i + BATCH_SIZE] for i in range(0, len(datos_ventas), BATCH_SIZE)]
            
            # Crear un pool de procesos para procesamiento paralelo
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Crear tareas para cada lote
                loop = asyncio.get_event_loop()
                tasks = []
                
                for batch in batches:
                    # Crear tarea para procesar el lote
                    task = loop.run_in_executor(
                        executor,
                        partial(
                            ProyeccionDAO._procesar_lote,
                            batch=batch,
                            by_store=by_store
                        )
                    )
                    tasks.append(task)
                
                # Esperar a que todas las tareas terminen
                resultados_batches = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Combinar resultados de todos los lotes
                resultados_validos = []
                for batch_result in resultados_batches:
                    if isinstance(batch_result, Exception):
                        logger.error(f"Error procesando lote: {str(batch_result)}")
                        continue
                    if batch_result:
                        resultados_validos.extend(batch_result)
                
                return resultados_validos
            
        except Exception as e:
            logger.error(f"Error al obtener la proyección: {str(e)}")
            raise Exception(f"Error al obtener la proyección: {str(e)}")
    
    @staticmethod
    def _procesar_lote(batch: List[Dict], by_store: bool) -> List[Dict[str, Any]]:
        """
        Procesa un lote de series temporales en paralelo
        """
        resultados = []
        for serie in batch:
            try:
                # Crear DataFrame para la serie temporal
                df = pd.DataFrame({
                    'ds': pd.to_datetime(serie['ds']),
                    'y': pd.to_numeric(serie['y'])
                })
                
                resultado = ProyeccionDAO._procesar_serie(
                    df=df,
                    store_id=serie['store_id'],
                    art_codigo=serie['art_codigo'],
                    by_store=by_store
                )
                
                if resultado is not None:
                    resultados.append(resultado)
                    
            except Exception as e:
                logger.error(f"Error procesando serie {serie.get('store_id')}-{serie.get('art_codigo')}: {str(e)}")
                continue
                
        return resultados
    
    @staticmethod
    def _procesar_serie(df: pd.DataFrame, store_id: str, art_codigo: str, by_store: bool) -> Dict[str, Any]:
        """
        Procesa una serie temporal individual
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de la serie temporal (ds, y)
        store_id : str
            ID de la tienda
        art_codigo : str
            Código del artículo
        by_store : bool
            Indica si el pronóstico debe realizarse por tienda
            
        Retorna:
        --------
        Dict[str, Any] - Resultado del procesamiento de la serie
        """
        try:
            # Verificar si hay suficientes datos para Prophet
            if df['ds'].nunique() < 2:
                return ProyeccionDAO._generar_proyeccion_simple(df, store_id, art_codigo, by_store)
            
            # Crear DataFrame con el formato requerido por run_forecast
            df_forecast = pd.DataFrame({
                'store_id': [store_id] * len(df),
                'art_codigo': [art_codigo] * len(df),
                'ds': df['ds'],
                'y': df['y']
            })
            
            # Ejecutar pronóstico con valores predeterminados
            json_results, _ = run_forecast(
                input_df=df_forecast,
                by_store=by_store,
                nivel_servicio=0.95,
                manejar_atipicos=True,
                umbral_atipicos=3.0,
                lead_time=1
            )
            
            # Convertir resultados JSON a diccionario Python
            resultados = json.loads(json_results) if isinstance(json_results, str) else json_results
            
            # Obtener la clave correcta del resultado
            key = f"{store_id}_{art_codigo}" if by_store else art_codigo
            data = resultados.get(key, {})
            
            if not data:
                logger.warning(f"No se encontraron resultados para la serie {store_id}-{art_codigo}")
                return None
            
            # Crear objeto de resultado
            return {
                "id_sucursal": store_id if by_store else "global",
                "art_codigo": art_codigo,
                "demanda_pronosticada_7d": data.get("demanda_pronosticada", {}).get("7d", 0),
                "demanda_pronosticada_30d": data.get("demanda_pronosticada", {}).get("30d", 0),
                "stock_seguridad_7d": data.get("stock_seguridad", {}).get("7d", 0),
                "stock_seguridad_30d": data.get("stock_seguridad", {}).get("30d", 0),
                "stock_recomendado_7d": data.get("stock_recomendado", {}).get("7d", 0),
                "stock_recomendado_30d": data.get("stock_recomendado", {}).get("30d", 0),
                "intervalo_confianza_inferior": data.get("intervalos_confianza", {}).get("95", {}).get("inferior", 0),
                "intervalo_confianza_superior": data.get("intervalos_confianza", {}).get("95", {}).get("superior", 0),
                "tendencia": data.get("insights", {}).get("tendencia", "estable")
            }
            
        except Exception as e:
            logger.error(f"Error procesando serie {store_id}-{art_codigo}: {str(e)}")
            raise
    
    @staticmethod
    def _generar_proyeccion_simple(df: pd.DataFrame, store_id: str, art_codigo: str, by_store: bool) -> Dict[str, Any]:
        """
        Genera una proyección simple cuando hay muy pocos datos
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de la serie temporal
        store_id : str
            ID de la tienda
        art_codigo : str
            Código del artículo
        by_store : bool
            Indica si el pronóstico debe realizarse por tienda
            
        Retorna:
        --------
        Dict[str, Any] - Resultado de la proyección simple
        """
        # Calcular estadísticas básicas
        valor_medio = df['y'].mean()
        desv_est = df['y'].std() if len(df) > 1 else valor_medio * 0.2
        
        # Factor de confianza para nivel de servicio 0.95
        z_score = 1.645
        
        # Calcular proyecciones
        demanda_7d = round(valor_medio * 7)
        demanda_30d = round(valor_medio * 30)
        
        # Calcular stock de seguridad
        stock_seg_7d = round(z_score * desv_est * np.sqrt(7))
        stock_seg_30d = round(z_score * desv_est * np.sqrt(30))
        
        # Calcular stock recomendado
        stock_rec_7d = demanda_7d + stock_seg_7d
        stock_rec_30d = demanda_30d + stock_seg_30d
        
        # Calcular intervalos de confianza
        intervalo_inf = max(0, round(valor_medio - z_score * desv_est))
        intervalo_sup = round(valor_medio + z_score * desv_est)
        
        return {
            "id_sucursal": store_id if by_store else "global",
            "art_codigo": art_codigo,
            "demanda_pronosticada_7d": demanda_7d,
            "demanda_pronosticada_30d": demanda_30d,
            "stock_seguridad_7d": stock_seg_7d,
            "stock_seguridad_30d": stock_seg_30d,
            "stock_recomendado_7d": stock_rec_7d,
            "stock_recomendado_30d": stock_rec_30d,
            "intervalo_confianza_inferior": intervalo_inf,
            "intervalo_confianza_superior": intervalo_sup,
            "tendencia": "estable"
        } 