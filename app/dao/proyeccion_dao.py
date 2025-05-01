import pandas as pd
import json
from app.dao.modelo import run_forecast
from datetime import date, timedelta
from typing import List, Dict, Any
import numpy as np

class ProyeccionDAO:
    @staticmethod
    async def obtener_proyeccion(datos_ventas: list):
        """
        Obtiene la proyección utilizando el modelo de pronóstico
        
        Parámetros:
        -----------
        datos_ventas : list
            Lista de datos de ventas diarias
            
        Retorna:
        --------
        List[Dict[str, Any]] - Lista de resultados de proyección
        """
        try:
            # Convertir los datos de ventas a DataFrame
            if not datos_ventas:
                raise ValueError("Datos de ventas no proporcionados")
            
            df = pd.DataFrame(datos_ventas)
            
            # Asegurar que los datos tienen las columnas necesarias
            required_columns = ['store_id', 'art_codigo', 'ds', 'y']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Los datos deben contener las columnas: {required_columns}")
            
            # Convertir fechas a datetime
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Asegurar que "y" (cantidad_articulos) sea numérico
            df['y'] = pd.to_numeric(df['y'])
            
            if df.empty:
                raise ValueError("No hay datos para procesar")
            
            # Verificar si tenemos suficientes datos para Prophet (al menos 2 fechas diferentes)
            fechas_unicas = df['ds'].nunique()
            
            # Si no hay suficientes datos para Prophet (menos de 2 fechas)
            if fechas_unicas < 2:
                # Generar datos sintéticos basados en los datos proporcionados
                return await ProyeccionDAO._generar_proyeccion_simple(df)
            
            # Si hay suficientes datos, usar Prophet
            # Ejecutar el pronóstico con valores predeterminados
            json_results, report_text = run_forecast(
                input_df=df,
                by_store=True,  # Analizar por sucursal por defecto
                nivel_servicio=0.95,
                manejar_atipicos=True,
                umbral_atipicos=3.0,
                lead_time=1
            )
            
            # Convertir resultados JSON a diccionario Python
            resultados = json.loads(json_results) if isinstance(json_results, str) else json_results
            
            # Transformar resultados al formato de salida
            resultados_formateados = []
            
            for key, data in resultados.items():
                # Extraer id_sucursal y art_codigo de la clave
                if '_' in key:
                    id_sucursal, art_codigo = key.split('_', 1)
                else:
                    id_sucursal = "desconocida"
                    art_codigo = key
                
                # Crear objeto de resultado
                resultado = {
                    "id_sucursal": id_sucursal,
                    "art_codigo": art_codigo,
                    "demanda_pronosticada_7d": data["demanda_pronosticada"]["7d"],
                    "demanda_pronosticada_30d": data["demanda_pronosticada"]["30d"],
                    "stock_seguridad_7d": data["stock_seguridad"]["7d"],
                    "stock_seguridad_30d": data["stock_seguridad"]["30d"],
                    "stock_recomendado_7d": data["stock_recomendado"]["7d"],
                    "stock_recomendado_30d": data["stock_recomendado"]["30d"],
                    "intervalo_confianza_inferior": data["intervalos_confianza"]["95"]["inferior"],
                    "intervalo_confianza_superior": data["intervalos_confianza"]["95"]["superior"],
                    "tendencia": data["insights"]["tendencia"]
                }
                
                resultados_formateados.append(resultado)
            
            return resultados_formateados
            
        except Exception as e:
            raise Exception(f"Error al obtener la proyección: {str(e)}")
    
    @staticmethod
    async def _generar_proyeccion_simple(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Genera una proyección simple cuando hay muy pocos datos para usar Prophet
        
        Esta función toma el promedio de los datos disponibles y genera una proyección sencilla
        basada en ese promedio, con un stock de seguridad calculado simplemente.
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos limitados
            
        Retorna:
        --------
        List[Dict[str, Any]] - Lista de resultados simplificados
        """
        resultados = []
        
        # Agrupar por store_id y art_codigo
        grupos = df.groupby(['store_id', 'art_codigo'])
        
        # Factor de confianza para nivel de servicio 0.95
        z_score = 1.645
        
        for group_key, group_data in grupos:
            # Calcular el valor medio diario
            valor_medio = group_data['y'].mean()
            desv_est = group_data['y'].std() if len(group_data) > 1 else valor_medio * 0.2  # Usar 20% como estimación si solo hay 1 dato
            
            # Calcular demanda pronosticada simple
            demanda_7d = round(valor_medio * 7)
            demanda_30d = round(valor_medio * 30)
            
            # Calcular stock de seguridad simple
            stock_seg_7d = round(z_score * desv_est * np.sqrt(7))
            stock_seg_30d = round(z_score * desv_est * np.sqrt(30))
            
            # Calcular stock recomendado
            stock_rec_7d = demanda_7d + stock_seg_7d
            stock_rec_30d = demanda_30d + stock_seg_30d
            
            # Calcular intervalos de confianza
            intervalo_inf = max(0, round(valor_medio - z_score * desv_est))
            intervalo_sup = round(valor_medio + z_score * desv_est)
            
            # Extraer id_sucursal y art_codigo
            id_sucursal, art_codigo = group_key
            
            resultado = {
                "id_sucursal": id_sucursal,
                "art_codigo": art_codigo,
                "demanda_pronosticada_7d": demanda_7d,
                "demanda_pronosticada_30d": demanda_30d,
                "stock_seguridad_7d": stock_seg_7d,
                "stock_seguridad_30d": stock_seg_30d,
                "stock_recomendado_7d": stock_rec_7d,
                "stock_recomendado_30d": stock_rec_30d,
                "intervalo_confianza_inferior": intervalo_inf,
                "intervalo_confianza_superior": intervalo_sup,
                "tendencia": "estable"  # Suponemos tendencia estable con pocos datos
            }
            
            resultados.append(resultado)
        
        return resultados 