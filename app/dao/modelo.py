import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import json
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import Tuple, Dict, Any, Optional
import logging
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

# Constantes para optimización
UNCERTAINTY_SAMPLES = 30  # Reducido para mejor rendimiento
MIN_POINTS_FOR_SEASONALITY = 90  # Mínimo de puntos para considerar estacionalidad
CACHE_SIZE = 1000  # Tamaño del caché para resultados frecuentes

@lru_cache(maxsize=CACHE_SIZE)
def get_cached_forecast(art_codigo: str, store_id: str, data_hash: str) -> Optional[Dict]:
    """
    Caché para resultados de pronósticos frecuentes
    """
    return None

def preparar_datos_para_clave(df, clave):
    """
    Filtra y prepara datos para una clave específica de un DataFrame
    
    Parámetros:
    -----------
    df : DataFrame
        DataFrame de entrada con columnas [store_id, art_codigo, ds, y]
    clave : tuple o str
        La clave (store_id, art_codigo) o art_codigo para filtrar
        
    Retorna:
    --------
    DataFrame con datos diarios de ventas para la clave específica
    """
    try:
        # Filtrar para la clave específica
        if isinstance(clave, tuple):
            # Caso cuando by_store es True
            df_clave = df[(df['store_id'] == clave[0]) & (df['art_codigo'] == clave[1])].copy()
        else:
            # Caso cuando by_store es False
            df_clave = df[df['art_codigo'] == clave].copy()
        
        # Asegurar que ds es datetime
        df_clave['ds'] = pd.to_datetime(df_clave['ds'])
        
        # Ordenar por fecha
        df_clave = df_clave.sort_values('ds')
        
        # Mantener solo las columnas requeridas para Prophet
        df_ventas_diarias = df_clave[['ds', 'y']].copy()
        
        return df_ventas_diarias
        
    except Exception as e:
        raise ValueError(f"Error preparando datos para la clave {clave}: {str(e)}")

def manejar_atipicos(df, columna='y', metodo='zscore', umbral=3.0, estrategia='cap'):
    """
    Detecta y maneja valores atípicos en los datos
    
    Parámetros:
    -----------
    df : DataFrame
        DataFrame de entrada
    columna : str
        Columna a verificar para valores atípicos
    metodo : str
        Método para detectar atípicos ('zscore' o 'iqr')
    umbral : float
        Umbral para detección de atípicos
    estrategia : str
        Cómo manejar atípicos ('cap', 'remove', o 'impute')
        
    Retorna:
    --------
    DataFrame con valores atípicos manejados
    """
    # Crear una copia del DataFrame
    df_procesado = df.copy()
    
    # Detectar atípicos
    if metodo == 'zscore':
        z_scores = np.abs(stats.zscore(df_procesado[columna], nan_policy='omit'))
        mascara_atipicos = z_scores > umbral
    elif metodo == 'iqr':
        Q1 = df_procesado[columna].quantile(0.25)
        Q3 = df_procesado[columna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - umbral * IQR
        limite_superior = Q3 + umbral * IQR
        mascara_atipicos = (df_procesado[columna] < limite_inferior) | (df_procesado[columna] > limite_superior)
    else:
        raise ValueError(f"Método de detección de atípicos desconocido: {metodo}")
    
    cantidad_atipicos = mascara_atipicos.sum()
    indices_atipicos = df_procesado.index[mascara_atipicos].tolist()
    
    # Manejar atípicos según la estrategia
    if estrategia == 'cap':
        if metodo == 'zscore':
            # Calcular media y desviación estándar sin atípicos
            media = df_procesado.loc[~mascara_atipicos, columna].mean()
            desv_est = df_procesado.loc[~mascara_atipicos, columna].std()
            # Limitar atípicos
            df_procesado.loc[mascara_atipicos, columna] = df_procesado.loc[mascara_atipicos, columna].apply(
                lambda x: media + umbral * desv_est if x > media else media - umbral * desv_est
            )
        elif metodo == 'iqr':
            # Limitar atípicos
            df_procesado.loc[df_procesado[columna] > limite_superior, columna] = limite_superior
            df_procesado.loc[df_procesado[columna] < limite_inferior, columna] = limite_inferior
    
    elif estrategia == 'remove':
        df_procesado = df_procesado.loc[~mascara_atipicos]
    
    elif estrategia == 'impute':
        # Imputar con mediana móvil
        tamano_ventana = 5
        for idx in indices_atipicos:
            # Obtener índices de la ventana (manejar casos límite)
            inicio_idx = max(0, idx - tamano_ventana//2)
            fin_idx = min(len(df_procesado), idx + tamano_ventana//2 + 1)
            
            # Obtener valores no atípicos en la ventana
            valores_ventana = df_procesado.iloc[inicio_idx:fin_idx].loc[~mascara_atipicos[inicio_idx:fin_idx], columna]
            
            if len(valores_ventana) > 0:
                # Reemplazar con mediana de valores no atípicos en la ventana
                df_procesado.loc[idx, columna] = valores_ventana.median()
            else:
                # Si todos los valores en la ventana son atípicos, usar mediana global
                df_procesado.loc[idx, columna] = df_procesado.loc[~mascara_atipicos, columna].median()
    
    return df_procesado

def preprocesar_datos(df_ventas_diarias, manejar_atipicos_flag=True, umbral_atipicos=3.0):
    """
    Preprocesa los datos manejando valores atípicos si está habilitado
    
    Parámetros:
    -----------
    df_ventas_diarias : DataFrame
        Datos diarios de ventas con columnas 'ds' y 'y'
    manejar_atipicos_flag : bool
        Si se deben manejar valores atípicos
    umbral_atipicos : float
        Umbral para detección de atípicos
        
    Retorna:
    --------
    DataFrame preprocesado
    """
    # Manejar atípicos si está habilitado
    if manejar_atipicos_flag:
        df_ventas_diarias = manejar_atipicos(
            df_ventas_diarias, 
            columna='y', 
            metodo='zscore', 
            umbral=umbral_atipicos, 
            estrategia='cap'
        )
    
    return df_ventas_diarias

def calcular_stock_seguridad(forecast, nivel_servicio=0.95, lead_time=1):
    """
    Calcula el stock de seguridad basado en la incertidumbre del pronóstico
    
    Parámetros:
    -----------
    forecast : DataFrame
        DataFrame de pronóstico de Prophet
    nivel_servicio : float
        Nivel de servicio deseado (0.95 para 95% de confianza)
    lead_time : int
        Tiempo de entrega en días (por defecto 1 día)
        
    Retorna:
    --------
    float : Valor del stock de seguridad
    """
    # Obtener la desviación estándar del pronóstico
    desv_est = (forecast['yhat_upper'] - forecast['yhat_lower']) / (2 * 1.96)  # Convertir IC 95% a desv. est.
    
    # Calcular z-score basado en el nivel de servicio
    z_scores = {
        0.90: 1.282,  # 90% de confianza
        0.95: 1.645,  # 95% de confianza
        0.99: 2.326   # 99% de confianza
    }
    z_score = z_scores.get(nivel_servicio, 1.645)  # Usar 95% como valor por defecto
    
    # Calcular demanda promedio diaria (asegurar que sea positiva)
    demanda_promedio = max(0, forecast['yhat'].mean())
    
    # Calcular desviación estándar de la demanda
    desv_est_demanda = desv_est.mean()
    
    # Calcular stock de seguridad usando la fórmula:
    # SS = z * σ * √(lead_time)
    # donde:
    # z = z-score del nivel de servicio
    # σ = desviación estándar de la demanda
    # lead_time = tiempo de entrega en días
    stock_seguridad = z_score * desv_est_demanda * np.sqrt(lead_time)
    
    # Asegurar un stock de seguridad mínimo basado en la demanda promedio
    stock_minimo = demanda_promedio * 0.2  # 20% de la demanda promedio como mínimo
    
    # Asegurar un stock de seguridad máximo basado en la demanda promedio
    stock_maximo = demanda_promedio * 0.5  # 50% de la demanda promedio como máximo
    
    return min(max(stock_seguridad, stock_minimo), stock_maximo)

def downsampling_series(df: pd.DataFrame, min_points: int = 30) -> pd.DataFrame:
    """
    Realiza downsampling inteligente de la serie temporal
    """
    if len(df) < min_points:
        return df
    
    # Obtener la fecha de corte (último trimestre)
    fecha_corte = df['ds'].max() - pd.Timedelta(days=90)
    
    # Separar datos recientes y antiguos
    df_reciente = df[df['ds'] >= fecha_corte].copy()
    df_antiguo = df[df['ds'] < fecha_corte].copy()
    
    if len(df_antiguo) > 0:
        # Convertir datos antiguos a frecuencia semanal
        df_antiguo = df_antiguo.set_index('ds').resample('W').agg({
            'y': 'sum'
        }).reset_index()
    
    # Combinar datos
    return pd.concat([df_antiguo, df_reciente]).sort_values('ds')

def modelo_simple(df: pd.DataFrame, dias_pronostico: int = 30) -> Tuple[float, float, float, float]:
    """
    Modelo simple optimizado para series cortas
    """
    # Crear características de tiempo
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['y'].values
    
    # Ajustar modelo
    model = LinearRegression(n_jobs=-1)  # Usar todos los núcleos disponibles
    model.fit(X, y)
    
    # Predecir
    X_future = np.array(range(len(df), len(df) + dias_pronostico)).reshape(-1, 1)
    y_pred = model.predict(X_future)
    
    # Calcular estadísticas
    demanda_7d = max(0, np.sum(y_pred[:7]))
    demanda_30d = max(0, np.sum(y_pred))
    desv_est = np.std(df['y'])
    z_score = 1.645  # Para 95% de confianza
    
    return demanda_7d, demanda_30d, desv_est, z_score

def run_forecast(input_df: pd.DataFrame, by_store: bool = True, nivel_servicio: float = 0.95,
                manejar_atipicos: bool = True, umbral_atipicos: float = 3.0, lead_time: int = 1) -> Tuple[str, str]:
    """
    Función optimizada de pronóstico que implementa un modelo híbrido con caché
    """
    results = {}
    report_lines = []
    tiempos_procesamiento = {}
    
    # Agrupar datos según el parámetro by_store
    group_cols = ['store_id', 'art_codigo'] if by_store else ['art_codigo']
    
    for group_key, group_data in input_df.groupby(group_cols):
        try:
            tiempo_inicio = time.time()
            
            # Preparar datos
            df = preparar_datos_para_clave(group_data, group_key)
            
            # Verificar caché
            if isinstance(group_key, tuple):
                store_id, art_codigo = group_key
            else:
                store_id, art_codigo = "global", group_key
                
            data_hash = hash(str(df[['ds', 'y']].values.tobytes()))
            cached_result = get_cached_forecast(art_codigo, store_id, str(data_hash))
            
            if cached_result is not None:
                results['_'.join(str(k) for k in group_key)] = cached_result
                continue
            
            # Determinar el tipo de modelo a usar
            dias_datos = (df['ds'].max() - df['ds'].min()).days
            
            if dias_datos < 30:  # Serie muy corta
                # Usar modelo simple
                demanda_7d, demanda_30d, desv_est, z_score = modelo_simple(df)
                stock_seg_7d = z_score * desv_est * np.sqrt(7)
                stock_seg_30d = z_score * desv_est * np.sqrt(30)
                
            else:
                # Realizar downsampling inteligente
                df = downsampling_series(df)
                
                # Preprocesar datos
                df = preprocesar_datos(df, manejar_atipicos, umbral_atipicos)
                
                # Configurar modelo según la longitud de la serie
                if dias_datos < 180:  # Serie corta
                    model = Prophet(
                        yearly_seasonality=False,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.05,
                        seasonality_prior_scale=0.1,
                        seasonality_mode='additive',
                        uncertainty_samples=UNCERTAINTY_SAMPLES,
                        changepoint_range=0.8
                    )
                else:  # Serie larga
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.001,
                        seasonality_prior_scale=10.0,
                        seasonality_mode='additive',
                        uncertainty_samples=UNCERTAINTY_SAMPLES,
                        changepoint_range=0.8
                    )
                
                # Agregar feriados solo para series largas
                if dias_datos >= 180:
                    model.add_country_holidays(country_name='MX')
                
                # Ajustar modelo
                model.fit(df)
                
                # Generar pronóstico
                future_dates = model.make_future_dataframe(periods=30)
                forecast = model.predict(future_dates)
                
                # Obtener última fecha en datos de entrenamiento
                last_date = df['ds'].max()
                
                # Calcular métricas
                pronostico_7d = max(0, forecast[forecast['ds'] > last_date].head(7)['yhat'].sum())
                pronostico_30d = max(0, forecast[forecast['ds'] > last_date]['yhat'].sum())
                
                # Calcular stock de seguridad
                stock_seg_7d = calcular_stock_seguridad(forecast[forecast['ds'] > last_date].head(7), nivel_servicio, lead_time)
                stock_seg_30d = calcular_stock_seguridad(forecast[forecast['ds'] > last_date], nivel_servicio, lead_time)
            
            # Calcular niveles de stock recomendados
            stock_rec_7d = max(0, pronostico_7d + stock_seg_7d)
            stock_rec_30d = max(0, pronostico_30d + stock_seg_30d)
            
            # Obtener intervalos de confianza
            if dias_datos < 30:
                ci_95 = np.array([
                    [max(0, pronostico_7d - z_score * desv_est), max(0, pronostico_7d + z_score * desv_est)],
                    [max(0, pronostico_30d - z_score * desv_est), max(0, pronostico_30d + z_score * desv_est)]
                ])
            else:
                ci_95 = forecast[forecast['ds'] > last_date][['yhat_lower', 'yhat_upper']].values
                ci_95 = np.maximum(0, ci_95)
            
            # Almacenar resultados
            key = '_'.join(str(k) for k in group_key) if isinstance(group_key, tuple) else str(group_key)
            result = {
                'demanda_pronosticada': {
                    '7d': int(np.ceil(pronostico_7d)),
                    '30d': int(np.ceil(pronostico_30d))
                },
                'stock_seguridad': {
                    '7d': int(np.ceil(stock_seg_7d)),
                    '30d': int(np.ceil(stock_seg_30d))
                },
                'stock_recomendado': {
                    '7d': int(np.ceil(stock_rec_7d)),
                    '30d': int(np.ceil(stock_rec_30d))
                },
                'intervalos_confianza': {
                    '95': {
                        'inferior': int(np.ceil(ci_95[:, 0].mean())),
                        'superior': int(np.ceil(ci_95[:, 1].mean()))
                    }
                },
                'insights': {
                    'tendencia': 'creciente' if dias_datos >= 30 and forecast['trend'].iloc[-1] > forecast['trend'].iloc[0] else 'estable',
                    'modelo_usado': 'simple' if dias_datos < 30 else 'prophet_light' if dias_datos < 180 else 'prophet_full',
                    'nivel_servicio': nivel_servicio,
                    'tiempo_procesamiento': time.time() - tiempo_inicio
                }
            }
            
            results[key] = result
            tiempos_procesamiento[key] = time.time() - tiempo_inicio
            
        except Exception as e:
            logger.error(f"Error procesando serie {group_key}: {str(e)}")
            continue
    
    # Generar reporte con tiempos de procesamiento
    report_text = generar_reporte(results, by_store, nivel_servicio, tiempos_procesamiento)
    
    return json.dumps(results, indent=2), report_text

def generar_reporte(results: Dict, by_store: bool, nivel_servicio: float, tiempos_procesamiento: Dict) -> str:
    """Genera el reporte de resultados con información de rendimiento"""
    report_lines = [
        "INFORME DE PLANIFICACIÓN DE INVENTARIO - NOVACAJA",
        "===========================================\n",
        f"Análisis realizado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Nivel de pronóstico: {'Tienda-Producto' if by_store else 'Producto'}",
        f"Nivel de servicio: {nivel_servicio*100}%",
        f"Número de pronósticos: {len(results)}\n",
        "\nResumen de tiempos de procesamiento:",
        f"Tiempo promedio por serie: {np.mean(list(tiempos_procesamiento.values())):.2f} segundos",
        f"Tiempo total: {sum(tiempos_procesamiento.values()):.2f} segundos\n"
    ]
    
    for key, data in results.items():
        report_lines.extend([
            f"\nAnálisis para {key}:",
            f"Modelo usado: {data['insights']['modelo_usado']}",
            f"Tiempo de procesamiento: {data['insights']['tiempo_procesamiento']:.2f} segundos",
            f"Demanda Pronosticada:",
            f"  7 días: {data['demanda_pronosticada']['7d']} unidades",
            f"  30 días: {data['demanda_pronosticada']['30d']} unidades",
            f"Stock de Seguridad:",
            f"  7 días: {data['stock_seguridad']['7d']} unidades",
            f"  30 días: {data['stock_seguridad']['30d']} unidades",
            f"Niveles de Stock Recomendados:",
            f"  7 días: {data['stock_recomendado']['7d']} unidades",
            f"  30 días: {data['stock_recomendado']['30d']} unidades",
            f"Intervalo de Confianza 95%: [{data['intervalos_confianza']['95']['inferior']}, {data['intervalos_confianza']['95']['superior']}]",
            f"Tendencia: {data['insights']['tendencia']}"
        ])
    
    report_lines.extend([
        "\n\nGenerado por el Sistema de Pronóstico de Ventas de Novacaja",
        "Para más información y servicios, visite www.novacaja.com"
    ])
    
    return "\n".join(report_lines)

if __name__ == "__main__":
    # Ejemplo de uso
    dates = pd.date_range(start='2023-01-01', end='2023-12-31')
    stores = ['S1']
    products = ['P1']
    
    # Definir patrones base para cada tienda y producto
    store_patterns = {
        'S1': {'base': 100, 'trend': 0.5},  # Tienda 1 con tendencia creciente
    }
    
    product_patterns = {
        'P1': {'seasonality': 1.2, 'weekly': 1.5},  # Producto 1 con mayor estacionalidad
    }
    
    sample_data = []
    for store in stores:
        for product in products:
            for date in dates:
                # Obtener patrones base
                store_base = store_patterns[store]['base']
                store_trend = store_patterns[store]['trend']
                product_seasonality = product_patterns[product]['seasonality']
                product_weekly = product_patterns[product]['weekly']
                
                # Calcular ventas con variaciones
                base_sales = store_base + (date - dates[0]).days * store_trend
                monthly_variation = 20 * np.sin(2 * np.pi * date.month / 12)
                weekly_variation = 30 * product_weekly * (date.dayofweek == 5)  # Mayor venta los sábados
                random_noise = np.random.normal(0, 5)  # Ruido aleatorio
                
                # Combinar todos los efectos
                sales = (base_sales + monthly_variation + weekly_variation + random_noise) * product_seasonality
                
                # Asegurar que las ventas sean positivas
                sales = max(0, sales)
                
                # Agregar a los datos de muestra
                sample_data.append({
                    'store_id': store,
                    'art_codigo': product,
                    'ds': date,
                    'y': sales
                })
    
    sample_df = pd.DataFrame(sample_data)
    
    # Ejecutar pronóstico con 95% de nivel de servicio y manejo de atípicos
    json_results, report = run_forecast(
        sample_df, 
        by_store=False, 
        nivel_servicio=0.95,
        manejar_atipicos=True,
        umbral_atipicos=3.0,
        lead_time=1
    )
    
    print("Resultados JSON:")
    print(json_results)
    print("\nReporte:")
    print(report) 