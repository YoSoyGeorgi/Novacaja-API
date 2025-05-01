import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import json
from scipy import stats

def preparar_datos_para_clave(df, clave):
    """
    Filtra y prepara datos para una clave específica de un DataFrame
    
    Parámetros:
    -----------
    df : DataFrame
        DataFrame de entrada con columnas [store_id, art_codigo, ds, y]
    clave : tuple
        La clave (store_id, art_codigo) o art_codigo para filtrar
        
    Retorna:
    --------
    DataFrame con datos diarios de ventas para la clave específica
    """
    try:
        # Filtrar para la clave específica
        if isinstance(clave, tuple):
            df_clave = df[(df['store_id'] == clave[0]) & (df['art_codigo'] == clave[1])].copy()
        else:
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

def run_forecast(input_df, by_store=True, nivel_servicio=0.95, manejar_atipicos=True, umbral_atipicos=3.0, lead_time=1):
    """
    Función simplificada de pronóstico que toma un DataFrame con store_id, art_codigo, ds, y
    y retorna pronósticos con intervalos de confianza y recomendaciones de stock.
    
    Parámetros:
    -----------
    input_df : DataFrame
        DataFrame de entrada con columnas [store_id, art_codigo, ds, y]
    by_store : bool
        Si es True, pronostica para cada combinación tienda/producto
        Si es False, pronostica para cada producto en todas las tiendas
    nivel_servicio : float
        Nivel de servicio deseado para recomendaciones de stock (0.95 para 95% de confianza)
    manejar_atipicos : bool
        Si se deben manejar valores atípicos en los datos
    umbral_atipicos : float
        Umbral para detección de atípicos
    lead_time : int
        Tiempo de entrega en días para cálculo de stock de seguridad
        
    Retorna:
    --------
    Tuple de (json_results, report_text)
    """
    # Validar DataFrame de entrada
    required_columns = ['store_id', 'art_codigo', 'ds', 'y']
    if not all(col in input_df.columns for col in required_columns):
        raise ValueError(f"El DataFrame de entrada debe contener las columnas: {required_columns}")
    
    # Convertir ds a datetime si no lo es
    input_df['ds'] = pd.to_datetime(input_df['ds'])
    
    # Agrupar datos según el parámetro by_store
    if by_store:
        group_cols = ['store_id', 'art_codigo']
    else:
        group_cols = ['art_codigo']
    
    results = {}
    report_lines = []
    
    for group_key, group_data in input_df.groupby(group_cols):
        # Preparar datos para Prophet
        df = preparar_datos_para_clave(group_data, group_key)
        
        # Preprocesar datos (manejar atípicos)
        df = preprocesar_datos(df, manejar_atipicos, umbral_atipicos)
        
        # Crear y ajustar modelo
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.001,
            seasonality_prior_scale=10.0,
            seasonality_mode='multiplicative',
            holidays_prior_scale=0.01
        )
        
        # Agregar feriados mexicanos para ajuste consistente
        model.add_country_holidays(country_name='MX')
        
        model.fit(df)
        
        # Generar fechas futuras para 30 días
        future_dates = model.make_future_dataframe(periods=30)
        forecast = model.predict(future_dates)
        
        # Obtener última fecha en datos de entrenamiento
        last_date = df['ds'].max()
        
        # Calcular demanda pronosticada (asegurar que sea positiva)
        pronostico_7d = max(0, forecast[forecast['ds'] > last_date].head(7)['yhat'].sum())
        pronostico_30d = max(0, forecast[forecast['ds'] > last_date]['yhat'].sum())
        
        # Calcular stock de seguridad
        stock_seg_7d = calcular_stock_seguridad(forecast[forecast['ds'] > last_date].head(7), nivel_servicio, lead_time)
        stock_seg_30d = calcular_stock_seguridad(forecast[forecast['ds'] > last_date], nivel_servicio, lead_time)
        
        # Calcular niveles de stock recomendados (asegurar que sean positivos)
        stock_rec_7d = max(0, pronostico_7d + stock_seg_7d)
        stock_rec_30d = max(0, pronostico_30d + stock_seg_30d)
        
        # Obtener intervalos de confianza (asegurar que sean positivos)
        ci_95 = forecast[forecast['ds'] > last_date][['yhat_lower', 'yhat_upper']].values
        ci_95 = np.maximum(0, ci_95)  # Asegurar valores positivos
        
        # Almacenar resultados
        key = '_'.join(str(k) for k in group_key) if isinstance(group_key, tuple) else str(group_key)
        results[key] = {
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
                'tendencia': 'creciente' if forecast['trend'].iloc[-1] > forecast['trend'].iloc[0] else 'decreciente',
                'estacionalidad_semanal': float(forecast['weekly'].iloc[-1]),
                'estacionalidad_anual': float(forecast['yearly'].iloc[-1]),
                'nivel_servicio': nivel_servicio
            }
        }
        
        # Agregar al reporte
        report_lines.append(f"\nAnálisis para {'Tienda ' + str(group_key[0]) + ' - ' if by_store else ''}Producto {group_key[-1]}:")
        report_lines.append(f"Demanda Pronosticada:")
        report_lines.append(f"  7 días: {int(np.ceil(pronostico_7d))} unidades")
        report_lines.append(f"  30 días: {int(np.ceil(pronostico_30d))} unidades")
        report_lines.append(f"Stock de Seguridad (al {nivel_servicio*100}% de nivel de servicio):")
        report_lines.append(f"  7 días: {int(np.ceil(stock_seg_7d))} unidades")
        report_lines.append(f"  30 días: {int(np.ceil(stock_seg_30d))} unidades")
        report_lines.append(f"Niveles de Stock Recomendados:")
        report_lines.append(f"  7 días: {int(np.ceil(stock_rec_7d))} unidades")
        report_lines.append(f"  30 días: {int(np.ceil(stock_rec_30d))} unidades")
        report_lines.append(f"Intervalo de Confianza 95%: [{int(np.ceil(ci_95[:, 0].mean()))}, {int(np.ceil(ci_95[:, 1].mean()))}]")
        report_lines.append(f"Tendencia: {results[key]['insights']['tendencia']}")
    
    # Generar reporte final
    report_text = "INFORME DE PLANIFICACIÓN DE INVENTARIO - NOVACAJA\n"
    report_text += "===========================================\n\n"
    report_text += f"Análisis realizado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_text += f"Nivel de pronóstico: {'Tienda-Producto' if by_store else 'Producto'}\n"
    report_text += f"Nivel de servicio: {nivel_servicio*100}%\n"
    report_text += f"Número de pronósticos: {len(results)}\n\n"
    report_text += "\n".join(report_lines)
    report_text += "\n\n"
    report_text += "Generado por el Sistema de Pronóstico de Ventas de Novacaja\n"
    report_text += "Para más información y servicios, visite www.novacaja.com"
    
    return json.dumps(results, indent=2), report_text

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
        by_store=True, 
        nivel_servicio=0.95,
        manejar_atipicos=True,
        umbral_atipicos=3.0,
        lead_time=1
    )
    
    print("Resultados JSON:")
    print(json_results)
    print("\nReporte:")
    print(report) 