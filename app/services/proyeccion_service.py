from app.dao.proyeccion_dao import ProyeccionDAO
from app.dto.proyeccion_dto import ProyeccionInput, ProyeccionOutput, ResultadoProyeccion, DatoVentaDiaria
from datetime import datetime
from typing import Dict, Any, List, Union
from fastapi import HTTPException, status
import logging

# Configurar logger
logger = logging.getLogger(__name__)

async def calcular_proyeccion(datos_ventas: List[DatoVentaDiaria], by_store: bool = True) -> ProyeccionOutput:
    """
    Servicio para calcular la proyección de ventas y stock recomendado
    
    Parámetros:
    -----------
    datos_ventas : List[DatoVentaDiaria]
        Lista de datos de ventas diarias
    by_store : bool
        Indica si el pronóstico debe realizarse por tienda
        
    Retorna:
    --------
    ProyeccionOutput
        Resultado de la proyección con valores de pronóstico y stock recomendado
    """
    try:
        # Convertir datos_ventas a lista de diccionarios
        datos_ventas_list = [dato.dict() for dato in datos_ventas]
        
        # Obtener resultados del DAO
        resultados = await ProyeccionDAO.obtener_proyeccion(
            datos_ventas=datos_ventas_list,
            by_store=by_store
        )
        
        # Convertir resultados al formato del DTO
        resultados_formateados = []
        for resultado in resultados:
            resultados_formateados.append(
                ResultadoProyeccion(
                    id_sucursal=resultado["id_sucursal"],
                    art_codigo=resultado["art_codigo"],
                    demanda_pronosticada_7d=resultado["demanda_pronosticada_7d"],
                    demanda_pronosticada_30d=resultado["demanda_pronosticada_30d"],
                    stock_seguridad_7d=resultado["stock_seguridad_7d"],
                    stock_seguridad_30d=resultado["stock_seguridad_30d"],
                    stock_recomendado_7d=resultado["stock_recomendado_7d"],
                    stock_recomendado_30d=resultado["stock_recomendado_30d"],
                    intervalo_confianza_inferior=resultado["intervalo_confianza_inferior"],
                    intervalo_confianza_superior=resultado["intervalo_confianza_superior"],
                    tendencia=resultado["tendencia"]
                )
            )
        
        # Crear respuesta final
        return ProyeccionOutput(
            resultados=resultados_formateados,
            fecha_calculo=datetime.now(),
            mensaje="Proyección calculada exitosamente"
        )
        
    except Exception as e:
        logger.error(f"Error al calcular proyección: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al calcular proyección: {str(e)}"
        ) 