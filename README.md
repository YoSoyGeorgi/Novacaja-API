# API de Proyección de Ventas

Esta API permite calcular proyecciones de ventas y recomendaciones de stock utilizando un modelo de serie temporal.

## Estructura del Proyecto

La aplicación sigue una arquitectura orientada a servicios con tres capas principales:

- **DTO (Data Transfer Objects)**: Define las estructuras de datos para la entrada y salida de la API.
- **Services**: Contiene la lógica de negocio que conecta los endpoints con los DAOs.
- **DAO (Data Access Objects)**: Maneja la interacción con el modelo de pronóstico.

## Endpoints

### Verificar Estado

```
GET /health
```

Retorna el estado de la API.

### Calcular Proyección

```
POST /proyeccion
```

Calcula una proyección de ventas y recomendación de stock basada en datos históricos.

#### Cuerpo de la Solicitud

```json
{
  "store_id": "string",
  "art_codigo": "string",
  "datos_ventas": [
    {
      "store_id": "string",
      "art_codigo": "string",
      "ds": "YYYY-MM-DD",
      "y": 0
    }
  ],
  "by_store": true,
  "nivel_servicio": 0.95,
  "manejar_atipicos": true,
  "umbral_atipicos": 3.0,
  "lead_time": 1
}
```

#### Respuesta

```json
{
  "store_id": "string",
  "art_codigo": "string",
  "forecast_values": [
    {
      "ds": "YYYY-MM-DD",
      "yhat": 0,
      "yhat_lower": 0,
      "yhat_upper": 0
    }
  ],
  "stock_recomendado": 0,
  "mensaje": "string"
}
```

### Obtener README

```
GET /readme
```

Retorna el contenido de este archivo README.

## Cómo Ejecutar

1. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```

2. Iniciar el servidor:
   ```
   uvicorn main:app --reload
   ```

3. Acceder a la documentación de la API:
   ```
   http://localhost:8000/docs
   ```

# Sistema de Pronóstico de Ventas y Planificación de Inventario - NOVACAJA

Este sistema proporciona pronósticos de ventas y recomendaciones de inventario utilizando modelos de series temporales avanzados.

## Conceptos Técnicos Estadísticos

### Modelo de Pronóstico (Prophet)

El sistema utiliza **Prophet**, un modelo de series temporales desarrollado por Facebook que descompone una serie temporal en:

- **Tendencia**: El componente a largo plazo que indica si las ventas están creciendo o decreciendo.
- **Estacionalidad**: Patrones recurrentes en diferentes escalas temporales:
  - Estacionalidad anual: Variaciones en las ventas a lo largo del año.
  - Estacionalidad semanal: Patrones de comportamiento dentro de la semana.
- **Efectos de días festivos**: Impacto específico de fechas como días festivos nacionales o promociones.

### Parámetros del Modelo

- **changepoint_prior_scale (0.001)**: Controla la flexibilidad de la tendencia. Un valor bajo hace que la tendencia sea más rígida.
- **seasonality_prior_scale (10.0)**: Controla la fuerza de la estacionalidad. Un valor alto permite que el modelo capture patrones estacionales fuertes.
- **seasonality_mode ('multiplicative')**: Define cómo la estacionalidad interactúa con la tendencia. En modo multiplicativo, la amplitud de la estacionalidad aumenta con el nivel de la serie.
- **holidays_prior_scale (0.01)**: Controla cuánto influyen los días festivos en el pronóstico.

### Días Festivos Mexicanos

El modelo incluye automáticamente los días festivos de México (`model.add_country_holidays(country_name='MX')`) para mejorar la precisión de los pronósticos. Esto incluye:

- **Días festivos oficiales**: Como el Día de la Independencia (16 de septiembre), Día de la Revolución (20 de noviembre), etc.
- **Festivos importantes para ventas**: Como El Buen Fin (equivalente mexicano al Black Friday).
- **Períodos vacacionales**: Como Semana Santa y temporada navideña.

El parámetro `holidays_prior_scale` (0.01) está configurado para dar un peso equilibrado a estos eventos, permitiendo que influyan en el pronóstico sin dominar otros patrones.

### Manejo de Valores Atípicos

El sistema incluye métodos para detectar y manejar valores atípicos (outliers) que podrían distorsionar el pronóstico:

- **Método Z-score**: Detecta valores que están a cierta cantidad de desviaciones estándar de la media.
- **Método IQR (Rango Intercuartil)**: Detecta valores fuera de los límites definidos por los cuartiles.
- **Estrategias de manejo**:
  - **Cap (Recorte)**: Limita los valores atípicos a un umbral máximo o mínimo.
  - **Remove (Eliminación)**: Excluye completamente los valores atípicos.
  - **Impute (Imputación)**: Reemplaza los valores atípicos con valores típicos.

### Cálculo del Stock de Seguridad

El stock de seguridad se calcula considerando:

1. **Desviación estándar de la demanda**: Extraída de los intervalos de confianza del pronóstico.
2. **Nivel de servicio**: Determina cuánta protección contra incertidumbre se desea.
   - 90% = z-score de 1.282
   - 95% = z-score de 1.645
   - 99% = z-score de 2.326
3. **Tiempo de entrega (lead time)**: Cuánto tiempo toma reabastecerse.

La fórmula utilizada es:
- SS = z × σ × √(lead_time)

Donde:
- z = z-score correspondiente al nivel de servicio
- σ = desviación estándar de la demanda
- lead_time = tiempo de entrega en días

### Intervalos de Confianza

Los intervalos de confianza del 95% indican el rango dentro del cual se espera que caigan las ventas reales con una probabilidad del 95%.

### Stock Recomendado

Se calcula como la suma de:
- La demanda pronosticada para el período.
- El stock de seguridad calculado para el mismo período.

## Interpretación de Resultados

### Entendiendo los Pronósticos

El sistema genera pronósticos para períodos de 7 y 30 días, junto con recomendaciones de stock. Para interpretar correctamente estos resultados:

- **Demanda Pronosticada**: Es la cantidad que se espera vender en el período.
  - Si la demanda es alta y creciente (como S1_P1), indica productos populares en crecimiento.
  - Si la demanda es estable (como S1_P2), indica productos con ventas consistentes.
  - Si la demanda está disminuyendo (como S2_P1), puede indicar productos al final de su ciclo de vida.
  - Una demanda de cero (como S2_P1 a 30 días) sugiere discontinuar el producto o investigar problemas potenciales.

- **Stock de Seguridad**: 
  - Un valor alto en proporción a la demanda indica alta variabilidad o incertidumbre.
  - Un valor bajo indica demanda predecible.
  - El stock de seguridad varía según el nivel de servicio elegido (90%, 95%, 99%).

- **Stock Recomendado**: 
  - Es su guía principal para decisiones de compra/producción.
  - Incluye tanto la demanda esperada como el margen de seguridad necesario.

### Indicadores de Tendencia y Estacionalidad

- **Tendencia**: "creciente" o "decreciente" indica la dirección general del producto.
- **Estacionalidad semanal**: Valores positivos o negativos indican qué días de la semana tienen más o menos ventas que el promedio.
- **Estacionalidad anual**: Indica la variación estacional actual (mes actual respecto al promedio anual).

### Efecto de los Días Festivos

- Los días festivos aparecerán como picos o valles en el pronóstico, según el patrón histórico.
- Para fechas cercanas a festividades importantes como Navidad o El Buen Fin, espere:
  - Mayor demanda pronosticada
  - Posiblemente mayor stock de seguridad debido a la volatilidad
  - Recomendaciones de stock más altas para esas fechas

## Limitaciones y Consideraciones

- Los pronósticos son más precisos a corto plazo que a largo plazo.
- Eventos no históricos (como nuevas promociones) no son automáticamente considerados.
- La calidad del pronóstico depende de la calidad y cantidad de datos históricos.
- Se recomienda revisar y ajustar manualmente las recomendaciones para productos críticos o nuevos.
- Los días festivos nuevos o que cambian de fecha cada año pueden requerir ajustes manuales adicionales.

---

© 2025 Novacaja. Todos los derechos reservados.

Este software y su documentación son propiedad de Novacaja. Está prohibida su reproducción, distribución o modificación sin autorización expresa por escrito de Novacaja. 