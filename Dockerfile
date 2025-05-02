# Usa Python 3.11 para que haya wheels disponibles
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Copia solo requirements y actualiza pip antes de instalar
COPY requirements.txt .
# Copia los archivos y carpetas necesarios al contenedor
COPY ./app /app/app
COPY ./main.py /app/main.py

# 1. Instala build-tools (si lo necesitas)
RUN apt-get update \
 && apt-get install -y build-essential libatlas-base-dev gfortran \
 && rm -rf /var/lib/apt/lists/*

# 2. Actualiza pip y empaquetadores
RUN pip install --upgrade pip setuptools wheel

# Instala las dependencias de Python (si tienes un archivo requirements.txt)
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Exponer el puerto en el que correrá la aplicación (por defecto FastAPI corre en el puerto 80)
EXPOSE 80

# Comando para ejecutar la aplicación (ajusta según tu aplicación)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
