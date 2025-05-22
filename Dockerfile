# Usa Python 3.8 para mejor compatibilidad con pystan 2.19.1.1
FROM python:3.8-slim

# Directorio de trabajo
WORKDIR /app

ENV PROPHET_BACKEND PYSTAN
ENV TMPDIR /tmp

# Copia solo requirements y actualiza pip antes de instalar
COPY requirements.txt .
# Copia los archivos y carpetas necesarios al contenedor
COPY ./app /app/app
COPY ./main.py /app/main.py
COPY /app/README.md /app/README.md

# 1. Instala build-tools (si lo necesitas)
RUN apt-get update \
 && apt-get install -y build-essential libatlas-base-dev gfortran gcc g++ python3-dev \
 && rm -rf /var/lib/apt/lists/*

# 2. Actualiza pip y empaquetadores
RUN pip install --upgrade pip setuptools wheel

# Instala dependencias de compilación para pystan
RUN pip install --no-cache-dir Cython==0.29.24 numpy==1.21.6

# Instala pystan primero por separado para asegurar compilación correcta
RUN pip install --no-cache-dir pystan==2.19.1.1

# Instala las dependencias de Python (si tienes un archivo requirements.txt)
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Exponer el puerto en el que correrá la aplicación (por defecto FastAPI corre en el puerto 80)
EXPOSE 80

# Comando para ejecutar la aplicación (ajusta según tu aplicación)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
