# Usa Python 3.11 para que haya wheels disponibles
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Copia solo requirements y actualiza pip antes de instalar
COPY requirements.txt .
# Copia los archivos y carpetas necesarios al contenedor
COPY ./app /app/app
COPY ./main.py /app/main.py
COPY /app/README.md /app/README.md
COPY ./startup.sh /app/startup.sh

# 1. Instala build-tools (si lo necesitas)
RUN apt-get update \
 && apt-get install -y build-essential libatlas-base-dev gfortran \
 && rm -rf /var/lib/apt/lists/*

# 2. Actualiza pip y empaquetadores
RUN pip install --upgrade pip setuptools wheel

# Configurar variables de entorno para Prophet/Stan
ENV STAN_BACKEND=CMDSTANPY
ENV CMDSTAN_NO_BOOST=1
ENV STAN_THREADS=1

# Crear directorio para archivos temporales de Stan con permisos correctos
RUN mkdir -p /tmp/prophet_stan && chmod -R 777 /tmp/prophet_stan
ENV TMPDIR=/tmp/prophet_stan

# Instala las dependencias de Python (si tienes un archivo requirements.txt)
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Hacer el script de inicio ejecutable
RUN chmod +x /app/startup.sh

# Exponer el puerto en el que correrá la aplicación (por defecto FastAPI corre en el puerto 80)
EXPOSE 80

# Comando para ejecutar la aplicación (ajusta según tu aplicación)
CMD ["/app/startup.sh"]
