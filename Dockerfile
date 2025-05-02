# Usa Python 3.11 para que haya wheels disponibles
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Copia solo requirements y actualiza pip antes de instalar
COPY requirements.txt .

# 1. Instala build-tools (si lo necesitas)
RUN apt-get update \
 && apt-get install -y build-essential libatlas-base-dev gfortran \
 && rm -rf /var/lib/apt/lists/*

# 2. Actualiza pip y empaquetadores
RUN pip install --upgrade pip setuptools wheel

# 3. Instala dependencias desde wheels
RUN pip install -r requirements.txt

# Copia el resto del código
COPY . .

# Por ejemplo, arranca uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
