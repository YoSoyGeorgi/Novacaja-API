#!/bin/bash

# Ensure Stan temporary directory exists with proper permissions
mkdir -p /tmp/prophet_stan
chmod -R 777 /tmp/prophet_stan

# Export Stan environment variables
export STAN_BACKEND=CMDSTANPY
export CMDSTAN_NO_BOOST=1
export STAN_THREADS=1
export TMPDIR=/tmp/prophet_stan

# Start the application
exec uvicorn main:app --host 0.0.0.0 --port 80 --workers 1 