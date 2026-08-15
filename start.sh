#!/bin/sh
set -e

# inference.py does `from models.resnet18 import get_model`, so uvicorn
# needs to run with backend/ as its working directory, same as
# backend/Dockerfile's CMD.
(cd backend && uvicorn main:app --host 0.0.0.0 --port 8000) &

# CORS/XSRF protection breaks the file uploader when the app is embedded in
# the iframe HF Spaces serves it through, so both are disabled here.
exec streamlit run frontend/app.py \
    --server.address=0.0.0.0 \
    --server.port=7860 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
