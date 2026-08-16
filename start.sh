#!/bin/sh
set -e

# uvicorn needs backend/ as its working directory (sibling-module imports).
(cd backend && uvicorn main:app --host 0.0.0.0 --port 8000) &

# Disabled: breaks the file uploader when embedded in an iframe (HF Spaces).
exec streamlit run frontend/app.py \
    --server.address=0.0.0.0 \
    --server.port=7860 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
