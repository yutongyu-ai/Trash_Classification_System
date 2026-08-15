FROM python:3.12-slim

WORKDIR /app

# Combined backend + frontend dependencies for the single-container deployment
# used by Hugging Face Spaces. backend/Dockerfile and frontend/Dockerfile are
# unrelated and stay in use for the two-container docker-compose setup.
COPY backend/requirements.txt backend-requirements.txt
COPY frontend/requirements.txt frontend-requirements.txt
RUN pip install --no-cache-dir -r backend-requirements.txt -r frontend-requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY start.sh .
RUN chmod +x start.sh

# Frontend talks to the backend over localhost since both processes run in
# the same container here (docker-compose overrides this to the "backend"
# service hostname for the two-container setup).
ENV BACKEND_URL=http://localhost:8000/predict

# HF Spaces' Docker SDK expects the app on 7860 by default.
EXPOSE 7860

CMD ["./start.sh"]
