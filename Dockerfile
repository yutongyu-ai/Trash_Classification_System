FROM python:3.12-slim

WORKDIR /app

# Single-container build (Cloud Run, HF Spaces on a paid plan — see README).
# backend/Dockerfile and frontend/Dockerfile still cover docker-compose.
COPY backend/requirements.txt backend-requirements.txt
COPY frontend/requirements.txt frontend-requirements.txt
RUN pip install --no-cache-dir -r backend-requirements.txt -r frontend-requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY start.sh .
RUN chmod +x start.sh

# Both processes share this container, so localhost works here.
ENV BACKEND_URL=http://localhost:8000/predict

# HF Spaces' Docker SDK expects the app on 7860 by default.
EXPOSE 7860

CMD ["./start.sh"]
