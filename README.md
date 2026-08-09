# ♻️ Trash Classification System

An end-to-end deep learning application for real-time waste image classification using **ResNet18**, **FastAPI**, **Streamlit**, and **Docker**.

The system classifies waste images into six categories from the TrashNet dataset and provides an interactive web interface for inference. The project is fully containerized for portable and scalable deployment.

---

# 🚀 Features

- Fine-tuned **ResNet18** model for 6-class waste classification
- Real-time inference with FastAPI backend
- Interactive Streamlit frontend
- Dockerized full-stack deployment
- Low-latency prediction pipeline (~60ms steady-state latency)
- Clean modular project structure
- Portable and reproducible environment

---

# 🧠 Model Performance

| Metric | Result |
|---|---|
| Model Architecture | ResNet18 |
| Dataset | TrashNet |
| Classes | 6 |
| Validation Accuracy | ~95% |
| Inference Latency | ~60 ms |

---

# 🗂️ Waste Categories

The model classifies images into the following categories:

- 📦 Cardboard
- 🍾 Glass
- 🥫 Metal
- 📄 Paper
- 🧴 Plastic
- 🗑️ Trash

---

# 🏗️ System Architecture

```text
             ┌────────────────────┐
             │     Streamlit      │
             │   Frontend UI      │
             └─────────┬──────────┘
                       │ HTTP Request
                       ▼
             ┌────────────────────┐
             │      FastAPI       │
             │   Inference API    │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │    ResNet18 Model  │
             │   PyTorch Runtime  │
             └────────────────────┘
```

---

# 🛠️ Tech Stack

## Backend
- Python
- FastAPI
- PyTorch
- Torchvision
- Pillow
- Uvicorn

## Frontend
- Streamlit
- Requests

## Deployment
- Docker
- Docker Compose

---

# 📦 Dockerized Deployment

The application is fully containerized using Docker Compose.

## Services

| Service | Description |
|---|---|
| `backend` | FastAPI inference server |
| `frontend` | Streamlit web application |

The `backend` service runs CPU-only inference by default. For a single-image
ResNet18 forward pass, CPU latency is generally fine for interactive use; GPU
acceleration mainly pays off for batch training/inference, not this service.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/trash-classification-system.git
cd trash-classification-system
```

---

## 2. Build and Run with Docker

```bash
docker compose up --build
```

The application will be available at:

| Service | URL |
|---|---|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000 |

---

# 🧪 Running Without Docker

## Backend

```bash
cd backend

pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Frontend

```bash
cd frontend

pip install -r requirements.txt

streamlit run app.py
```

---

# 🧠 Training Pipeline

The model training pipeline includes:

- Data augmentation
- Transfer learning with ResNet18
- Hyperparameter optimization
- Validation monitoring
- Checkpoint saving

---

# 📁 Project Structure

```text
project/
│
├── backend/
│   ├── checkpoints/          # Model checkpoints used for inference
│   ├── models/               # Model architecture and utilities
│   ├── Dockerfile
│   ├── inference.py          # Inference pipeline
│   ├── main.py               # FastAPI application
│   └── requirements.txt
│
├── frontend/
│   ├── app.py                # Streamlit frontend
│   ├── Dockerfile
│   └── requirements.txt
│
├── configs/                  # Training and experiment configs
├── data/                     # Dataset directory
├── outputs/                  # Training outputs and logs
├── utils/                    # Utility functions
│
├── .dockerignore
├── docker-compose.yml
├── hpo.py                    # Hyperparameter optimization
├── train.py                  # Training script
├── main.py 
├── README.md
```
