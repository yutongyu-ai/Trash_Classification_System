# ♻️ Trash Classification System

[![CI](https://github.com/yutongyu-ai/Trash_Classification_System/actions/workflows/ci.yml/badge.svg)](https://github.com/yutongyu-ai/Trash_Classification_System/actions/workflows/ci.yml)

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
| Best Validation Accuracy | 94.2% (epoch 28/30) |
| Test Accuracy | 93.7% (359/383 images) |
| Inference Latency | ~60 ms |

## Per-Class Results (test set, 383 images)

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| cardboard | 61 | 98.3% | 96.7% | 97.5% |
| glass | 76 | 92.3% | 94.7% | 93.5% |
| metal | 62 | 88.2% | 96.8% | 92.3% |
| paper | 90 | 98.8% | 93.3% | 95.9% |
| plastic | 73 | 93.0% | 90.4% | 91.7% |
| trash | 21 | 85.7% | 85.7% | 85.7% |

`trash` is the weakest class on both precision and recall — it also has by
far the fewest test examples (21, vs. 61-90 for the others). TrashNet's
class imbalance is handled during training via inverse-frequency class
weighting (`utils/datasets.py::get_class_weights`), which helps the
training signal, but 21 test images is still a small sample to generalize
from — more `trash`-class data would likely help more than further loss
reweighting at this point.

![Confusion matrix](docs/eval/trashnet_confusion_matrix.png)

## Training Curves

Train accuracy climbs to ~99.5% while validation accuracy plateaus around
92-94% from epoch ~10 onward — the growing train/val gap and validation
loss flattening (while training loss keeps dropping) are signs of
overfitting on this small (~2,500 image) dataset, despite the
augmentation (random flip/rotation) and dropout (p=0.5) already in the
training pipeline. The checkpoint actually shipped is whichever epoch had
the best validation accuracy (epoch 28 here), not the final epoch, so this
doesn't directly hurt the deployed model's accuracy — but it's a signal
that stronger augmentation, partial backbone freezing, or more data would
likely generalize better than training longer.

| Accuracy | Loss |
|---|---|
| ![Accuracy curve](docs/eval/trashnet_acc.png) | ![Loss curve](docs/eval/trashnet_loss.png) |

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

## Model Weights

No manual setup needed — on first startup, `backend` automatically downloads
the trained checkpoint from
[Hugging Face Hub](https://huggingface.co/tonghahaha/trashnet-resnet18) if
`checkpoints/best_resnet18_trashnet.pth` isn't already present locally (e.g.
from running your own training via `train_hpo.slurm` / `main.py`). A local
checkpoint always takes priority over the hosted one.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yutongyu-ai/Trash_Classification_System.git
cd Trash_Classification_System
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
│   ├── models/               # Model architecture and utilities
│   ├── Dockerfile
│   ├── inference.py          # Inference pipeline (loads from HF Hub if no local checkpoint)
│   ├── main.py               # FastAPI application
│   └── requirements.txt
│
├── frontend/
│   ├── app.py                # Streamlit frontend
│   ├── Dockerfile
│   └── requirements.txt
│
├── tests/                     # pytest suite (unit + integration tests)
├── docs/eval/                 # Evaluation plots referenced in this README
├── checkpoints/               # Model checkpoints (used for both training output and inference)
├── data/                      # Dataset directory
├── outputs/                   # Training outputs and logs
├── utils/                     # Utility functions
│
├── .github/workflows/ci.yml   # Lint, tests, Docker build check
├── docker-compose.yml
├── hpo.py                     # Hyperparameter optimization
├── train.py                   # Training script
├── train_hpo.slurm            # SLURM job spec for HPC training
├── main.py
├── README.md
```
