FROM python:3.12-slim

WORKDIR /app

# Install CUDA 12.1 build of torch/torchvision explicitly, matching the
# driver's max supported CUDA version. Using --index-url (not
# --extra-index-url) here pins these two packages to the PyTorch cu121
# index exclusively, so pip can't silently resolve a newer CUDA build
# from the default PyPI index.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY . .

EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]