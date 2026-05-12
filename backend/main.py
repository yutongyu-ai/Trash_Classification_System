import io
import time
from PIL import Image
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from inference import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API running"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return {"error": "File must be an image"}

    start_time = time.perf_counter()

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image file: {e}"}

    result = predict(image)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = (time.perf_counter() - start_time) * 1000

    return {
        **result,
        "latency_ms": round(latency, 2)
    }









