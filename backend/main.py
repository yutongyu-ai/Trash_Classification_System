import io
import time
from contextlib import asynccontextmanager

from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from inference import predict, load_model

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_DIMENSION = 4096  # px, per side — plenty for a phone photo


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load at startup so a broken checkpoint fails loudly here, not on request.
    load_model()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "API running"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"error": "File must be an image"}

    start_time = time.perf_counter()

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return {"error": f"File too large: max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"}

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image file: {e}"}

    if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
        return {
            "error": f"Image dimensions too large: max {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}px"
        }

    result = predict(image)

    latency = (time.perf_counter() - start_time) * 1000

    return {
        **result,
        "latency_ms": round(latency, 2)
    }
