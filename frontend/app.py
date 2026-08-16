import os

import requests

from ui import image_to_png_bytes, render_app

# docker-compose overrides this to the "backend" service hostname.
API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000/predict")


def predict_via_api(image):
    files = {"file": ("image.png", image_to_png_bytes(image), "image/png")}
    response = requests.post(API_URL, files=files)
    if response.status_code != 200:
        return {"error": f"Request failed with status code: {response.status_code}"}
    return response.json()


render_app(predict_via_api)
