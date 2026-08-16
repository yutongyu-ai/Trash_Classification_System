import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import inference
import main as backend_main


@pytest.mark.integration
def test_predict_end_to_end_downloads_real_model():
    """No monkeypatching — downloads the real checkpoint from HF Hub (if no
    local checkpoint is present) and runs a real forward pass. Marked
    `integration` so a temporary HF Hub outage doesn't block a PR."""
    inference._model = None

    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(120, 80, 40)).save(buf, format="PNG")

    client = TestClient(backend_main.app)
    with client:
        response = client.post(
            "/predict",
            files={"file": ("image.png", buf.getvalue(), "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["class"] in inference.CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
