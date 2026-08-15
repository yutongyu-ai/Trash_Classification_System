import io
from unittest.mock import Mock

from fastapi.testclient import TestClient
from PIL import Image

import main as backend_main


def _fake_image_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _make_client(monkeypatch, predict_return=None):
    # Avoid any real model loading / HF Hub download during unit tests.
    monkeypatch.setattr(backend_main, "load_model", Mock())
    fake_predict = Mock(return_value=predict_return)
    monkeypatch.setattr(backend_main, "predict", fake_predict)
    return TestClient(backend_main.app), fake_predict


def test_root_health_check(monkeypatch):
    monkeypatch.setattr(backend_main, "load_model", Mock())
    client = TestClient(backend_main.app)
    with client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API running"}


def test_predict_returns_class_and_confidence(monkeypatch):
    client, _ = _make_client(
        monkeypatch, predict_return={"class": "cardboard", "confidence": 0.97}
    )
    with client:
        response = client.post(
            "/predict",
            files={"file": ("image.png", _fake_image_bytes(), "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["class"] == "cardboard"
    assert body["confidence"] == 0.97
    assert "latency_ms" in body


def test_predict_rejects_non_image_file(monkeypatch):
    client, fake_predict = _make_client(monkeypatch)
    with client:
        response = client.post(
            "/predict",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
    assert response.status_code == 200
    assert "error" in response.json()
    fake_predict.assert_not_called()


def test_predict_rejects_corrupted_image_bytes(monkeypatch):
    client, fake_predict = _make_client(monkeypatch)
    with client:
        response = client.post(
            "/predict",
            files={"file": ("image.png", b"not-a-real-png", "image/png")},
        )
    assert response.status_code == 200
    assert "error" in response.json()
    fake_predict.assert_not_called()
