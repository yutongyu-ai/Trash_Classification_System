import sys
import time
from pathlib import Path

# Streamlit Community Cloud runs a single process, so this calls
# inference.predict() directly instead of over HTTP (frontend/app.py is the
# HTTP version — see README's "Live Demo"). Both share frontend/ui.py.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "frontend"))
from inference import predict  # noqa: E402
from ui import render_app  # noqa: E402


def predict_in_process(image):
    start_time = time.perf_counter()
    result = predict(image)
    result["latency_ms"] = (time.perf_counter() - start_time) * 1000
    return result


render_app(predict_in_process)
