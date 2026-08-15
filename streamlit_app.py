import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

# Streamlit Community Cloud runs a single Python process, so this calls
# backend/inference.py's predict() directly instead of over HTTP.
# frontend/app.py is the HTTP-based version used with docker-compose and
# local no-Docker dev — see README's "Live Demo" section.
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from inference import predict  # noqa: E402

CLASS_INFO = {
    "cardboard": "📦 Cardboard",
    "glass": "🍾 Glass",
    "metal": "🥫 Metal",
    "paper": "📄 Paper",
    "plastic": "🧴 Plastic",
    "trash": "🗑️ General Waste",
}

st.set_page_config(page_title="Trash Classification", layout="centered")

st.title("♻️ Trash Classification System")
st.markdown("Upload an image and let the AI identify the type of waste.")

st.sidebar.header("About")
st.sidebar.write(
    "This application uses a deep learning model (ResNet18) trained on the TrashNet dataset."
)
st.sidebar.subheader("Classes")
for v in CLASS_INFO.values():
    st.sidebar.write(f"- {v}")

st.subheader("Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            start_time = time.perf_counter()
            result = predict(image)
            latency_ms = (time.perf_counter() - start_time) * 1000

            if "error" in result:
                st.error(result["error"])
            else:
                pred_class = result["class"]
                confidence = result["confidence"]

                st.success("Prediction Complete!")
                st.caption("Inference time includes preprocessing and model prediction.")
                st.subheader("Result")
                st.markdown(f"### {CLASS_INFO.get(pred_class, pred_class)}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", f"{confidence:.2%}")
                with col2:
                    st.metric("Latency", f"{latency_ms:.2f} ms")
