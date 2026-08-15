import io
import os
import requests
import streamlit as st
from PIL import Image

# Defaults to localhost, which is correct for the single-container HF Spaces
# deployment and for running both services on the same machine without
# Docker. docker-compose overrides this to the "backend" service hostname
# for the two-container local dev setup.
API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000/predict")

# ===== TrashNet Classes =====
CLASS_INFO = {
    "cardboard": "📦 Cardboard",
    "glass": "🍾 Glass",
    "metal": "🥫 Metal",
    "paper": "📄 Paper",
    "plastic": "🧴 Plastic",
    "trash": "🗑️ General Waste"
}

# ===== Page Config =====
st.set_page_config(
    page_title="Trash Classification",
    layout="centered"
)

# ===== Title =====
st.title("♻️ Trash Classification System")
st.markdown(
    "Upload an image and let the AI identify the type of waste."
)

# ===== Sidebar =====
st.sidebar.header("About")
st.sidebar.write(
    "This application uses a deep learning model (ResNet) trained on the TrashNet dataset."
)

st.sidebar.subheader("Classes")
for v in CLASS_INFO.values():
    st.sidebar.write(f"- {v}")

# ===== Upload Section =====
st.subheader("Upload Image")
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # ===== Show Image =====
    st.image(image, caption="Uploaded Image", width=300)

    # Convert to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    # ===== Predict Button =====
    if st.button("Predict"):
        with st.spinner("Running inference..."):

            try:
                files = {"file": ("image.png", img_bytes, "image/png")}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        pred_class = result["class"]
                        confidence = result["confidence"]
                        latency = result.get("latency_ms", None)

                        # ===== Result Section =====
                        st.success("Prediction Complete!")
                        st.caption("Inference time includes preprocessing and model prediction.")

                        st.subheader("Result")

                        # Pretty class display
                        display_name = CLASS_INFO.get(pred_class, pred_class)

                        st.markdown(f"### {display_name}")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Confidence", f"{confidence:.2%}")

                        with col2:
                            if latency is not None:
                                st.metric("Latency", f"{latency:.2f} ms")

                else:
                    st.error(f"Request failed with status code: {response.status_code}")

            except Exception as e:
                st.error(f"Request error: {e}")

