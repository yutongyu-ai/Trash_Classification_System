"""Shared Streamlit page for both entry points: frontend/app.py (calls the
FastAPI backend over HTTP) and streamlit_app.py (calls inference.predict()
in-process). Each supplies its own predict_fn(image) -> dict and everything
else — layout, low-confidence handling — lives here once."""
import io

import streamlit as st
from PIL import Image

CLASS_INFO = {
    "cardboard": "📦 Cardboard",
    "glass": "🍾 Glass",
    "metal": "🥫 Metal",
    "paper": "📄 Paper",
    "plastic": "🧴 Plastic",
    "trash": "🗑️ General Waste",
}

# Below this, the model's top prediction is shown as a low-confidence
# warning with runner-up candidates instead of a plain success message —
# on a ~2,500-image dataset like TrashNet, a prediction this close to the
# 6-class random baseline (16.7%) isn't reliable enough to present as a
# confident answer.
CONFIDENCE_THRESHOLD = 0.5


def render_app(predict_fn):
    """predict_fn(image: PIL.Image.Image) -> dict with either an "error"
    key, or "class"/"confidence"/"top3" (and optionally "latency_ms")."""
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

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if not st.button("Predict"):
        return

    with st.spinner("Running inference..."):
        try:
            result = predict_fn(image)
        except Exception as e:
            st.error(f"Request error: {e}")
            return

    if "error" in result:
        st.error(result["error"])
        return

    pred_class = result["class"]
    confidence = result["confidence"]
    latency = result.get("latency_ms")
    top3 = result.get("top3", [])

    low_confidence = confidence < CONFIDENCE_THRESHOLD
    if low_confidence:
        st.warning("⚠️ Low confidence — the model isn't sure about this one.")
    else:
        st.success("Prediction Complete!")
    st.caption("Inference time includes preprocessing and model prediction.")

    st.subheader("Result")
    st.markdown(f"### {CLASS_INFO.get(pred_class, pred_class)}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confidence", f"{confidence:.2%}")
    with col2:
        if latency is not None:
            st.metric("Latency", f"{latency:.2f} ms")

    if low_confidence and len(top3) > 1:
        st.caption("Other likely candidates:")
        for item in top3[1:]:
            name = CLASS_INFO.get(item["class"], item["class"])
            st.write(f"- {name}: {item['confidence']:.1%}")


def image_to_png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
