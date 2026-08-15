import torch
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms

from models.resnet18 import get_model

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = BASE_DIR.parent / "checkpoints" / "best_resnet18_trashnet.pth"

# Fallback source when no local checkpoint is present (e.g. a fresh clone
# that hasn't run training): https://huggingface.co/tonghahaha/trashnet-resnet18
HF_REPO_ID = "tonghahaha/trashnet-resnet18"
HF_FILENAME = "best_resnet18_trashnet.pth"

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

_model = None


def _resolve_checkpoint_path() -> Path:
    if LOCAL_MODEL_PATH.exists():
        return LOCAL_MODEL_PATH
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME))


def load_model():
    """Load the model on first use (or force a reload) instead of at import
    time, so importing this module never fails just because a checkpoint
    isn't available yet."""
    global _model
    checkpoint_path = _resolve_checkpoint_path()
    m = get_model(num_classes=len(CLASSES))
    m.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    m.to(device)
    m.eval()
    _model = m
    return _model


def predict(image: Image.Image):
    try:
        model = _model if _model is not None else load_model()
        print(f"Using device: {device}")
        img = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)

        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        return {
            "class": CLASSES[pred],
            "confidence": float(probs[0][pred])
        }

    except Exception:
        import traceback
        err = traceback.format_exc()
        print(err, flush=True)
        return {"error": err}