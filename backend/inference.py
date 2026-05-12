import torch
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms

from models.resnet18 import get_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "checkpoints" / "best_resnet18_trashnet.pth"

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model(num_classes=6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict(image: Image.Image):
    try:
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

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(err, flush=True)
        return {"error": err}