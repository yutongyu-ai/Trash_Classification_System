import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"

# backend/inference.py and backend/main.py import sibling modules
# (`models`, `inference`) as top-level names, matching how they're actually
# run in production (Docker's WORKDIR is backend/, so those names resolve
# from there). BACKEND is inserted after ROOT so it ends up earlier in
# sys.path, making `import main` in tests resolve to backend/main.py (the
# FastAPI app) rather than the training script at the repo root.
for path in (ROOT, BACKEND):
    sys.path.insert(0, str(path))
