import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"

# backend/*.py import sibling modules as top-level names (matches Docker's
# WORKDIR=backend/ in production). Insert BACKEND after ROOT so it's
# earlier in sys.path, making `import main` resolve to backend/main.py
# (the FastAPI app), not the repo-root training script.
for path in (ROOT, BACKEND):
    sys.path.insert(0, str(path))
