import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"

# BACKEND inserted after ROOT so it ends up earlier in sys.path — makes
# `import main` resolve to backend/main.py, not the repo-root training script.
for path in (ROOT, BACKEND):
    sys.path.insert(0, str(path))
